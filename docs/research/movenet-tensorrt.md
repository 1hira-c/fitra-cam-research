# MoveNet Thunder 再検証メモ

## 結論

- 現時点では **MoveNet Thunder は採用候補から外す**。CPU / CUDA / TensorRT の実行先に関わらず、Web preview 上で人体として成立する skeleton が得られていない。
- 同じカメラ入力で RTMPose は人体を正常に認識できているため、入力映像や Web 描画ではなく MoveNet モデル実行経路の問題として扱う。
- MoveNet を再開する場合は、速度や TensorRT 化ではなく、まず公式 SavedModel / TFLite 単体で保存フレームから人体認識できることを確認する。
- MoveNet Thunder は COCO 17 点の単一人物モデルなので、正しく動けば現行 Web payload の keypoint schema には合わせやすい。
- 一方で、RTMPose の top-down detector + tracker とは違い、MoveNet singlepose は **中央または前フレーム crop にいる 1 人**へ寄る。複数人や人物 ID 維持は初期スコープから外す。
- Jetson の再検証は **SavedModel/TFLite baseline を作ってから ONNX 化し、必要になった場合だけ ONNX Runtime CUDA や TensorRT を評価**する流れに戻す。
- TFLite FP16 モデルを TensorRT が直接読むわけではないため、TensorRT FP16 には ONNX 変換ステップが必要になる。

## モデル特性

MoveNet は Lightning と Thunder の 2 系統があり、Thunder は高精度寄り、Lightning は低レイテンシ寄り。TensorFlow Lite の公式 benchmark では Thunder FP16 が 12.6 MB / mAP 72.0、Lightning FP16 が 4.8 MB / mAP 63.0 とされている。

今回の目的では、RTMPose の精度から落とし過ぎたくないため Thunder を最初の再検証対象にしていた。ただし実機では人体認識が成立していないため、Lightning への差し替えや TensorRT 化より前に、公式 runtime での単体推論を確認する。

### 入出力

- 入力: `1 x 256 x 256 x 3`
- 入力型:
  - SavedModel: `int32`
  - TFLite: `uint8`
- 出力: `1 x 1 x 17 x 3`
- 出力座標: `[y, x, score]` の正規化座標
- keypoint 配列: COCO 17 点で、現行 `COCO_KEYPOINT_NAMES` と同じ順序

出力は現行 payload 用に `[x_px, y_px]` と `scores` へ変換できる。現行の Kalman smoothing もそのまま再利用可能。

## RTMPose との差分

### 良い点

- detector を別に走らせないため、RTMDet + RTMPose の top-down 構成より推論段数が減る。
- COCO 17 点なので Web UI、snapshot、payload schema の変更が少ない。
- `single_person=True` 前提の現行運用とは相性がよい。

### 注意点

- 実機確認では、MoveNet ONNX は人体を認識できていない。壊れた skeleton を preview / payload に流さないよう、実装側では異常な正規化座標を抑制する。
- MoveNet singlepose は複数人物には向かない。画面中央または crop にいる人物を追う前提になる。
- crop 更新ロジックが精度と安定性に強く効く。初期フレームは全体、以降は前フレームの torso/身体範囲から square crop を更新する。
- 現行 `PoseTracker` の detection frequency / tracking パラメータは MoveNet backend では意味が薄くなるため、CLI 表示上の扱いを整理する必要がある。
- TensorRT 化の成否は ONNX 変換後の operator support に依存する。現時点の実装では MoveNet の TensorRT 実行は無効化し、まず ONNX Runtime CPU/CUDA baseline を正しくする。

## TensorRT FP16 経路

### 推奨順

1. 代表フレームを保存し、RTMPose で人体認識できる入力であることを確認する。
2. TF Hub SavedModel または公式 TFLite runtime で、その保存フレームから人体を認識できることを確認する。
3. 公式 runtime で成立した場合だけ `tf2onnx` で SavedModel を ONNX へ変換する。
4. Jetson の `onnxruntime-gpu` で `CPUExecutionProvider` / `CUDAExecutionProvider` baseline を測る。
5. skeleton が人体として安定して描画できることを確認してから、必要なら別タスクで TensorRT EP を検討する。

SavedModel からの変換を優先する理由は、TFLite からの ONNX 変換よりグラフ検証と修正がしやすいため。TFLite FP16 は baseline として有用だが、Jetson で TensorRT FP16 へ寄せる主経路にはしない。

### 変換コマンド

```bash
python -m pip install tensorflow tensorflow-hub tf2onnx onnx
python scripts/export_movenet_onnx.py \
  --model thunder \
  --saved-model-dir outputs/models/movenet_thunder_saved_model \
  --output outputs/models/movenet_thunder.onnx \
  --opset 17
```

Jetson に TensorFlow 一式を入れたくない場合は、開発機でこのスクリプトを実行し、変換済み ONNX だけを Jetson 側へ置く。

## 実装案

### Backend adapter

`dual_rtmpose_core.py` に RTMPose 固有の `PoseTracker` を直接保持しているため、次は小さい adapter 境界を作る。

- `PoseEstimator` protocol: `__call__(frame) -> tuple[keypoints, scores]`
- `RtmposeEstimator`: 既存 `PoseTracker` wrapper
- `MovenetEstimator`: MoveNet ONNX/TFLite wrapper

`CameraRuntime.tracker` は `estimator` に読み替える。payload 以降は現行の `keypoints, scores` 形式を保つ。

### MoveNet preprocessing

初期実装は TensorFlow Hub tutorial の `crop_and_resize` と座標復元に合わせる。

1. 初期 crop region は画像全体を square にする正規化 box として持つ。
2. OpenCV で `tf.image.crop_and_resize` 相当の入力を作り、画像外は 0 padding にする。
3. ONNX Runtime CPU/CUDA で推論する。
4. 出力 `[y, x, score]` を crop region 経由で元画像 pixel 座標へ戻す。
5. 既定では full-frame crop を毎回使い、必要なときだけ `--movenet-crop-tracking` で前フレーム keypoint から crop を更新する。

crop 更新は公式 tutorial の考え方を移植する。ただし `tf.image.crop_and_resize` には依存せず、OpenCV の remap と座標変換で実装する。

## 評価計画

### 比較条件

- RTMPose 現状: `--device cuda` と `--device tensorrt --trt-models det`
- MoveNet 公式 SavedModel / TFLite baseline
- 公式 baseline で人体認識できた場合のみ MoveNet Thunder ONNX CPU / CUDA
- 必要なら MoveNet Lightning 公式 baseline

### 測るもの

- 人体として成立する skeleton が出るか
- keypoint の正規化座標が `[0, 1]` の範囲に収まるか
- camera ごとの `stage_ms.pose`
- `recent_fps` / `avg_fps`
- dual camera 同時実行時の drop rate
- keypoint jitter
- 主要姿勢での欠損率
- CPU/CUDA provider と `stage_ms.pose`

### 合格ライン

- dual camera で Web publish 目標 FPS に届くこと。
- 主要 17 点の欠損が RTMPose より許容範囲に収まること。
- crop 更新で人物が画面端に移動しても追従できること。
- `--device cuda` で `CUDAExecutionProvider` が有効になっていることを provider metadata で確認できること。

## 当面の作業順

1. ライブ運用と Web preview は RTMPose に固定する。
2. MoveNet から壊れた keypoint が出た場合は preview / payload へ流さない。
3. 代表フレームに対して公式 SavedModel / TFLite baseline を作る。
4. 公式 baseline で人体認識できた場合だけ ONNX Runtime CPU / CUDA へ戻す。

## 参考

- TensorFlow Lite pose estimation overview: https://www.tensorflow.org/lite/examples/pose_estimation/overview
- TensorFlow Hub MoveNet tutorial: https://www.tensorflow.org/hub/tutorials/movenet
- TensorFlow Blog MoveNet/TFLite edge overview: https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html
- tf2onnx: https://github.com/onnx/tensorflow-onnx
- NVIDIA TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
