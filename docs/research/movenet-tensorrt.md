# MoveNet Thunder / TensorRT FP16 検討

## 結論

- RTMPose が目標 FPS に届かない場合の次候補は、まず **MoveNet Thunder singlepose** を評価する。
- MoveNet Thunder は COCO 17 点の単一人物モデルなので、現行 Web payload の keypoint schema には合わせやすい。
- 一方で、RTMPose の top-down detector + tracker とは違い、MoveNet singlepose は **中央または前フレーム crop にいる 1 人**へ寄る。複数人や人物 ID 維持は初期スコープから外す。
- Jetson の最短評価は **SavedModel/TFLite baseline を作ってから ONNX 化し、ONNX Runtime TensorRT EP または TensorRT engine で FP16 評価**する流れがよい。
- TFLite FP16 モデルを TensorRT が直接読むわけではないため、TensorRT FP16 には ONNX 変換ステップが必要になる。

## モデル特性

MoveNet は Lightning と Thunder の 2 系統があり、Thunder は高精度寄り、Lightning は低レイテンシ寄り。TensorFlow Lite の公式 benchmark では Thunder FP16 が 12.6 MB / mAP 72.0、Lightning FP16 が 4.8 MB / mAP 63.0 とされている。

今回の目的では、RTMPose の精度から落とし過ぎたくないため **Thunder FP16 を第一候補**にする。ただし Thunder でも dual camera で目標 FPS に届かない場合は、Lightning FP16 を同じ adapter で差し替えて比較する。

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

- MoveNet singlepose は複数人物には向かない。画面中央または crop にいる人物を追う前提になる。
- crop 更新ロジックが精度と安定性に強く効く。初期フレームは全体、以降は前フレームの torso/身体範囲から square crop を更新する。
- 現行 `PoseTracker` の detection frequency / tracking パラメータは MoveNet backend では意味が薄くなるため、CLI 表示上の扱いを整理する必要がある。
- TensorRT 化の成否は ONNX 変換後の operator support に依存する。最初から engine 前提にせず、ONNX Runtime CUDA baseline を作ってから TensorRT EP を試す。

## TensorRT FP16 経路

### 推奨順

1. TF Hub SavedModel を取得する。
2. `tf2onnx` で SavedModel を ONNX へ変換する。
3. Jetson の `onnxruntime-gpu` で `CUDAExecutionProvider` baseline を測る。
4. `TensorrtExecutionProvider` + FP16 + engine cache を有効化して測る。
5. TensorRT EP が不安定なら `trtexec --fp16` で静的 shape engine を作る。

SavedModel からの変換を優先する理由は、TFLite からの ONNX 変換よりグラフ検証と修正がしやすいため。TFLite FP16 は baseline として有用だが、Jetson で TensorRT FP16 へ寄せる主経路にはしない。

### 変換候補コマンド

```bash
python -m pip install tensorflow tensorflow-hub tf2onnx onnx
python - <<'PY'
from pathlib import Path

import tensorflow as tf
import tensorflow_hub as hub

output_dir = Path("outputs/models/movenet_thunder_saved_model")
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
print(model.signatures["serving_default"].structured_input_signature)
print(model.signatures["serving_default"].structured_outputs)
tf.saved_model.save(
    model,
    str(output_dir),
    signatures={"serving_default": model.signatures["serving_default"]},
)
print(output_dir)
PY

python -m tf2onnx.convert \
  --saved-model outputs/models/movenet_thunder_saved_model \
  --opset 17 \
  --output outputs/models/movenet_thunder.onnx
```

実際には `hub.load()` だけでは SavedModel directory が残らない場合があるため、Jetson ではなく開発機で export 手順を固定する。Jetson 側は変換済み ONNX と engine cache を受け取る形にした方が再現性が高い。

## 実装案

### Backend adapter

`dual_rtmpose_core.py` に RTMPose 固有の `PoseTracker` を直接保持しているため、次は小さい adapter 境界を作る。

- `PoseEstimator` protocol: `__call__(frame) -> tuple[keypoints, scores]`
- `RtmposeEstimator`: 既存 `PoseTracker` wrapper
- `MovenetEstimator`: MoveNet ONNX/TFLite wrapper

`CameraRuntime.tracker` は `estimator` に読み替える。payload 以降は現行の `keypoints, scores` 形式を保つ。

### MoveNet preprocessing

初期実装は次の順序でよい。

1. frame 全体を letterbox して 256x256 にする。
2. ONNX/CUDA で推論する。
3. 正規化 `[y, x]` を元画像 pixel 座標へ戻す。
4. torso が十分見えていれば次フレーム crop を更新する。
5. crop が壊れたら全体 crop に戻す。

crop 更新は公式 tutorial の考え方を移植する。ただし `tf.image.crop_and_resize` には依存せず、OpenCV の resize と座標変換で実装する。

## 評価計画

### 比較条件

- RTMPose 現状: `--device cuda` と `--device tensorrt --trt-models det`
- MoveNet Thunder ONNX CUDA
- MoveNet Thunder ONNX TensorRT FP16
- 必要なら MoveNet Lightning ONNX TensorRT FP16

### 測るもの

- camera ごとの `stage_ms.pose`
- `recent_fps` / `avg_fps`
- dual camera 同時実行時の drop rate
- keypoint jitter
- 主要姿勢での欠損率
- TensorRT engine 初回 build 時間と cache 再利用後の latency

### 合格ライン

- dual camera で Web publish 目標 FPS に届くこと。
- 主要 17 点の欠損が RTMPose より許容範囲に収まること。
- crop 更新で人物が画面端に移動しても追従できること。
- TensorRT EP fallback が起きていないことを provider metadata で確認できること。

## 当面の作業順

1. `PoseEstimator` 境界を追加し、既存 RTMPose を wrapper 化する。
2. MoveNet Thunder ONNX model の取得・変換手順を `scripts/` か docs に固定する。
3. ONNX Runtime CUDA の MoveNet backend を追加する。
4. TensorRT FP16 provider options を既存 RTMPose のものと共通化する。
5. Jetson で short run benchmark を取り、RTMPose と同じ summary 形式で比較する。

## 参考

- TensorFlow Lite pose estimation overview: https://www.tensorflow.org/lite/examples/pose_estimation/overview
- TensorFlow Hub MoveNet tutorial: https://www.tensorflow.org/hub/tutorials/movenet
- TensorFlow Blog MoveNet/TFLite edge overview: https://blog.tensorflow.org/2021/08/pose-estimation-and-classification-on-edge-devices-with-MoveNet-and-TensorFlow-Lite.html
- tf2onnx: https://github.com/onnx/tensorflow-onnx
- NVIDIA TensorRT documentation: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html
