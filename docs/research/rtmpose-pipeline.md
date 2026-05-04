# RTMPose 2D 推論パイプライン

## 結論

- Jetson での初期実装は **RTMDet + RTMPose の top-down 構成**が扱いやすい。
- ただし毎フレーム検出は重いので、**検出は間引き、間のフレームは tracker で補完**する方針がよい。
- TensorRT 化は **FP16 を第一候補**にし、INT8 は十分な評価と校正データがあるときだけ検討する。
- 初期モデルは **RTMPose-s か RTMPose-m、入力は 256x192 固定**から始めるのが妥当。

## モデル選定

### RTMPose の目安

OpenMMLab 公開情報では、RTMPose は t/s/m/l で速度と精度が分かれる。

| モデル | COCO AP の目安 | 使いどころ |
| --- | --- | --- |
| t | 最速、精度は控えめ | とにかく速度優先 |
| s | 良い初期点 | Jetson の初期検証向け |
| m | 精度寄りの実用候補 | 人数が少なめなら有力 |
| l / l-384 | 高精度 | Jetson では初期候補にしない |

今回の用途では、複数カメラ・3D 前段であることを考えると **s → m の順で評価**するのが自然。

## 推論パイプライン

### 推奨構成

1. カメラ入力
2. 人物検出 (`RTMDet` など)
3. tracker で人物 ID 維持
4. tracked bbox を crop
5. RTMPose で 2D keypoint 推論
6. timestamp 付きで多視点統合へ渡す

### スケジューリング

- **検出**: 毎フレームではなく `N` フレームごと
- **追跡**: 毎フレーム
- **姿勢推定**: tracked person のみ毎フレームまたは準毎フレーム

この構成にすると、人物検出の負荷を抑えつつ 2D keypoint の時間密度を維持しやすい。

## Jetson 向け実装方針

### 開発初期

- PyTorch / MMPose でまず成立確認
- モデル入力は **固定解像度**
- batch は **1 ベース**

### 最適化段階

v1 は既存 `rtmlib` の ONNX モデルを使ったまま、**ONNX Runtime TensorRT Execution Provider** を接続する。CLI では `--device tensorrt` を明示指定し、`--device auto` は初回 engine build の停止時間を避けるため CUDA 既定のままにする。

- `balanced` preset のまま `TensorrtExecutionProvider` を使う
- 実機確認では RTMPose 側のTensorRT EP出力が不安定だったため、v1の実用設定は **YOLOX detectorのみTensorRT、RTMPoseはCUDAExecutionProvider** とする
- provider fallback は `TensorrtExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider`
- FP16 と engine/timing cache を有効化する
- cache 生成後に再起動して CUDA baseline と比較する

v2 は **MMDeploy 経由の TensorRT engine 化**として別フェーズに分ける。

- RTMDet / RTMPose を MMDeploy で ONNX → TensorRT engine に変換する
- **静的 shape** の engine を作る
- CUDA baseline、ORT TensorRT EP、MMDeploy TensorRT engine の3系統で精度と速度を比較する
- 優先順は **FP16 → 必要なら INT8**

### FP16 を優先する理由

- keypoint localization は量子化誤差に敏感
- FP16 は精度劣化が小さく、Jetson でも速度改善を取りやすい
- INT8 は速いが、ポーズ精度の落ち方を実データで確認しないと危険

## オンライン推論上の制約

- MMPose の 3D demo / online 系では、**online モード時は未来フレームを使う時系列補助が使えない**。
- つまりリアルタイム構成では、単フレームに近い 2D 推論品質で戦う前提になる。
- そのため 3D 側での平滑化や補間も最初から考えておいた方がよい。

## 最初の成立条件

- 1 カメラで RTMDet + RTMPose が安定動作する
- 256x192 入力でレイテンシと keypoint 品質を評価できる
- ORT TensorRT EP の FP16 + cache 実行を `--trt-models det` で評価できる
- MMDeploy TensorRT engine 版へ差し替え可能な構成になっている
- bbox, keypoints, timestamp を次段へ渡せる

## 当面の推奨

1. **RTMPose-s + 256x192** で単カメラ成立
2. RTMDet は間引き実行、間フレームは tracker
3. ORT TensorRT EP で FP16 + engine cache を評価
4. MMDeploy TensorRT engine 化を別フェーズで評価
5. 複数カメラ化して同期済みフレームへ適用

## 参考

- RTMPose README / pipeline performance
- MMPose inference / 3D demo docs
- MMDeploy TensorRT deployment docs
