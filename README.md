# fitra-cam

Jetson Orin Nano Super 上で、2 台の USB カメラから映像を取り込み、RTMPose ベースの 2D 姿勢推定を回すための最小実装。

## セットアップ

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv gstreamer1.0-tools \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  gstreamer1.0-libav
chmod +x scripts/setup_jetson_env.sh
./scripts/setup_jetson_env.sh
```

このセットアップは **Jetson Orin Nano Super / JetPack 6.2.1 / Python 3.10** を前提にしている。

- `.venv` は `--system-site-packages` で作り、apt 版 `python3-opencv` の **GStreamer 有効 build** を使う
- `rtmlib` は `--no-deps` で入れ、pip の OpenCV wheel 混入を防ぐ
- `./scripts/setup_jetson_env.sh` は既定で **CPU 実行**、`ORT_VARIANT=gpu` で Jetson AI Lab の GPU wheel を入れられる
- runtime script は `PYTHONNOUSERSITE=1` を使うので、user-site の `opencv-python` があっても実行時は無視する

`./scripts/setup_jetson_env.sh` は、`.venv` 内の競合パッケージを掃除してから依存関係を入れ直し、最後に `cv2` の GStreamer と `onnxruntime` provider を確認する。

GPU 実行用の `onnxruntime-gpu` を入れる場合は、NVIDIA forum から案内されている **Jetson AI Lab (`jp6/cu126`)** の wheel を使う。JetPack 6.2.1 / Python 3.10 / aarch64 なら、次で入る:

```bash
ORT_VARIANT=gpu ./scripts/setup_jetson_env.sh
```

または手動で:

```bash
. .venv/bin/activate
python -m pip uninstall -y onnxruntime onnxruntime-gpu
PYTHONNOUSERSITE=1 python -m pip install --no-deps \
  https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

## 実行

既定では、今回の実機で確認済みの 2 台の `by-path` カメラを使う。

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_cameras.py --device auto --max-frames 120 --save-every 30
```

既定値は **VGA (`640x480`) / 30fps / `balanced` / `det-frequency 10`** にしてあり、Jetson 上での追従性と精度を優先しつつ 30fps 目標で動かす。

画面表示したい場合:

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_cameras.py --device auto --display
```

Web アプリで skeleton だけ表示したい場合:

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_web.py --device auto --host 0.0.0.0 --port 8000
```

ブラウザで `http://JETSON_IP:8000/` を開くと、2 カメラ分を別ペインで表示する。

30 秒間 2 カメラを録画してから、保存動画に RTMPose の skeleton を重ねた確認用動画を作る場合:

```bash
. .venv/bin/activate
python scripts/record_dual_rtmpose_overlay.py --device auto --seconds 30
```

出力は `outputs/recorded_rtmpose/YYYYMMDD_HHMMSS/` 配下に保存される。

- `raw_cam0.mp4` / `raw_cam1.mp4`: 推論前の録画
- `overlay_cam0.mp4` / `overlay_cam1.mp4`: 各カメラの姿勢推定オーバーレイ
- `overlay_side_by_side.mp4`: 2 カメラ横並びの目視確認用動画

## Jetson で低遅延化する既定方針

- 既定の capture 解像度は **VGA (`640x480`)**
- pose model は **`--mode balanced`** を既定にして、Jetson では 30fps 安定運用を目標にする
- backend / device は **`auto`** で、起動時に **`onnxruntime + CUDAExecutionProvider` → `opencv + CUDA` → CPU** の順に選ぶ
- detector は **`--det-frequency 10`** を既定にして、tracker で中間フレームを補完する
- ターゲットは1人だけの前提なので、既定で `--single-person` とし、最大bboxの1人だけをpose推論する
- GStreamer パイプラインは **leaky queue + latest-frame-first appsink** で古いフレームを溜めにくくしている
- 2 台の capture は別 thread で最新フレームだけ保持し、未処理フレームがある間は次の read を抑制する
- 推論は round-robin で公平に回して、片側だけが GPU を占有しない構成にしている
- 既定で 10 秒ごとに per-camera の `avg` / `recent` / `stage` / `pending` をログ出力する

JetPack 6.2.1 では **Jetson AI Lab (`jp6/cu126`) 配布の `onnxruntime-gpu` wheel** が使える。今回実機では `onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl` を導入し、`CUDAExecutionProvider` / `TensorrtExecutionProvider` / `CPUExecutionProvider` が見えることを確認した。

JetPack 6.2.1 上で CUDA 実行を明示したい場合:

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_web.py --backend onnxruntime --device cuda
```

`--device auto` でも、`CUDAExecutionProvider` が見えていれば自動で GPU 側を選ぶ。起動できない場合は、現在の Python 環境に **JetPack 6.2.1 と互換のある CUDA 対応 onnxruntime** が入っていないか、provider 依存ライブラリの解決に失敗している。

TensorRT Execution Provider を使う場合は、初回起動で RTMDet / RTMPose の engine build が走るため、数分止まったように見えることがある。`--device auto` は当面 TensorRT を自動選択しないので、明示的に指定する。

```bash
. .venv/bin/activate
python scripts/dual_rtmpose_web.py \
  --device tensorrt \
  --trt-cache-dir outputs/tensorrt_engines \
  --trt-fp16 \
  --trt-models det \
  --trt-warmup-frames 30 \
  --log-every 10
```

TensorRT 実行時は内部で `onnxruntime` backend を使い、provider は `TensorrtExecutionProvider` → `CUDAExecutionProvider` → `CPUExecutionProvider` の順で設定する。cache 生成後は同じコマンドを再起動して、2 回目以降の `recent fps` と `stage total` を評価する。モデル、ONNX Runtime、TensorRT、JetPack、FP16設定を変えた場合は、古い engine cache を消して作り直す。

TensorRT EPで RTMPose 側を実行すると keypoint / bbox が追従しないことがあったため、`--device tensorrt` の既定は `--trt-models det` とし、YOLOX detector だけTensorRT、RTMPoseはCUDAExecutionProviderで動かす。`--trt-models all` / `--trt-models pose` は切り分け用に残している。

cam1 だけ bbox が暴走する、または数秒おきに大きく外れる場合は tracker drift の可能性がある。まず `--det-frequency 1` か `--no-tracking` で確認する。
複数人検証が必要なときだけ `--multi-person` を指定する。

```bash
rm -rf outputs/tensorrt_engines
```

比較ベンチは同じ解像度と `--mode balanced --det-frequency 10` で行う。

```bash
python scripts/dual_rtmpose_web.py --device cuda --log-every 10
python scripts/dual_rtmpose_web.py --device tensorrt --trt-models det --trt-warmup-frames 30 --log-every 10
python scripts/dual_rtmpose_web.py --device tensorrt --trt-models det --log-every 10
python scripts/dual_rtmpose_web.py --device tensorrt --trt-models pose --log-every 10
python scripts/dual_rtmpose_web.py --device tensorrt --trt-models det --no-tracking --log-every 10
```

見る項目は、cam0/cam1 の `recent pose fps`、`stage total`、`tegrastats` の温度・GR3D・throttle兆候、keypoint品質の目視劣化。

セットアップ後の確認コマンド:

```bash
. .venv/bin/activate
PYTHONNOUSERSITE=1 python - <<'PY'
import cv2
import onnxruntime as ort

print("cv2:", cv2.__version__, cv2.__file__)
print("providers:", ort.get_available_providers())
print("device:", ort.get_device())
print(next(line.strip() for line in cv2.getBuildInformation().splitlines() if "GStreamer:" in line))
PY
```

## Web 表示で見られる指標

- camera ごとの `recv fps` / `render fps`
- camera ごとの `avg pose fps` / `recent pose fps` / `avg publish fps`
- camera ごとの `pending frames`
- camera ごとの `stage total` と簡易 `latency`
- runner の状態と最新 bundle sequence

`--publish-every N` を使うと、推論は毎フレーム続けながら Web への publish だけ間引ける。

## メモ

- 取り込みは **apt 版 `python3-opencv` + OpenCV CAP_GSTREAMER** を前提にしている
- `PYTHONNOUSERSITE=1` なしで手元確認すると、user-site の `opencv-python*` を拾って **GStreamer 無効 build** になることがある
- `.venv` に `opencv-python*` / `opencv-contrib-python*` が入っていると apt 版より優先されるので入れない
- もし `.venv` に誤って OpenCV wheel が入った場合は `. .venv/bin/activate && python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless` で外す
- 既定の GStreamer デコーダは `nvv4l2decoder`。`nvjpegdec` や `jpegdec` も `--gst-decoder` で選べる
- 2D 姿勢推定は `rtmlib + onnxruntime`
- 既定値は Jetson での 30fps 安定運用を優先して `--width 640 --height 480 --mode balanced --det-frequency 10`
- 既定では `--single-person` で最大bboxの1人だけを処理する
- backend / device が `auto` のときは、利用可能なら GPU 実行を優先し、なければ明示メッセージ付きで CPU にフォールバックする
- JetPack 6.2.1 では、**Jetson AI Lab の `jp6/cu126` wheel** または JetPack 6.2.1 向けに自前 build した ORT GPU wheel を使う
- 短時間スモークテストでは、2 台同時推論と annotated 画像保存を確認済み
- 出力画像は `outputs/dual_rtmpose/` に保存される
- Web 表示は **映像配信なし / skeleton-only** で、FastAPI + WebSocket + Canvas 2D を使う
