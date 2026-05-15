# fitra-cam (Python 退避版)

> このディレクトリは Python による旧実装。本プロジェクトは C++/TensorRT への移行中で、新実装はリポジトリ直下の `cpp/` 配下。背景と段階計画は `docs/cpp-migration-plan.md` を参照。
> ここに残した Python 版は **数値リファレンス** と **緊急時のフォールバック** として保守する。

Jetson Orin Nano Super 上で、2 台の USB カメラから映像を取り込み、YOLOX で人検出 → RTMPose で 2D 17 keypoint 姿勢推定を回す最小実装。

- 取り込みは `cv2.CAP_V4L2` 直接 (GStreamer 非依存)
- 推論は ONNX Runtime 直叩き (rtmlib 非依存)
- CLI snapshot / FastAPI + WebSocket skeleton viewer / 30 秒録画オーバーレイ の 3 経路
- CUDA / TensorRT 切り替え可能 (`--device {auto,cpu,cuda,tensorrt}`)

## 動作確認済み環境

- Jetson Orin Nano Super / JetPack 6.2.1 / Python 3.10 / aarch64
- USB カメラ 2 台 (by-path で固定: `2.3:1.0` と `2.4:1.0`)
- `onnxruntime-gpu 1.23.0` (Jetson AI Lab `jp6/cu126` 配布)
- 重みは `rtmlib` 配布アーカイブの ONNX をそのまま流用:
  - 検出: `yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx`
  - 姿勢: `rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.onnx` (既定 / 反応重視)
  - 姿勢精度を上げたい場合は `--pose-model` で `rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.onnx` に差し替え
  - 既定パス: `~/.cache/rtmlib/hub/checkpoints/...`

## セットアップ

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv
chmod +x python/scripts/setup_jetson_env.sh
./python/scripts/setup_jetson_env.sh
```

`python/scripts/setup_jetson_env.sh` は `python/.venv` がなければ `--system-site-packages` で作り、`python/requirements-jetson.txt` を流し込んでから `cv2` / `numpy` / `onnxruntime` のバージョンと利用可能プロバイダを表示する。venv はリポジトリ直下ではなく `python/.venv` に置く (リポジトリ直下は C++ 移行が使う)。

GPU 実行用の `onnxruntime-gpu` を入れる場合は、NVIDIA forum から案内されている **Jetson AI Lab (`jp6/cu126`)** wheel を使う。JetPack 6.2.1 / Python 3.10 / aarch64 なら:

```bash
. python/.venv/bin/activate
python -m pip uninstall -y onnxruntime onnxruntime-gpu
PYTHONNOUSERSITE=1 python -m pip install --no-deps \
  https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

モデル ONNX は `rtmlib` を一時的に入れて取得するか、配布元から手動で `~/.cache/rtmlib/hub/checkpoints/` 配下に置く。本リポジトリのコードは ONNX を直接読むので `rtmlib` のインストールは不要。

## CLI: 2 カメラ snapshot

短い動作確認:

```bash
. python/.venv/bin/activate
python python/scripts/dual_rtmpose_cameras.py --device auto --max-frames 120 --save-every 30
```

`outputs/dual_rtmpose/<ts>/camX_NNNNNN.jpg` に annotated JPEG が保存される。

画面に出したい場合は `--display` を付ける (要 `DISPLAY`):

```bash
python python/scripts/dual_rtmpose_cameras.py --device auto --display
```

## Web: WebSocket skeleton viewer

```bash
python python/scripts/dual_rtmpose_web.py --device auto --host 0.0.0.0 --port 8000
```

ブラウザで `http://JETSON_IP:8000/` を開くと、映像なし・skeleton のみが Canvas で 2 ペイン描画される。WebSocket は最大 30Hz で bundle を publish する。

Web 上で見られる指標:

- camera ごとの `recv_fps` / `render_fps`
- camera ごとの `recent_pose_fps` / `avg_pose_fps`
- camera ごとの `pending` (取り込み済みで処理が追いついていないフレーム数)
- camera ごとの `stage_ms` (capture から推論完了までの 1 フレーム所要)
- `latency_ms` (capture から bundle publish までの大ざっぱな遅延)
- 全体の `bundle_seq`

推論を間引かずに publish 側だけ減らしたい場合は `--publish-every N`。

## 録画: 30 秒撮って overlay

```bash
python python/scripts/record_dual_rtmpose_overlay.py --device auto --seconds 30
```

`outputs/recorded_rtmpose/<ts>/` 配下に下記 5 本の MP4 が保存される:

- `raw_cam0.mp4` / `raw_cam1.mp4`: 推論前の録画。VideoWriter は **実測 fps** をメタデータに書く (USB 2.0 バス共有で 30fps 出ない場合があるため)
- `overlay_cam0.mp4` / `overlay_cam1.mp4`: 各カメラに skeleton を重ねたもの
- `overlay_side_by_side.mp4`: 2 カメラ横並びの目視確認用

## TensorRT 実行

```bash
python python/scripts/dual_rtmpose_web.py \
  --device tensorrt \
  --trt-cache-dir outputs/tensorrt_engines \
  --trt-fp16 \
  --trt-warmup-frames 30 \
  --log-every 10
```

初回起動では engine build に数分かかる (実測 約 7 分)。同じコマンドを再実行すると cache 済みの engine で立ち上がる。

モデル別に provider を切り替える `--trt-models {det,pose,all}` を用意した:

- `det` (既定): YOLOX のみ TensorRT、RTMPose は CUDA EP のまま (pose 側 TRT で過去に drift が確認されているため)
- `pose`: RTMPose のみ TensorRT、YOLOX は CUDA EP
- `all`: 両方 TensorRT

engine cache はモデル / ONNX Runtime / TensorRT / FP16 設定が変わると無効になる:

```bash
rm -rf outputs/tensorrt_engines
```

比較ベンチ:

```bash
python python/scripts/dual_rtmpose_web.py --device cuda --log-every 10
python python/scripts/dual_rtmpose_web.py --device tensorrt --trt-models det --trt-fp16 --log-every 10
```

見る項目は cam0/cam1 の `recent_pose_fps`、`stage_ms`、`tegrastats` の温度・GR3D・throttle 兆候、keypoint 品質の目視劣化。

## CLI フラグ早見表

`python/scripts/pose_pipeline.py::add_common_args` で 3 ツール共通の引数を定義している。

| flag                  | 既定                                           | 説明                                                |
|-----------------------|-----------------------------------------------|----------------------------------------------------|
| `--cam0` / `--cam1`   | `2.3:1.0` / `2.4:1.0` by-path                 | 入力カメラのデバイスパス                            |
| `--width`/`--height`  | 640 / 480                                     | VGA。Jetson 上の品質優先既定                        |
| `--fps`               | 30                                            | 要求 fps。USB 2.0 バス共有で実測は下がる            |
| `--fourcc`            | `MJPG`                                        | V4L2 fourcc                                         |
| `--device`            | `auto`                                        | `auto`/`cpu`/`cuda`/`tensorrt`                      |
| `--det-model`         | `yolox_tiny ...onnx`                          | YOLOX ONNX                                          |
| `--pose-model`        | `rtmpose-s ...onnx`                           | RTMPose ONNX (精度優先で m に切替可)                |
| `--det-score`         | 0.5                                           | 検出スコア閾値                                      |
| `--det-frequency`     | 10                                            | 何フレームごとに YOLOX を回すか                     |
| `--multi-person`      | -                                             | 既定は single-person。複数追跡時のみ指定            |
| `--kp-thr`            | 0.3                                           | 描画用 keypoint スコア閾値                          |
| `--log-every`         | 10.0 s                                        | stats 行の間隔                                      |
| `--trt-cache-dir`     | `outputs/tensorrt_engines`                    | TensorRT engine cache                               |
| `--trt-fp16`          | -                                             | FP16 engine                                         |
| `--trt-warmup-frames` | 0                                             | 起動直後にダミー入力で N 回 forward                 |
| `--trt-models`        | `det`                                         | TRT 化対象 (`det`/`pose`/`all`)                     |

### `dual_rtmpose_cameras.py` 専用

- `--max-frames N`: 両カメラ合計の処理フレーム数で停止 (0 で無限)
- `--save-every N`: N フレームごとに annotated JPEG を保存 (0 で無効)
- `--display`: OpenCV imshow 表示
- `--output-dir outputs/dual_rtmpose`: 保存先

### `dual_rtmpose_web.py` 専用

- `--host` / `--port`
- `--publish-every N`: 推論は毎フレーム、WS publish だけ間引く

### `record_dual_rtmpose_overlay.py` 専用

- `--seconds`: 録画長さ
- `--output-dir outputs/recorded_rtmpose`: 保存先

## 確認スクリプト

セットアップ後の自己確認:

```bash
. python/.venv/bin/activate
PYTHONNOUSERSITE=1 python - <<'PY'
import cv2
import onnxruntime as ort

print("cv2:", cv2.__version__, cv2.__file__)
print("providers:", ort.get_available_providers())
print("device:", ort.get_device())
PY
```

## メモ

- 取り込みは `cv2.CAP_V4L2` を直接使う。GStreamer は明示的に使わない
- `python3-opencv` (apt) が `cv2.CAP_V4L2` を提供している前提
- `PYTHONNOUSERSITE=1` を runtime コマンド側で立てるので、user-site の OpenCV wheel が居ても無視される
- 既定の取り込みは `MJPG` 640x480 30fps。2 カメラを USB 2.0 ハブで共有している場合、実測は 〜15fps × 2 になる
- single-person 既定。`--multi-person` で複数 bbox 全部に pose を回す
- backend / device が `auto` のときは `CUDAExecutionProvider` が見えれば GPU、なければ CPU
- TensorRT は明示指定のみ。`auto` では選ばない (初回 engine build が数分かかるため)
- pose 側を TensorRT で動かすと keypoint drift が出る場合がある (既存観察)。`--trt-models det` (既定) は YOLOX のみ TRT で動かす
- 出力ディレクトリ:
  - `outputs/dual_rtmpose/<ts>/`
  - `outputs/recorded_rtmpose/<ts>/`
  - `outputs/tensorrt_engines/`
