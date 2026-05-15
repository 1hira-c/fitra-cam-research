# fitra-cam

Jetson Orin Nano Super 上で **3 USB カメラ同時 / リアルタイム** に YOLOX 人検出 + RTMPose 17 keypoint 2D 姿勢推定を回す。実装は TensorRT C++ + Jetson Multimedia API で zero-copy。

このリポジトリは現在 **C++ 移行中**:

- `cpp/` — TensorRT C++ + V4L2/NVJPEG zero-copy 実装 (進行中、本リポジトリの主軸)
- `python/` — 旧 Python 実装。動作する **数値リファレンス** と **緊急時のフォールバック** として残す
- `web/dual_rtmpose/` — Canvas ベースの skeleton viewer。Python / C++ どちらの WebSocket サーバからも serve できる
- `docs/cpp-migration-plan.md` — 段階計画とアーキテクチャ
- `docs/research/*.md` — 設計判断の根拠 (日本語、未実装事項を含む)

Python 版がそのまま動くのは Phase 0 完了時点の合意。`cpp/` build と入れ替わらないよう、Python 側の venv は `python/.venv` に閉じる。

## C++ ビルド (Phase 0)

```bash
sudo apt install -y cmake g++ libnvinfer-dev libnvinfer-plugin-dev nvidia-jetpack
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build -j
./cpp/build/main --help
```

最初の確認 (Phase 0 完了基準): `./cpp/build/main --help` がリンクして起動し、`cudaGetDeviceCount()` と `nvinfer1::createInferRuntime()` を呼べる。

依存:

| 用途 | 出所 |
|------|------|
| TensorRT 10.3 | apt (`libnvinfer-dev` ほか) |
| CUDA 12.6 | apt (`nvidia-jetpack` 同梱) |
| Jetson Multimedia API | `/usr/src/jetson_multimedia_api/` |
| OpenCV 4.x | apt (`libopencv-dev`) |
| Crow / spdlog / nlohmann_json / CLI11 / readerwriterqueue | CMake `FetchContent` (全て header-only) |

## Python 版を使う

数値リファレンスや旧来の Web UI を回す場合:

```bash
./python/scripts/setup_jetson_env.sh
. python/.venv/bin/activate
python python/scripts/dual_rtmpose_web.py --device auto --host 0.0.0.0 --port 8000
```

詳細は `python/README.md`。

## 評価データ

C++ 実装の correctness/bench は `outputs/recorded_rtmpose/20260515_064342/` の MP4 を入力に使う:

- `raw_cam0.mp4` / `raw_cam1.mp4` — 推論前の生録画 (C++ パイプの入力)
- `overlay_cam0.mp4` / `overlay_cam1.mp4` — Python ORT の出力 (リファレンス)

correctness check は Python ORT の keypoint 出力と C++ TRT の出力を frame 単位で diff し、bbox IoU > 0.99 / keypoint L2 < 1px を合格基準とする (`docs/cpp-migration-plan.md` Phase 1)。

## Jetson 共通の制約

`~/CLAUDE.md` に既出。要点だけ:

- `apt python3-opencv` / `libnvinfer-dev` を pip wheel で上書きしない
- `.engine` はデバイス固有。コミットしない (`models/` も同様)
- カメラは `/dev/v4l/by-path/...` で固定 (再起動・抜き差しで `/dev/video*` は順序が変わる)
- ベンチ前に `sudo nvpmodel -m 0 && sudo jetson_clocks`
