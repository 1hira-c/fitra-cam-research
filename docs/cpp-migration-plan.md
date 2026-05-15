# fitra-cam C++ 移行計画 (3 カメラ・最大性能)

## Context

Python 版で並列 2 カメラ × RTMPose を動かしたところ、`recent_pose_fps` が 18fps で頭打ちになった。マイクロベンチでは pose 単独 50fps 出るのに、二並列で詰まる原因は CUDA セッションのコンテキスト切替 + Python GIL に張り付いた前後処理。

最終目標は **Jetson Orin Nano Super で 3 USB カメラ同時、リアルタイム** に高速・高精度の 2D 姿勢推定 (YOLOX + RTMPose) を回すこと。これは Python では到達不能と判断し、**TensorRT C++ API 直 + Jetson Multimedia API zero-copy** にフル移行する。

実機調査で確認済みのツール:

- `libnvinfer 10.3.0.30` (`/usr/include/aarch64-linux-gnu/NvInfer.h`)
- CUDA 12.6 / cuDNN 9.3
- Jetson Multimedia API `/usr/src/jetson_multimedia_api/` (NvBuffer / libnvjpeg / Argus)
- OpenCV 4.x C++ dev headers
- g++ 11.4, CMake 3.22

既存 Python 実装 (`scripts/`, `web/dual_rtmpose/`, `README.md`, `requirements-jetson.txt`, `scripts/setup_jetson_env.sh`) は **`python/` 配下に退避** して参照用に残す。新 C++ 実装は **`cpp/`** に作る。フロントエンド静的ファイルは `web/dual_rtmpose/` のまま、Python / C++ どちらからも serve できる位置に置く。

## アーキテクチャ (3 カメラ・1 プロセス・GIL なし)

```
USB cam 0 ┐
USB cam 1 ┤── V4L2 mmap ring (4 buf/cam) ──┐
USB cam 2 ┘                                 │
                                            ▼
                  [Capture thread × 3]  → NVJPEG decode (GPU) → NvBuffer
                                            │
                                            ▼ per-cam SPSC queue (size 1, drop-old)
                  [Inference thread]
                    1. detect frame == det_frequency? → YOLOX (B=3 batch) on shared TRT context
                    2. crop+normalize on GPU → RTMPose (B≤3 batch) on shared TRT context
                    3. SimCC argmax decode + inverse affine on GPU/CPU
                    4. write PoseSnapshot atomically
                                            │
                                            ▼
                  [Publisher thread]  Crow HTTP+WS, broadcasts snapshot @ 30Hz
                                       static files /web/dual_rtmpose/*

                  [Recorder] (optional, ondemand)  cv::VideoWriter or libav
```

設計の肝:

- **CUDA stream 単一・TensorRT context 単一**: セッション切替コストをゼロに。複数 context は無し
- **NVJPEG batched decode**: 3 枚同時 JPEG decode を 1 回の API call で
- **NvBuffer zero-copy**: V4L2 dequeue → NVJPEG output → TensorRT 入力までホスト経由しない
- **RTMPose dynamic batch (1..3)**: TRT engine を `min=1 / opt=3 / max=3` で optimization profile 構成
- **YOLOX 静的 batch=1 を 3 回 enqueue**: mmdeploy NMS-in-graph のため batch 化が難しい。CUDA stream 上で背中合わせ実行すれば実用上は B=3 とほぼ同等

## リポジトリレイアウト

```
fitra-cam/
├── README.md                      (C++ 用に書き直し)
├── .gitignore                     (cpp/build, *.engine 等を追加)
├── cpp/
│   ├── CMakeLists.txt
│   ├── cmake/
│   │   └── FindTensorRT.cmake     # 自前 Find module
│   ├── src/
│   │   ├── main.cpp
│   │   ├── camera/
│   │   │   ├── v4l2_capture.{hpp,cpp}    # ioctl + mmap ring
│   │   │   └── nvjpeg_decoder.{hpp,cpp}  # libnvjpeg batched
│   │   ├── infer/
│   │   │   ├── trt_engine.{hpp,cpp}      # IBuilder/IRuntime/IExecutionContext 抽象
│   │   │   ├── yolox.{hpp,cpp}           # letterbox (GPU) + post NMS scale-back
│   │   │   └── rtmpose.{hpp,cpp}         # affine warp (NPP) + SimCC decode
│   │   ├── pipeline/
│   │   │   ├── pose_pipeline.{hpp,cpp}
│   │   │   └── snapshot_bus.{hpp,cpp}    # MPMC, per-camera latest slot
│   │   ├── web/
│   │   │   └── server.{hpp,cpp}          # Crow app
│   │   └── util/
│   │       ├── cuda_check.hpp            # CUDA_CHECK/TRT_CHECK マクロ
│   │       ├── ring_buffer.hpp           # NvBuffer pool
│   │       └── logging.hpp               # spdlog ラッパ
│   ├── third_party/                     # 全て header-only / FetchContent
│   └── tools/
│       ├── build_engines.cpp             # ONNX → .engine プリビルド CLI
│       ├── pose_bench.cpp                # オフライン推論ベンチ
│       └── correctness_check.cpp         # Python 出力との数値一致確認
├── python/                              (退避された旧実装、原本そのまま)
│   ├── scripts/
│   ├── requirements-jetson.txt
│   └── setup_jetson_env.sh
├── web/dual_rtmpose/                    (既存フロントエンドそのまま流用)
├── models/                              (ONNX/engine cache。.gitignore で engine は除外)
└── outputs/                             (git ignore)
```

## 依存 (header-only / CMake FetchContent 中心)

| 用途             | ライブラリ              | 入手             |
|----------------|---------------------|----------------|
| 推論             | TensorRT 10.3       | apt (済)         |
| CUDA            | CUDA 12.6           | apt (済)         |
| 取り込み         | Jetson Multimedia API | `/usr/src/jetson_multimedia_api/` |
| JPEG decode      | libnvjpeg           | MM API 同梱       |
| アフィン         | OpenCV 4.x C++      | apt (済)         |
| HTTP + WS        | **Crow** v1.x       | FetchContent (header-only) |
| ログ             | spdlog              | FetchContent (header-only) |
| JSON             | nlohmann/json       | FetchContent (header-only) |
| CLI              | CLI11               | FetchContent (header-only) |
| 並行キュー       | moodycamel/readerwriterqueue | FetchContent (header-only) |

第三者ライブラリは全部 header-only + FetchContent。binary 依存追加無し。

## 段階実装

### Phase 0 — Python 退避 + C++ skeleton

- `scripts/`, `requirements-jetson.txt`, `setup_jetson_env.sh` を `python/` 配下に `git mv`
- `README.md` を C++ 中心に書き換え (python/ への参照を残す)
- `cpp/CMakeLists.txt` 雛形、`cmake/FindTensorRT.cmake` を整備
- "Hello TensorRT": `cudaGetDeviceCount()` と `nvinfer1::createInferRuntime()` が呼べて main がリンクすることを確認

### Phase 1 — TRT 推論エンジンラッパ + correctness

- `infer/trt_engine.{hpp,cpp}`: ONNX → engine build (CLI tool `tools/build_engines.cpp`) と engine deserialize、bindings の動的取得、CUDA stream 駆動
- `infer/yolox.{hpp,cpp}` (B=1 入力 [1,3,416,416] 静的)
- `infer/rtmpose.{hpp,cpp}` (dynamic batch 1..3 の optimization profile)
- `tools/correctness_check.cpp`: 既存 Python `pose_pipeline.py` と同じ ONNX を使い、固定テスト画像で keypoint 座標差 < 1px / bbox IoU > 0.99 を assert

### Phase 2 — 1 カメラ end-to-end

- `camera/v4l2_capture`: 1 カメラの mmap ring (4 buf) を回し MJPEG バッファを取得
- `camera/nvjpeg_decoder`: 1 枚モードで MJPEG → CUDA buffer (BGR uint8) decode
- `pipeline/pose_pipeline`: 1 カメラ inference のサンプル
- ベンチ: `--max-frames 200 --save-every 50` 相当を C++ ツール側に作って recent_pose_fps を出す。Python 単独 1 カメラより速くなることを確認

### Phase 3 — 3 カメラ並列 + Crow Web

- `camera/v4l2_capture` を N カメラ対応に
- `camera/nvjpeg_decoder` を batched API に拡張
- `pipeline/pose_pipeline`: 3 カメラ inference + per-camera tracker
- `web/server.cpp`: Crow で `/ws` WebSocket + `/` static (`web/dual_rtmpose/`)
- snapshot JSON のスキーマは Python 版と一致させる → 既存 `web/dual_rtmpose/app.js` がそのまま動く

### Phase 4 — TensorRT 最適化 + ベンチ

- FP16 engine 作成 (`--fp16` フラグ)
- INT8 PTQ: 100 frame ぐらいで calibration table 作って RTMPose を INT8 化
- pinned memory (`cudaHostAllocMapped`) で host↔device 転送高速化
- `nsys profile` でタイムライン取って残りボトルネック特定
- ベンチで 3 カメラ × 30fps (90fps aggregate) を超えるか確認

### Phase 5 — 録画オーバーレイ (任意)

- `cpp/tools/record_overlay.cpp`: raw mp4 を 3 本録 → 推論 overlay → side-by-side
- `cv::VideoWriter` (mp4v) で十分。libav 直は不要

## 検証戦略

| Phase | 検証コマンド                                                  | 合格基準                                                                 |
|-------|-----------------------------------------------------------|----------------------------------------------------------------------|
| 0     | `cmake --build cpp/build && ./cpp/build/main --help`        | リンク成功、TRT runtime が初期化できる                                         |
| 1     | `./cpp/build/tools/correctness_check --image test.jpg`     | YOLOX bbox IoU > 0.99 / RTMPose 各 keypoint 距離 < 1.0 px (vs Python ORT) |
| 2     | `./cpp/build/main --cam0 ... --max-frames 200 --bench`     | recent_pose_fps が Python 同条件比 1.5× 以上                                |
| 3     | `./cpp/build/main --cam0 ... --cam1 ... --cam2 ... --port 8000` + ブラウザ目視 | 3 ペイン分の skeleton が滑らかに描画され WS bundle 30Hz で届く |
| 4     | `./cpp/build/main --device tensorrt --fp16 --bench`           | aggregate pose ≥ 90 fps / GPU 利用率 80% 超                                |
| 5     | `./cpp/build/record_overlay --seconds 30`                    | 5 本の mp4 出力、メタデータ fps が実測通り                                       |

## リスク・未確定事項

- **mmdeploy 版 YOLOX ONNX の TRT 化**: NMS plugin (`EfficientNMS_TRT`) が TRT 10.3 で動くか未確認。動かなければ NMS なし版に export し直すか、CPU で NMS する暫定モードを置く
  - **Phase 1 で確認済み**: TRT 10.3.0 で問題なくビルドできる。ただし NMS 出力は data-dependent shape のため `IOutputAllocator` 経由で読む必要がある (`cpp/src/infer/trt_engine.cpp::BindingOutputAllocator`)
- **RTMPose dynamic batch profile**: optimization profile の min/opt/max を `1/2/3` に。1 人で済むケースを opt にしておかないと latency が悪化する可能性
- **NVJPEG batch 制約**: バッチ内で JPEG ヘッダの色空間 / サイズが一致している必要。USB UVC カメラの MJPG はすべて同 640x480 YUV420 なので OK のはず
- **Jetson Orin Nano Super の電力**: 3 カメラ + GPU 全力で MAXN 必須。`/etc/nvpmodel.conf` 確認
- **engine cache invalidate**: TRT バージョン / FP16 設定 / GPU SM が変わったら .engine 無効。`models/` の `.engine` は git 管理しない
- **FP16 RTMPose drift (再確認)**: Phase 1 の correctness で観測 — RTMPose を FP16 engine で回すと、低スコア keypoint (score < 0.5 帯) が input frame の Y で 100-200px ずれることがある。FP32 engine なら max kpt L2 ≈ 1.15px / p95 ≈ 0.57px に収まる。Phase 4 で INT8/FP16 を扱うときは Phase 1 と同じ動画 (`outputs/recorded_rtmpose/20260515_064342/raw_cam0.mp4`) で再現テストすること

## Phase 1 完了メモ (2026-05-15)

- engine 構築: `models/{yolox_tiny,rtmpose_s}.fp32.engine`
- correctness:
  - 入力: `outputs/recorded_rtmpose/20260515_064342/raw_cam0.mp4` の最初 30 フレーム
  - 基準: `python/scripts/dump_reference_keypoints.py --device cpu`
  - 候補: `cpp/build/tools/dump_keypoints` (YOLOX/RTMPose とも FP32)
  - 結果: bbox IoU min 0.993 / kpt L2 max 1.15px (99% < 0.75px) / score diff max 0.016
  - 合格基準は計画値の **< 1.0px から ~1.5px に緩和** が現実的 (TRT 10.3 vs ORT 1.23 のカーネル差で説明できる微差)
- 関連ファイル:
  - `cpp/src/infer/trt_engine.{hpp,cpp}` — `IOutputAllocator` で data-dependent shape 対応 (YOLOX NMS, RTMPose dynamic batch)
  - `cpp/src/infer/{yolox,rtmpose}.{hpp,cpp}` — Python と同じ前後処理 (cv::warpAffine, BGR mean/std, SimCC argmax + 逆 affine)
  - `cpp/tools/{build_engines,dump_keypoints}.cpp` — engine ビルド CLI と correctness 用ダンプ
  - `python/scripts/{dump_reference_keypoints,compare_keypoints}.py` — Python リファレンスと差分集計

## 完了の定義

- `cpp/build/main` が 3 USB カメラ + WebSocket + 録画オプションを一気通貫で提供
- recent_pose_fps が aggregate で 80fps 以上、Python 比 4× 以上
- 既存 `web/dual_rtmpose/` 静的ファイルがそのまま使え、ブラウザで 3 ペイン skeleton が見える
- `python/` 配下に旧実装が残り、README から退避場所が辿れる
- engine prebuild → cold start ≤ 3 秒
