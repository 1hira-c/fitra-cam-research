# 調査レポート

Jetson Orin Nano Super 上で、複数 USB MJPEG カメラ入力、RTMPose、3D リフティング、WebUI 表示までを段階的に成立させるための調査メモ。

## レポート一覧

- `research/jetson-runtime.md`: Jetson 実行基盤の前提整理
- `research/multi-camera-ingest.md`: 複数 USB MJPEG カメラ入力の整理
- `research/sync-calibration.md`: 同期とキャリブレーション設計の整理
- `research/rtmpose-pipeline.md`: RTMPose 2D 推論パイプラインの整理
- `research/lifting-strategy.md`: 3D リフティング方式の整理
- `research/webui-3d.md`: WebUI 3D 表示方式の整理
- `research/integration-benchmark.md`: 統合と性能評価の整理

## 実機メモ

2026-04-25 時点で Jetson Orin Nano Super 実機上の確認結果:

- JetPack: `6.2.1+b38`
- L4T: `R36.4.7`
- Python: `3.10.12`
- GStreamer: `1.20.3`
- OpenCV: `4.8.0`
- 接続カメラ: Global Shutter Camera 2 台 (`32e4:0234`)
- capture node: `/dev/video0`, `/dev/video2`
- 非 capture node: `/dev/video1`, `/dev/video3`
- 安定識別: `/dev/v4l/by-path/...2.3... -> video0`, `/dev/v4l/by-path/...2.4... -> video2`

補足:

- `/dev/v4l/by-id` は 2 台を安定に表現できていないため、この機種では **by-path 優先**にする。
- カメラは **MJPEG** と **YUY2** をサポートし、実機では `1280x720@30fps MJPEG + nvjpegdec` を 2 台同時にスモークテスト済み。

以降の調査もこの配下へ追加していく。
