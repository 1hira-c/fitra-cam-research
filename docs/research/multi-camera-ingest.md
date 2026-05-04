# 複数 USB MJPEG カメラ入力の整理

## 結論

- USB カメラ入力は **GStreamer 前提**で組み、MJPEG デコードは **`nvjpegdec`** または **`nvv4l2decoder mjpeg=1`** を優先する。
- デバイス固定は `/dev/video*` を直接使わず、**`/dev/v4l/by-id`** か **udev ルール**で行う。
- 台数設計は机上見積もりだけで決めず、**2 台から解像度・fps・平均フレームサイズを実測**して増やす。

## 実機確認結果

- Jetson Orin Nano Super 上で 2 台の `Global Shutter Camera (32e4:0234)` を確認済み。
- 実 capture node は **`/dev/video0` と `/dev/video2`**。
- **`/dev/video1` と `/dev/video3` は capture device ではない**。
- `/dev/v4l/by-id` は 1 台分しか安定に出ておらず、今回の実機では **`/dev/v4l/by-path` を使う方が安全**。
- `gst-device-monitor-1.0` で、両カメラとも **MJPEG** と **YUY2** の両方をサポートすることを確認済み。
- 実機スモークテストでは **`1280x720@30fps MJPEG + nvjpegdec`** を **2 台同時**に通せた。

## 推奨パイプライン

```text
v4l2src device=/dev/v4l/by-id/... ! image/jpeg,width=...,height=...,framerate=... ! nvjpegdec ! nvvidconv ! appsink
```

または:

```text
v4l2src ! image/jpeg,format=MJPG,... ! nvv4l2decoder mjpeg=1 ! nvvidconv ! appsink
```

今回の実機では `by-id` より次のような **by-path** 指定が安全:

```text
/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0
/dev/v4l/by-path/platform-3610000.usb-usb-0:2.4:1.0-video-index0
```

## 実装方針

1. カメラごとに GStreamer パイプラインを生成する。
2. `appsink` は **最新フレームのみ取得**し、推論側でフレームドロップを許容する。
3. 推論と取り込みを分離し、カメラ入力スレッドはできるだけ軽く保つ。

## 注意点

- USB ハブは接続口を増やせても、**上流帯域は増えない**。
- MJPEG は取り回しが良い一方、**複数台では USB 帯域とデコード負荷が先に問題になりやすい**。
- Jetson Orin Nano Dev Kit は複数 USB ポートを持つが、**実効帯域はポート配置とハブ構成の実測確認が必要**。

## 最初の成立条件

- 2 台の USB MJPEG カメラを固定名で取得できる。
- 各カメラのフレームを `appsink` 経由で安定取得できる。
- 解像度・fps ごとの帯域とドロップ率を記録できる。

## 参考

- NVIDIA Jetson Accelerated GStreamer docs
- DeepStream `nvjpegdec` plugin docs
- Linux persistent video device naming (`/dev/v4l/by-id`, udev)
