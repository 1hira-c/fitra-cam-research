# 統合と性能評価の整理

## 結論

- 評価指標は **FPS だけでは不十分**。最低でも **end-to-end latency / stage latency / drop rate / jitter / CPU-GPU-RAM 使用率** を取る。
- Jetson 上では **`tegrastats` + 段階別 timestamp ログ** を基本計測系にする。
- 完成条件は「全部入りで一度動く」ではなく、**2 台構成で各段が測定可能な状態**を先に作るのが正しい。

## 計測対象

### 基本 KPI

- **Throughput**: fps
- **End-to-end latency**: capture から WebUI 表示用 3D pose 出力まで
- **Stage latency**: capture / decode / detect / pose / triangulation / smoothing / websocket publish
- **Drop rate**: 入力フレームに対して処理されなかった割合
- **Jitter**: latency の揺れ

### Jetson リソース

- CPU 使用率
- GPU 使用率
- RAM 使用量
- 温度
- 電力 / throttling の兆候

## 計測方法

### アプリ側

各フレームに同一 ID と monotonic timestamp を持たせ、各段の境界で記録する。

例:

1. `t_capture`
2. `t_decode_done`
3. `t_detect_done`
4. `t_pose_done`
5. `t_triangulate_done`
6. `t_publish_done`

これにより frame ごとの stage latency と end-to-end latency を算出できる。

### Jetson 側

- `tegrastats` をベースにログ取得
- 必要に応じて Nsight Systems で timeline を見る

## ボトルネック切り分けの順序

1. **USB / 取り込み** が詰まっていないか
2. **MJPEG decode** が CPU へ落ちていないか
3. **人物検出** が重すぎないか
4. **RTMPose** の単体 latency が支配的でないか
5. **triangulation / smoothing / publish** は軽いか
6. **WebUI** が描画側で詰まっていないか

## 推奨する段階的完成条件

### マイルストーン A

- 2 台カメラ入力が安定
- 取り込み fps / drop rate を測れる

### マイルストーン B

- 単カメラ RTMDet + RTMPose が安定
- detect / pose latency を分離して測れる

### マイルストーン C

- 2 台で 3D triangulation が成立
- reprojection error と 3D jitter を確認できる

### マイルストーン D

- WebSocket で 3D pose を WebUI に配信
- ブラウザ側で安定描画できる

### マイルストーン E

- 2 台 end-to-end で通し性能を記録
- その後 3 台以上へ増やせるかを判断

## 初期の目安

- WebUI 表示は **15–30 fps** で十分
- end-to-end latency は、まず **一貫して測れること**が重要
- drop rate は低く抑えたいが、リアルタイム性優先なら **古いフレームを捨てて最新を保つ**方針がよい

## 推奨

- まずは **2 台 / 1 人 / RTMPose-s か m / 256x192 / FP16 化前提**で計測系を作る
- 計測を入れる前に最適化しない
- カメラ台数を増やす判断は、**stage latency の内訳**を見てからにする

## 参考

- Jetson `tegrastats`
- Nsight Systems
- edge CV pipeline benchmarking の一般的指標
