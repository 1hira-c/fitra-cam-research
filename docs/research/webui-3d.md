# WebUI 3D 表示方式の整理

## 結論

- WebUI は **Three.js で 3D 骨格を描画**し、Jetson 側からは **WebSocket** で pose データを送る構成が第一候補。
- この用途は映像会議ではなく **telemetry / visualization** に近いため、最初から WebRTC に寄せる必要は薄い。
- 映像プレビューと 3D 骨格表示は **別レイヤ**で考える。まずは 3D pose 配信を軽く成立させる。

## 推奨構成

### バックエンド

- Python プロセスで推論・3D 化
- FastAPI などで WebSocket endpoint を持つ
- 送信データは `timestamp`, `person_id`, `keypoints_3d`, `confidence` を基本にする

### フロントエンド

- Three.js で joint と bone を描画
- `requestAnimationFrame` に合わせて最新 pose を反映
- OrbitControls で視点操作

## 通信方式

### まず WebSocket を使う理由

- 実装が単純
- サーバ集約で複数クライアントに配りやすい
- pose データ量は映像よりずっと小さく、TCP のオーバーヘッドが支配的になりにくい

### WebRTC を後回しにする理由

- NAT / TURN / signaling で構成が重い
- 今回の主目的は 3D pose の可視化で、超低遅延映像伝送ではない

## データ形式

### 初期実装

- JSON で十分
- 例:

```json
{
  "timestamp": 1713990000.123,
  "persons": [
    {
      "id": 1,
      "keypoints3d": [[x, y, z, c], ...]
    }
  ]
}
```

### 後で最適化

- クライアント数や fps が増えたら binary / msgpack を検討

## Three.js 表示方針

- joint は小さな sphere か point
- bone は `LineSegments`
- geometry は **毎フレーム再生成しない**
- buffer を pre-allocate して座標だけ更新する

## 表示品質

- ブラウザ表示は **15–30 fps** を目安に decimate してよい
- 視覚 jitter はフロントで軽く平滑化してもよいが、主平滑化はバックエンドの 3D pose 側に寄せる
- ネットワーク遅延より **pose jitter の方が見た目に効く**ことが多い

## 映像プレビューとの関係

- 最初は **3D 骨格だけを表示**
- 次段階で必要なら、2D カメラ映像プレビューを別ペインで表示
- 映像配信は別問題なので、最初から WebUI 内に重く統合しない

## 最初の成立条件

1. ブラウザが WebSocket 接続できる
2. 3D keypoint を受信して骨格を更新表示できる
3. 単一人物・単一 skeleton で安定表示できる
4. timestamp と person_id を保持したまま複数人物へ拡張可能

## 推奨

- **バックエンド**: FastAPI + WebSocket
- **フロント**: Three.js
- **配信**: pose telemetry を JSON で送る
- **拡張**: 後から binary 化、映像プレビュー追加

## 参考

- Three.js real-time skeleton visualization の一般的実装
- FastAPI WebSocket 配信
- WebSocket vs WebRTC の設計比較
