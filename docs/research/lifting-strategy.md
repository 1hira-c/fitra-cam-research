# 3D リフティング方式の整理

## 結論

- この構成では **多視点三角測量を主軸**にするのが正しい。
- **2D-to-3D lifting** は主経路ではなく、**欠損補間・平滑化・将来の単眼フォールバック**用途として位置付けるのが安全。
- 3D 品質を左右する順序は、おおむね **同期 / 外部校正 / 2D keypoint 品質 / 三角測量のロバスト化** で、lifting モデル選定はその後。

## なぜ三角測量を主軸にするか

### 向いている理由

- すでに **複数台の校正済みカメラ** を前提にしている
- 多視点なら、同時刻の 2D keypoint から **幾何的に 3D を再構成**できる
- 学習済み 3D lifting のデータ分布に引きずられにくい
- カメラ配置や人物動作の変更に対して解釈しやすい

### lifting を主軸にしない理由

- 2D-to-3D lifting は学習データ依存が強い
- Human3.6M 系 3D モデルは、実運用の姿勢や設置条件とずれることがある
- RTMPose の 2D keypoint 定義と 3D モデル入力定義が **そのまま一致しない**ことが多い

## 推奨アーキテクチャ

1. 各カメラで 2D keypoint + confidence を得る
2. timestamp で同時刻フレームを束ねる
3. 人物 ID を視点間で対応付ける
4. **confidence-weighted triangulation**
5. reprojection error で outlier view を除外
6. 3D keypoint 列に対して時間方向の平滑化

## ロバスト化の要点

### 欠損・オクルージョン

- 各関節は **2 視点以上**で見えているときだけ三角測量
- 低 confidence 視点は除外
- reprojection error が大きい視点は outlier とみなす

### 重み付け

- triangulation は **confidence-weighted least squares** が第一候補
- 2D confidence をそのまま使うのではなく、下限 threshold を設けてから使う

### 平滑化

- まずは 3D 化後に **EMA / Savitzky-Golay / Kalman** のいずれかで平滑化
- 初期実装では複雑な時系列 NN より、軽量で追いやすいフィルタを優先

## keypoint 定義の注意

- RTMPose は COCO 系 17 keypoint を使うことが多い
- 一方、Human3.6M 系 3D lifting は pelvis / thorax などを含み、**定義差**がある
- lifting を使う場合は **COCO → Human3.6M のマッピング**が必要
- 三角測量主軸なら、まずは **2D 出力定義そのままの 3D 点群**として扱えるので手戻りが少ない

## 初期マイルストーン

1. 2 台構成で 17 keypoint の 3D triangulation を成立
2. reprojection error と欠損率を可視化
3. 3D jitter に軽量平滑化を適用
4. その後、必要なら lifting 補助を検討

## 推奨

- **主経路**: calibrated multi-view triangulation
- **補助**: confidence weighting + outlier rejection + temporal smoothing
- **後段拡張**: 2D-to-3D lifting は fallback / hole filling / monocular mode 用

## 参考

- MMPose multi-view 3D / 3D demo の前提
- confidence-weighted triangulation の一般的手法
- Human3.6M と COCO の keypoint convention 差分
