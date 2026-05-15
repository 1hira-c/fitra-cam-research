# Phase 7 — マルチカメラ 3D + IK + Kalman 計画

## Context

Phase 6 で C++ 側 2-cam aggregate 170 fps を達成し、2D 推論のスループットには十分余裕がある。次の課題は **複数カメラを生かして 2D から 3D に持ち上げ、関節のガタつきを実用レベルまで抑えること**。既存研究ドキュメント (`docs/research/lifting-strategy.md`, `docs/research/sync-calibration.md`) は既に「多視点三角測量を主軸」「ChArUco でキャリブ」「3D 化後に Kalman 等で平滑化」の方針を決めており、本プランはその方針を Phase 7 として C++ に直接組み込む。新規追加は IK (ボーン長 + 関節角度制限) のみ。

確定方針:

- 実装言語: **最初から C++** (Python プロトタイプを挟まない)
- IK スコープ: **ボーン長 + 関節角度制限** (FABRIK ベース、必要なら CCD フォールバック)
- Kalman 位置: **3D 化後に各関節 6D 状態 (pos + vel)**
- 段階: **Phase 7 として `docs/cpp-migration-plan.md` の "段階実装" に追記**

## ゴール / 完了条件

1. ChArUco で 2/3 カメラの intrinsic + extrinsic が取得・保存できる。
2. 同時刻の per-cam 2D keypoint を時刻揃え → confidence-weighted triangulation で 3D skeleton (17 joint) を出せる。
3. 各関節 3D に 6D Kalman を入れ、欠損フレームでは予測のみで補間。
4. IK でボーン長を学習済み定数に固定、肘・膝・首・腰の hyperextension を防止。
5. 既存 WebSocket JSON スキーマと両立する形で 3D を `/ws3d` (新 endpoint) に publish。
6. 3-cam live で再投影誤差 median < 3 px、ボーン長変動 < 5%、3D 関節 jitter (静止時の SD) < 10 mm を測れる。
7. aggregate スループット 90 fps 以上を維持 (3D 段階のレイテンシ ≤ 3 ms/frame target)。

## 段階 (Phase 7a–f)

### 7a: ChArUco キャリブレーションツール

- 新規: `cpp/tools/calibrate_cameras.cpp` (CLI11)
  - 内部: 単カメラごとに `cv::aruco::CharucoDetector` + `cv::aruco::calibrateCameraCharuco` で intrinsic + distCoeffs
  - 外部: 全カメラから同時に見える ChArUco 画像を複数収集 → カメラ 0 を世界基準 → `cv::stereoCalibrate` を pair で順次解いてカメラ 1, 2 を 0 に位置合わせ
  - 入力: `--cam <path>` を複数、`--squares-x/-y`, `--square-len`, `--marker-len`, `--dict`
  - 出力: `calibrations/cam_params.yaml` (gitignore 必須) — `intrinsics[id]: { K[9], dist[5], width, height }`, `extrinsics[id]: { R[9], t[3] }` (世界 = cam0)
- 新規: `cpp/src/lift/calib_io.{hpp,cpp}` — yaml-cpp で読み書き (FetchContent or apt `libyaml-cpp-dev`)、ランタイム側も使う
- `.gitignore` に `calibrations/` を追加 (engine 同様にデバイス固有)
- 検証: re-projection RMS を YAML の `quality` フィールドに記録、stdout にも出す

### 7b: 三角測量モジュール

- 新規: `cpp/src/lift/triangulator.{hpp,cpp}`
  - 入力: `vector<PerCamSnapshot>` (各 `{ cam_id, captured_at, vector<Person> }`)
  - 同時刻揃え: `cpp/src/lift/sync.{hpp,cpp}` (下記 7e) が ±10ms 窓で組をまとめてから入る
  - 人物対応付け: single-person 既定なら最大 bbox 同士、multi-person は bbox 中心を cam0 へ重心再投影しての greedy 一致 (Phase 7a 完了時点では single-person 固定で良い)
  - 各関節独立に: conf > `--kp-conf-thresh` (既定 0.3) の視点だけ集めて DLT (4×4 SVD)、視点 2 未満なら invalid マーク
  - Confidence-weight: 各行に `sqrt(conf_v)` を掛けた重み付き DLT (`lifting-strategy.md:43` 準拠)
  - Reprojection outlier rejection: 再投影誤差 > `--max-reproj-px` (既定 6) の視点を外して 1 回再解く
- 新規型: `cpp/src/infer/types.hpp` に `Joint3D { float x,y,z,score; bool valid; }` と `Skeleton3D { array<Joint3D, kNumKeypoints>; }` を追加
- 計算は Eigen で十分軽量 (17 joint × 4×4 SVD ≪ 0.5 ms)、依存追加: Eigen3 を `find_package` (apt `libeigen3-dev`)

### 7c: 3D Kalman フィルタ

- 新規: `cpp/src/lift/kalman.{hpp,cpp}`
  - per-joint 6D constant-velocity モデル: x = [px, py, pz, vx, vy, vz]
  - F = block(I, dt*I; 0, I), H = [I 0], Q tunable (process noise), R = triangulator 由来の共分散か固定値
  - Update 段で `Joint3D.valid == false` の場合は予測のみ
  - Person tracker: single-person 既定では 1 トラック、`KalmanTracker` がフレーム間ペアリングと再初期化を担当 (長期欠損 → reset)
- パラメータは CLI 経由で調整可能 (`--kalman-q-pos`, `--kalman-q-vel`, `--kalman-r-meas`)

### 7d: IK ソルバ (ボーン長 + 関節角度制限)

- 新規: `cpp/src/lift/ik.{hpp,cpp}`
  - COCO-17 木構造を `cpp/src/lift/skeleton_def.hpp` に定数で定義: parent[17], bone_pairs[], hinge_joints (肘/膝)
  - ボーン長キャリブ: 起動直後の N 秒間 (`--bone-calib-sec` 既定 5) で per-bone 長を中央値で確定 → 以降ロック
  - Solver: **FABRIK** ベース (CCD より関節角度限制との相性が良い)
    - 全関節を Kalman 後の 3D 位置に初期化
    - Forward pass: end effector (手首・足首・頭) を観測位置へ → 親に向かってボーン長を維持しながら戻す
    - Backward pass: 骨盤 (左右 hip の中点) を観測位置に戻す → 末端方向にボーン長維持しながら進む
    - Hinge 制約: 肘・膝のステップ後に、隣接 3 関節の角度を [0°, 180°] にクランプ
  - 反復: 上限 5 回 or position delta < 1 mm で停止
- 推定 < 0.5 ms / person、依存追加なし (Eigen で完結)

### 7e: 時刻同期 & 集約

- 新規: `cpp/src/lift/sync.{hpp,cpp}`
  - 既存 `multi_pipeline.cpp` が per-cam に 2D 結果を生成しているところに hook を入れ、各 cam の最新結果を `monotonic timestamp` キーでバッファ
  - aggregator: 全 cam の最新が出揃ったか、最も古い cam timestamp が `--sync-window-ms` (既定 15) 超で揃わなければ揃った分だけで進める
  - drop-old policy は既存方針 (latest-frame-wins) を踏襲
- `multi_pipeline.cpp` に「2D 結果が来たら sync agg → triangulate → kalman → ik → publish」の段を追加

### 7f: 3D WebSocket 出力 + 診断

- `cpp/src/web/crow_server.{hpp,cpp}` を拡張:
  - 既存 `/ws` (2D) は維持
  - 新 `/ws3d`: `{ frame, ts, persons_3d: [ { id, joints: [[x,y,z,score,valid], ...17] } ], stats: { tri_fps, reproj_err_med_px, bone_len_drift_pct, kalman_innov_avg } }`
- 新規: `cpp/tools/dump_keypoints_3d.cpp` (既存 `dump_keypoints.cpp` ベース、`--overlay-3d` で各 cam 動画に再投影点を重ねた MP4 を吐く) — correctness 検証の主ツール
- `docs/research/lifting-strategy.md` の "初期マイルストーン" に対応: reproj error / 欠損率 / 3D jitter の数値ダンプ機能

## 既存資産の再利用

- `cpp/src/infer/types.hpp:18-29` の `Keypoint` / `Person` をそのまま triangulator 入力に使う
- `cpp/src/pipeline/multi_pipeline.cpp` の 2D 結果集約ループに 3D 段階を挿入 (新規スレッドは作らない、main thread 上)
- `cpp/src/web/crow_server.cpp` の JSON 組み立てパターンと subscriber 管理をそのまま流用
- `cpp/tools/dump_keypoints.cpp` のオフライン dump フォーマット (JSON Lines) を 3D dump にも継承
- `outputs/recorded_rtmpose/20260515_064342/raw_cam{0,1}.mp4` を correctness 検証の固定入力に使う (`overlay_cam{0,1}.mp4` は 2D リファレンス)
- `python/scripts/dual_rtmpose_web.py` `_publisher_loop` の JSON フォーマット規約は `/ws` 側で維持

## 変更ファイル一覧 (予定)

新規:

- `cpp/src/lift/calib_io.{hpp,cpp}`
- `cpp/src/lift/triangulator.{hpp,cpp}`
- `cpp/src/lift/kalman.{hpp,cpp}`
- `cpp/src/lift/ik.{hpp,cpp}`
- `cpp/src/lift/sync.{hpp,cpp}`
- `cpp/src/lift/skeleton_def.hpp`
- `cpp/src/CMakeLists.txt` に `lift/` サブディレクトリ追加
- `cpp/tools/calibrate_cameras.cpp`
- `cpp/tools/dump_keypoints_3d.cpp`
- `cpp/tools/CMakeLists.txt` に上記 2 ツールを追加

修正:

- `cpp/CMakeLists.txt` — `find_package(Eigen3 3.4 REQUIRED)`、`find_package(yaml-cpp REQUIRED)` (or FetchContent)、OpenCV `aruco` モジュール必須化
- `cpp/src/infer/types.hpp` — `Joint3D`, `Skeleton3D` 追加
- `cpp/src/pipeline/multi_pipeline.{hpp,cpp}` — 2D 集約後の 3D 段階フック
- `cpp/src/main.cpp` — `--calib`, `--kp-conf-thresh`, `--max-reproj-px`, Kalman/IK パラメータの CLI 追加
- `cpp/src/web/crow_server.{hpp,cpp}` — `/ws3d` endpoint
- `.gitignore` — `calibrations/` 追加
- `docs/cpp-migration-plan.md` — Phase 7 セクション追記 (7a–f の完了条件含む)
- `web/dual_rtmpose/` — (オプション) Three.js 3D viewer プロト or 既存ページに 3D toggle (本プランの最終段で別タスク化)

## 検証手順

1. **キャリブ単体**:

   ```bash
   ./cpp/build/calibrate_cameras \
       --cam /dev/v4l/by-path/...:2.3:1.0 \
       --cam /dev/v4l/by-path/...:2.4:1.0 \
       --squares-x 5 --squares-y 7 --square-len 0.04 --marker-len 0.03 \
       --out calibrations/cam_params.yaml
   ```

   → YAML 出力、stdout の reproj RMS が < 0.5 px、`quality.intrinsic_rms` 記録あり。

2. **オフライン三角測量 correctness** (録画ファイルベース):

   ```bash
   ./cpp/build/dump_keypoints_3d \
       --video outputs/recorded_rtmpose/20260515_064342/raw_cam0.mp4 \
       --video outputs/recorded_rtmpose/20260515_064342/raw_cam1.mp4 \
       --calib calibrations/cam_params.yaml \
       --overlay-3d outputs/3d_reproj/{cam0,cam1}.mp4 \
       --out outputs/3d_reproj/joints3d.jsonl
   ```

   → 再投影 overlay 動画で 3D 点が原 2D 上に乗ること、JSONL の bone length 各行の変動係数 (CoV) < 5%。

3. **Kalman / IK 数値特性**:
   - 静止状態 3 秒録画で 3D 関節 SD < 10 mm (Kalman 効果)。
   - 急加速 (パンチ動作) で過剰平滑による遅延 < 50 ms (パラメータ調整可能性確認)。
   - IK 適用前後でボーン長分散の比較 (IK 後はほぼ 0)。
   - 肘を 0° 以下に折る入力を意図的に triangulator に与え、IK 出力で 0–180° にクランプされること。

4. **Live 3-cam 動作**:

   ```bash
   sudo nvpmodel -m 0 && sudo jetson_clocks
   ./cpp/build/main --cam0 ... --cam1 ... --cam2 ... \
       --calib calibrations/cam_params.yaml --enable-3d
   ```

   → ブラウザで `/ws3d` を購読、stats が 3-cam aggregate ≥ 90 fps、`tri_fps` ≥ 30、`reproj_err_med_px` < 3、`bone_len_drift_pct` < 5。

5. **2D 互換性**: `/ws` 側スキーマ・出力 fps が Phase 6 から悪化していないことを既存ビューアで確認。

## リスクと対応

- **IK の関節定義**: COCO-17 には脊柱が無く、骨盤も hip 中点で代用。FABRIK が不安定なら段階的に CCD にフォールバック (実装枠は同じ `lift/ik.{hpp,cpp}`)。
- **Eigen + Crow + spdlog の ABI**: 既存 FetchContent 群と Eigen3 apt 版の混在はトラブルになりやすい。最初に空の `lift/` モジュールでビルド通過を確認してから中身を入れる。
- **C++ から始めるリスクのヘッジ**: triangulator の単体テスト (合成 3D 点 → 既知 K/R/t で投影 → 復元) を `cpp/tools/test_triangulator.cpp` として最初に書き、Python 比較なしでも閉じた correctness が取れるようにする。
- **3-cam 配線**: USB 2.0 帯域は 2-cam ですでに飽和しているので、3-cam 時の MJPG 解像度を下げる選択肢を `--cam2-size` で温存。
