# RTMPose S/M/L TensorRT INT8 評価計画

> Status: **計画 (将来タスク)** — 着手前。Phase 6b 着地 (aggregate 170 fps) 後の残課題から派生。
> Owner: 未定 / Target branch: 未割当 / 想定工期: 0.5–1 週

## 背景

fitra-cam の C++/TensorRT 移行は Phase 6b で aggregate 170 fps (Phase 3 比 5.5×) に到達した。
現状の運用モデルは **RTMPose-M FP16**。次のレバーとして以下が残っている
(`docs/cpp-migration-plan.md` Phase 4 残課題, 行 140-147 / 233 / 306):

- **INT8 PTQ** (RTMPose / YOLOX): `cpp/tools/build_engines.cpp:107,140-141` は `--int8`
  フラグだけ存在し、calibrator は "no calibrator wired yet" の状態
- **精度 vs 速度のトレードオフ可視化**: 現運用 (M-FP16) 以外の組み合わせは
  系統的に測られていない

本計画は **RTMPose の S / M / L 3 variant × FP16 / INT8 の組み合わせ 6 engine** を
Jetson Orin Nano Super 上で評価し、運用モデル選定の判断材料を作る。
YOLOX 側 INT8 は本計画のスコープ外。

## ゴール

1. INT8 calibrator (`nvinfer1::IInt8EntropyCalibrator2`) を `cpp/src/infer/` に実装し、
   `build_engines --int8 --int8-blobs <path>` でエンジン生成が完結する
2. RTMPose-L ONNX を取得 (S/M は既存)
3. 6 engine について以下を表で残す
   - 精度: Python ORT FP32 を基準に **max / p95 / mean kpt L2** と **bbox IoU**
   - 速度: `pose_bench` の **rtm_ms / recent_pose_fps**
4. 散布図 (rtm_ms × max-L2) で運用候補を可視化

成功定義: `docs/research/rtmpose-int8-eval.md` (結果報告ファイル) に 6 行の表と
散布図、運用モデル変更可否の判断記述があること。

## 設計の柱

### 入力サイズと preprocess は全 variant 共通
RTMPose-S/M/L はいずれも **256×192 / 17 kp simcc 出力**
(`cpp/src/infer/rtmpose.hpp:13-16`, `python/scripts/pose_pipeline.py:274-275`)。
**前処理・後処理コードは変更不要**。engine 差し替えだけで variant 切替が完結する。

### Calibration data
**`outputs/recorded_rtmpose/20260515_064342/raw_cam{0,1}.mp4`**
(Phase 1 correctness で実績ある 2 視点 × 30 秒) から 100–200 frame をサンプリング。
RTMPose は person crop 入力なので、生フレームではなく
**`pose_pipeline` と同じ YOLOX → bbox crop → 256×192 affine 後の blob** をダンプする。
2 視点ばらけるよう 1 視点 ~80 frame 均等サンプリング。

注: calibration データが室内・同一被写体に偏るリスクは
`## リスク / 留意` で受容。後段の `dump_calibration_blobs.py` は再実行可能な
形で残し、データ拡張時の再ビルドを容易にしておく。

### 既存資産の最大利用
本計画は **新規ランタイムコード 0、ビルダ拡張のみ** で完遂できる:

| 用途 | 既存資産 | 流用形態 |
|---|---|---|
| Engine ビルダ | `cpp/src/infer/trt_builder.{hpp,cpp}` | `BuildOptions` 拡張・`setInt8Calibrator` 配線 |
| ビルダ CLI | `cpp/tools/build_engines.cpp` (rtmpose preset 含む) | `--int8-blobs` / `--int8-cache` を追加 |
| Preprocess (Python) | `python/scripts/pose_pipeline.py` の `build_engines_for` | calibration blob ダンプに流用 |
| Preprocess (C++) | `cpp/src/infer/rtmpose.cpp:102-137` | 変更なし |
| 数値比較 | `python/scripts/compare_keypoints.py` (IoU + L2, 閾値 0.99 / 1.0px) | そのまま転用 |
| リファレンス生成 | `python/scripts/dump_reference_keypoints.py` | variant ごとに ORT-CPU リファレンス |
| 速度計測 | `cpp/tools/pose_bench.cpp:211-224` (rolling fps / stage_ms) | そのまま転用 |
| 動的バッチ profile | `cpp/tools/build_engines.cpp:160-169` (min=1, opt=3, max=3) | そのまま流用 |

## 実装ステップ

### Step 1 — RTMPose-L ONNX 取得
- 取得元: rtmlib body7 系
  (`rtmpose-l_simcc-body7_pt-body7_420e-256x192-*.onnx`、~85MB 想定)
- 配置: `~/.cache/rtmlib/hub/checkpoints/` (S/M と同じ規約)
- 取得手段: rtmlib の初回ロードを起こすか、URL を
  `python/scripts/setup_jetson_env.sh` に追記
- 検証: `python python/scripts/dump_reference_keypoints.py --pose-model <L-onnx>`
  で 1 フレーム推論が通る

### Step 2 — Calibration blob ダンパ (Python)
新規: `python/scripts/dump_calibration_blobs.py`

- `pose_pipeline.py` の YOLOX + RTMPose 前処理 (affine + ImageNet normalize) を流用
- 入力例: `--video raw_cam0.mp4 raw_cam1.mp4 --target 150 --output models/calib_rtmpose_256x192.bin`
- 出力: `(N, 3, 256, 192)` float32 raw binary (ヘッダなし、最も単純)
- N は 100–200、両視点均等サンプリング

### Step 3 — INT8 Calibrator (C++)
新規: `cpp/src/infer/int8_calibrator.{hpp,cpp}`

```cpp
class Int8EntropyCalibrator2 : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator2(std::string blob_path, int batch_size,
                            std::string cache_path);
    ~Int8EntropyCalibrator2() override;
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(std::size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, std::size_t length) noexcept override;
private:
    // mmap / 単一 cudaMalloc / cache file I/O
};
```

- batch_size = 1 (calibration は逐次で十分)
- cache_path に対する read/write を実装 → 2 回目以降のビルドを高速化
- 単一 cudaMalloc を確保、destructor で cudaFree

### Step 4 — `trt_builder` 配線
`cpp/src/infer/trt_builder.hpp` の `BuildOptions` に追加:

```cpp
std::string int8_blob_path;
std::string int8_cache_path;
int int8_batch_size = 1;
```

`cpp/src/infer/trt_builder.cpp:76-81` の `if (opts.int8)` 分岐で
calibrator を生成し `config->setInt8Calibrator(...)` を呼ぶ。
**重要**: calibrator は `config` より長生きさせる必要があるため、
関数スコープで `std::unique_ptr` を 1 つ保持する。

`cpp/tools/build_engines.cpp` のオプションパーサに:
- `--int8-blobs PATH`
- `--int8-cache PATH` (デフォルト: engine_path に `.cache` を付けたパス)
- help 文の "no calibrator wired yet" を更新

`cpp/CMakeLists.txt` の `fitra_infer` ターゲットに `int8_calibrator.cpp` を追加。

### Step 5 — エンジンビルド (オペレーション)
```bash
# S
build_engines --preset rtmpose --onnx <S.onnx> --output models/rtmpose_s.fp16.engine --fp16
build_engines --preset rtmpose --onnx <S.onnx> --output models/rtmpose_s.int8.engine \
    --int8 --int8-blobs models/calib_rtmpose_256x192.bin
# M / L 同様
```

出力: `models/rtmpose_{s,m,l}.{fp16,int8}.engine` (+ `.cache`)。
S/M の FP32 は数値リファレンスとして残置。

### Step 6 — 精度評価 (correctness)
1. `dump_reference_keypoints.py --pose-model <S/M/L ONNX>` で 3 個の ORT-CPU
   リファレンス JSONL
2. `dump_keypoints --pose-engine models/rtmpose_<v>.<prec>.engine
   --video raw_cam0.mp4 --output dump_<v>_<prec>.jsonl` で 6 個
3. `compare_keypoints.py --reference ref_<v>.jsonl --target dump_<v>_<prec>.jsonl`
   で **max / p95 / mean L2** と **bbox IoU**
4. INT8 engine は **同 variant の FP32 ONNX** に対する drift を見る
   (variant 横断は別表で)

期待される drift 帯
(`docs/cpp-migration-plan.md:172` の FP16 S 既知 drift = max 237px の経験則から):

| variant | 判断ライン | 根拠 |
|---|---|---|
| S-INT8 | max L2 < 50px | FP16 で既に荒れている — INT8 で更に悪化見込み |
| M-INT8 | max L2 < 20px | 現運用 M-FP16 の代替候補 |
| L-INT8 | max L2 < 10px | 容量が効くので最も整う想定 |

### Step 7 — 速度評価 (bench)
```bash
sudo nvpmodel -m 0 && sudo jetson_clocks  # 必須 (~/CLAUDE.md 規約)
./cpp/build/pose_bench --pose-engine models/rtmpose_<v>.<prec>.engine \
    --camera /dev/v4l/by-path/... --duration 30 --log-every-s 1.0
```

取得値: `recent_pose_fps` (1 cam), `rtm_ms` (RTMPose stage)。
3 カメラ aggregate は 1 cam 値から推定 (Phase 6b と同じやり方);
物理 3 cam が用意できれば実測を追加。

### Step 8 — 結果ドキュメント
新規: `docs/research/rtmpose-int8-eval.md` (本ファイルとは別、結果報告用)

表フォーマット (1 行 = engine):

| variant | precision | engine MB | rtm_ms | recent_pose_fps | max L2 | p95 L2 | mean L2 | bbox IoU |
|---|---|---|---|---|---|---|---|---|

散布図 (rtm_ms × max-L2) を 6 点プロット。
`docs/cpp-migration-plan.md` の Phase 4 着地メモ末尾に短く追記:
> INT8 評価結果は `research/rtmpose-int8-eval.md` 参照。

## 主要ファイル

### 新規
- `python/scripts/dump_calibration_blobs.py` — calibration blob ダンパ
- `cpp/src/infer/int8_calibrator.{hpp,cpp}` — Calibrator 実装
- `models/calib_rtmpose_256x192.bin` — calibration blob (gitignore)
- `models/rtmpose_{s,m,l}.int8.engine` (+ `.cache`) — 評価対象 (gitignore)
- `docs/research/rtmpose-int8-eval.md` — 結果報告

### 改修
- `cpp/src/infer/trt_builder.hpp` — `BuildOptions` に int8_blob_path / int8_cache_path / int8_batch_size
- `cpp/src/infer/trt_builder.cpp:76-81` — `setInt8Calibrator` 配線
- `cpp/tools/build_engines.cpp` — `--int8-blobs` / `--int8-cache` 配線、help 文更新
- `cpp/CMakeLists.txt` — `int8_calibrator.cpp` を `fitra_infer` に追加
- `python/scripts/setup_jetson_env.sh` — RTMPose-L ONNX 取得を追加 (可能なら)
- `docs/cpp-migration-plan.md` Phase 4 着地メモ — 結果へのリンク追記

## End-to-End 検証手順 (smoke)

1. `sudo nvpmodel -m 0 && sudo jetson_clocks`
2. RTMPose-L ONNX が `~/.cache/rtmlib/hub/checkpoints/` にある
3. `python python/scripts/dump_calibration_blobs.py --target 150
   --output models/calib_rtmpose_256x192.bin`
   → 出力サイズ `150 * 3 * 256 * 192 * 4 = 44.2 MB` に近い
4. `./cpp/build/build_engines --preset rtmpose --onnx <L-onnx>
   --output models/rtmpose_l.int8.engine --int8
   --int8-blobs models/calib_rtmpose_256x192.bin`
   → TRT logger に calibration 進捗、2 回目で cache が効く
5. `./cpp/build/dump_keypoints --pose-engine models/rtmpose_l.int8.engine
   --video outputs/recorded_rtmpose/20260515_064342/raw_cam0.mp4
   --output /tmp/dump_l_int8.jsonl`
6. `python python/scripts/compare_keypoints.py --reference /tmp/ref_l.jsonl
   --target /tmp/dump_l_int8.jsonl`
   → max L2 が想定帯 (< 50px) 内
7. `./cpp/build/pose_bench --pose-engine models/rtmpose_l.int8.engine
   --camera /dev/v4l/by-path/... --duration 30`
   → `rtm_ms` が FP16 比で短縮 (Orin Nano INT8 で ~1.5–2× 期待)
8. 6 engine 全部について Step 6 / 7 を回し、結果ファイルに表を埋める

## スコープ外

- YOLOX の INT8 化 (`docs/cpp-migration-plan.md:306` の残課題だが本計画は RTMPose に閉じる)
- ランタイム (推論パス) 側のコード変更 — engine 差し替えで完結する設計を前提
- 多視点 3D lifting (`docs/research/lifting-strategy.md` の Phase 7) — INT8 評価結果後の別計画
- ベンチ用 3 カメラ目の物理セットアップ — 1 cam 値 × 3 で aggregate 推定

## リスク / 留意

- **calibration 偏り**: raw_cam0/1 は同一被写体・室内のため、本番分布に対し
  過適合の可能性。結果ファイルで限界を明記し、`dump_calibration_blobs.py`
  を再実行可能形態で残す
- **L-INT8 のメモリ**: Orin Nano は LPDDR5 8GB。L 重みは大きい
  (engine ~120MB FP16 → INT8 で ~60MB) ため、3 カメラ同時 + YOLOX と合わせて
  `tegrastats` で RAM 監視
- **FP16 S 既知 drift**: `docs/cpp-migration-plan.md:172` の現象が INT8 で
  更に悪化し得る。S-INT8 が運用 NG でも L-INT8 が代替案として残るので
  3-way 評価には意味がある
- **calibrator の寿命管理**: `IInt8EntropyCalibrator2` は `config` より
  長生きさせる必要がある。`build_engine()` 関数スコープで `unique_ptr`
  を持つ実装にして、TRT API 仕様違反を避ける
