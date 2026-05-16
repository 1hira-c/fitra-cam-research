# RTMPose TensorRT INT8 初回評価

## 2026-05-15 初回結果

実装:

- `build_engines --int8 --int8-blobs` で TensorRT PTQ calibration を実行可能にした
- `python/scripts/dump_calibration_blobs.py` で録画 MP4 から RTMPose 入力 blob を生成可能にした
- C++ ランタイム推論パスは変更していない。engine 差し替えだけで評価する

生成物:

| file | note |
|---|---|
| `models/calib_rtmpose_256x192.bin` | 150 samples, 84.4 MiB |
| `models/rtmpose_m.int8.engine.calib.cache` | TensorRT calibration cache |
| `models/rtmpose_m.int8.engine` | RTMPose-M INT8 engine, 15.9 MiB |

検証:

| target | result |
|---|---|
| CMake / C++ build | OK |
| `build_engines --help` | `--int8-blobs`, `--int8-cache`, `--int8-batch-size` 表示 OK |
| calibration dump | CPU EP で 150 samples 生成 OK |
| RTMPose-M INT8 engine build | OK |
| `dump_keypoints` 30 frame smoke | OK, overlay 生成 OK |

300 frame の録画 MP4 比較:

| engine | command path | throughput |
|---|---|---:|
| `rtmpose_m.fp16.engine` | `dump_keypoints --prebaked` | 26.66 fps |
| `rtmpose_m.int8.engine` | `dump_keypoints --prebaked` | 28.06 fps |

INT8 vs FP16 drift:

| metric | value |
|---|---:|
| keypoint L2 mean | 9.57 px |
| keypoint L2 p95 | 36.47 px |
| keypoint L2 p99 | 77.14 px |
| keypoint L2 max | 148.53 px |
| frame max L2 p95 | 88.97 px |

判断:

- 現在の RTMPose-M INT8 は速度差が小さく、keypoint drift が大きいので運用候補にはできない
- INT8+FP16 fallback build は calibration cache まで読めたが、engine 最適化が長時間化したため中断した
- 次は RTMPose-L INT8、または TensorRT の層別 precision 制約で SimCC 出力付近を FP16/FP32 に残す方向を評価する
