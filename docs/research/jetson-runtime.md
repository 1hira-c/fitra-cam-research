# Jetson 実行基盤の整理

## 結論

- 前提は **JetPack 6.2.1** で見るのが安全。
- Python は **3.10 系** を前提にし、OpenCV は **apt 版 `python3-opencv`** を使って `CAP_GSTREAMER` を維持する。
- リポジトリに以前置いていた `onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl` は **JetPack 6.2.1 とは非互換**だったが、Jetson AI Lab の `jp6/cu126` 配布 wheel は JetPack 6.2.1 で使用できる。
- このプロトタイプは **apt OpenCV(GStreamer 有効)** を維持したまま、`onnxruntime-gpu` を Jetson AI Lab wheel で入れるのが現時点の最短ルート。

## 実機前提

2026-04-29 の Jetson Orin Nano Super 実機:

- JetPack: `6.2.1`
- L4T: `R36.4.7`
- Python: `3.10.12`
- CUDA: `12.6`
- TensorRT: `10.3`
- cuDNN: `9.3`

## 重要ポイント

1. JetPack 6.2.1 では `nvidia-jetpack` で必要コンポーネントを導入できる。
2. `python3-opencv` は GStreamer 有効 build を提供するため、pip OpenCV wheel を混ぜないことが重要。
3. ONNX Runtime の shared provider は依存ライブラリが揃わないとロードされず、**Python import 自体は成功しても `CUDAExecutionProvider` が見えない**。
4. Jetson 上の OpenMMLab 系は aarch64 依存で相性が出やすく、**MMCV / MMDeploy はバージョン固定前提**で扱うべき。

## 推奨スタック

| 項目 | 推奨 |
| --- | --- |
| OS / BSP | JetPack 6.2.1 |
| Python | 3.10 系 |
| 即実行の 2D 推論 | Jetson AI Lab 配布 `onnxruntime-gpu` |
| GStreamer 取り込み | apt 版 OpenCV |
| GPU 推論の次段階 | 自前 build した ONNX Runtime / TensorRT |
| 性能最適化 | TensorRT FP16 |
| モデル変換 | MMDeploy |

## 今回確認した互換性問題と解決

`onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl` を読み込むと、provider library 自体は存在する:

- `libonnxruntime_providers_cuda.so`
- `libonnxruntime_providers_tensorrt.so`

ただし `ldd` で確認すると `libonnxruntime_providers_cuda.so` が **`libcudnn.so.8`** を要求していた。JetPack 6.2.1 側は **`libcudnn.so.9`** 系なので、そのままでは CUDA provider をロードできない。

つまり旧 wheel では:

1. wheel の import は通る
2. `ort.get_available_providers()` には `CPUExecutionProvider` しか出ない
3. `--backend onnxruntime --device cuda` を指定すると runtime error になる

その後、NVIDIA forum で案内されている **Jetson AI Lab** の `jp6/cu126` index を確認したところ、次の wheel が公開されていた:

- `onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl`

この wheel を Jetson Orin Nano Super / JetPack 6.2.1 実機へ入れると:

1. `ort.__version__ == 1.23.0`
2. `ort.get_available_providers()` が `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
3. `ort.get_device()` が `GPU`

となり、CUDA / TensorRT provider を確認できた。

## 実行方針

### 推奨の導入方法

- `.venv` は `python3 -m venv --system-site-packages .venv`
- `onnxruntime` / `onnxruntime-gpu` の競合を避けるため、どちらか一方だけ入れる
- JetPack 6.2.1 では Jetson AI Lab の wheel を使う:

```bash
. .venv/bin/activate
python -m pip uninstall -y onnxruntime onnxruntime-gpu
PYTHONNOUSERSITE=1 python -m pip install --no-deps \
  https://pypi.jetson-ai-lab.io/jp6/cu126/+f/4eb/e6a8902dc7708/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

### すぐ動かす場合

- `.venv` は `python3 -m venv --system-site-packages .venv`
- `ORT_VARIANT=gpu ./scripts/setup_jetson_env.sh` または上記 wheel install を使う
- `rtmlib==0.0.15` は `--no-deps`
- 実行は `--device auto` または `--device cuda`

### GPU 実行に進む場合

- Jetson AI Lab wheel にない新しい版が必要なら、JetPack 6.2.1 の **CUDA 12.6 / cuDNN 9.3 / TensorRT 10.3** に合わせて ONNX Runtime を build する
- ONNX Runtime 公式の Jetson build 手順を基準に wheel を作る
- その wheel で `ort.get_available_providers()` に `CUDAExecutionProvider` が出ることを確認してから使う

## リスク

- Jetson 向けでない PyTorch / torchvision / mmcv の組み合わせは壊れやすい。
- JetPack 5 系向けの `onnxruntime-gpu` wheel を JetPack 6.2.1 に流用すると、cuDNN SONAME 不一致で CUDA provider が見えない。
- `onnxruntime` と `onnxruntime-gpu` を同じ venv に同居させると import / provider 判定が不安定になるので混在させない。
- TensorRT 化は最終的に必要だが、初期段階でそこに寄せ過ぎるとデバッグが重くなる。

## 参考

- NVIDIA JetPack install/setup
- NVIDIA JetPack 6.2.1 release notes
- NVIDIA forum thread for JetPack 6.2.1 ORT wheel
- Jetson AI Lab package index (`jp6/cu126`)
- NVIDIA PyTorch for Jetson
- ONNX Runtime build docs for Jetson
- MMPose deployment guide
- MMDeploy MMPose / TensorRT docs
