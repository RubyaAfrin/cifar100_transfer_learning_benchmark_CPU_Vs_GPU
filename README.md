# CIFAR-100 Transfer Learning and CPU vs GPU Benchmark

Fine-tuning an ImageNet-pretrained **DenseNet121** to classify the 100-class **CIFAR-100** dataset, and measuring training speed and resource utilization of CPU and GPU.

**Tools:** Python · TensorFlow / Keras · Google Colab · Pandas · Matplotlib

---

## Method

### Data
- **CIFAR-100:** 60,000 colour images (32×32 pixels) across 100 classes.
- **Split:** 45,000 training, 5,000 validation (held out from the training set), and 10,000 test images. The test set is used only once, for the final evaluation.
- **Augmentation:** random horizontal flips and random crops (pad to 36×36, then crop back to 32×32).
- **Upscaling:** images are resized to 128×128 **inside the model**, so the work runs on the GPU instead of the CPU.

### Model and training
- **Backbone:** DenseNet121 pretrained on ImageNet, with its own preprocessing, then global average pooling, dropout (0.3), and a 100-class softmax layer.
- **Phase 1:** train only the new classifier head with the backbone frozen (3 epochs, Adam, learning rate 0.001).
- **Phase 2:** unfreeze the whole network and fine-tune it (up to 15 epochs, Adam, learning rate 0.0001 with cosine decay). Batch-normalization statistics stay frozen, and early stopping restores the best epoch by validation accuracy.
- **Other settings:** batch size 64, label smoothing 0.1, mixed precision on GPU.

### Benchmark
- Each hardware runs **full fine-tuning** (the compute-heavy workload) for 3 short epochs: 200 steps on GPU and 20 on CPU.
- **The first epoch is excluded** from the speed average, because it includes one-time model compilation.
- **Inference** is timed after a warm-up pass.
- Throughput is measured in **images per second**, so runs of different lengths are directly comparable. CPU, RAM, and GPU utilisation are logged every epoch.

### Resilient training
Free Colab sessions can disconnect. The notebook saves checkpoints to Google Drive after phase 1 and after every fine-tuning epoch, and resumes automatically when rerun.

---

## Results

### Accuracy (test set of 10,000 images)

| Metric | Result |
|---|---|
| Top-1 accuracy | **⏳ to be added after the full run** |
| Top-5 accuracy | **⏳ to be added after the full run** |

*Top-1 is how often the model's first guess is correct. Top-5 is how often the correct class is among its five most likely guesses.*

### Training time and resource utilisation: CPU vs GPU

| Metric | CPU | GPU | GPU vs CPU |
|---|---|---|---|
| Hardware | Intel Xeon @ 2.20 GHz (2 vCPUs) | NVIDIA Tesla T4 | |
| Precision | `float32` | mixed precision (`float16`) | |
| Batch size | 64 | 64 | |
| **Training throughput** | 3.9 images/s | 166.8 images/s | **42× faster** |
| **Inference throughput** | 18.7 images/s | 954.9 images/s | **51× faster** |
| Time per training epoch (45,000 images)* | about 3.2 hours | about 4.5 minutes | |
| CPU usage | 98.7% | 60.7% | |
| RAM usage | about 47% | about 35% | |
| GPU utilisation | – | 81.0% | |
| GPU memory used | – | 27.5% | |

*\*Estimated from training throughput. Speeds and utilisation are averaged over the steady-state epochs; the first epoch includes one-time model compilation and is excluded.*

- **The CPU run was compute-bound:** the CPU worked at nearly full capacity, so its speed reflects its real limit.
- **On the GPU run, the GPU did most of the work** while the CPU prepared and fed the images.
- **Mixed precision:** the GPU used `float16` and the CPU used `float32`, since CPUs don't benefit from mixed precision. The speedup reflects the standard way to run each.
- **Room for improvement:** the GPU was idle about 19% of the time and used under a third of its memory. A larger batch size (for example 128) would likely raise utilisation and speed further.

---

## Improvement over my first version

My original notebook reached **35.1%** test accuracy. Reviewing it, I found and fixed several problems:

| Problem in the first version | Fix |
|---|---|
| Test set used for validation during training | Separate 5,000-image validation split. Test set used once |
| Batch size of 1 (about 10 minutes per epoch) | Batch size 64 with mixed precision |
| 32×32 images fed to a network pretrained on 224×224 images | Upscaled to 128×128 on the GPU |
| Pixels divided by 255 instead of the backbone's preprocessing | DenseNet121's own `preprocess_input` |
| Backbone kept frozen, no augmentation | Two-phase fine-tuning with augmentation |
| Resource usage recorded as a single snapshot | Averaged across each epoch |

---

## Limitations
- **TPU was not tested.** A TPU runtime wasn't available on free Colab during this project. The notebook supports TPU, so it can be added later.
- **Colab hardware varies** between sessions, so exact speeds may differ slightly on rerun.

---

## Repository structure

```
cifar100_transfer_learning_benchmark.ipynb   the full notebook (training, evaluation, benchmark)
results/
  benchmark_CPU_DenseNet121_128px.json        CPU benchmark summary
  benchmark_GPU_DenseNet121_128px.json        GPU benchmark summary
  full_GPU_DenseNet121_128px.json             accuracy run summary (added after the full run)
  *_epochs.csv                                per-epoch timings and resource usage
```

## How to run
Open the notebook in Google Colab.
- **Accuracy run:** choose a GPU runtime, set `RUN_MODE = "full"`, and run all cells.
- **Benchmark:** set `RUN_MODE = "benchmark"` and run all cells once on a CPU runtime and once on a GPU runtime. The last section compares the saved results.
