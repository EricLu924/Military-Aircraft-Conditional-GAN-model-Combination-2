# Military-Aircraft-Conditional-GAN-model-Combination2
【 Conditional GAN (cGAN) Image Generation Framework-Combination-2 】

This repository provides a **comprehensive PyTorch implementation of a Conditional Generative Adversarial Network (cGAN)** for **class-conditional image generation**, equipped with **extensive quantitative evaluation metrics**.

The framework is designed for **research, academic experiments, and data augmentation**, and supports **long-horizon stable training** with **FID, SSIM, PSNR, and MSE** evaluation.

This implementation was trained and validated using a **primarily developed and validated using the Military Aircraft Detection Dataset from Kaggle** and is suitable for tasks such as **dataset balancing, visual data augmentation, and generative modeling research**.

---

## 🚀 Key Features

* Conditional GAN (cGAN) with label embeddings
* DCGAN-style generator and discriminator
* Spectral Normalization in discriminator for stability
* Support for **multi-class conditional generation**
* Long training (up to 500 epochs)
* Quantitative evaluation:

  * **FID (Fréchet Inception Distance)**
  * **SSIM (Structural Similarity Index)**
  * **PSNR (Peak Signal-to-Noise Ratio)**
  * **MSE (Mean Squared Error)**
* Automatic best-model selection based on **lowest FID**
* Periodic image sampling and visualization
* Full training logs and metric curves

---

## 📁 Project Structure

```bash
.
├── Conditional_GAN_model_epoch500_lr0.0002.py   # Main training script
├── manufacturers.txt                            # List of class labels
├── dataset_root/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
├── output_dir/
│   ├── images_new_test_resize/                  # Generated samples during training
│   ├── epoch_x/temp_real/                       # Real images for FID
│   ├── epoch_x/temp_fake/                       # Generated images for FID
│   ├── best/
│   │   ├── best_G.pth
│   │   ├── best_D.pth
│   │   └── best_metrics.json
│   ├── final_models/
│   │   ├── final_generator.pth
│   │   └── final_discriminator.pth
│   ├── metrics.json
│   └── metrics_curve.png
```

---

## 🧠 Model Architecture

### Generator

* Input:

  * Random noise vector `z` (default: 100-dim)
  * Class label embedding
* Architecture:

  * Transposed convolutions
  * Batch Normalization
  * ReLU activations
  * Tanh output
* Output resolution: **64 × 64 RGB images**

### Discriminator

* Input:

  * Image (real or generated)
  * Embedded class label (spatially expanded)
* Architecture:

  * Convolutional layers with **Spectral Normalization**
  * LeakyReLU activations
  * Sigmoid output

---

## 📦 Dataset Format
This project is **primarily designed and evaluated using the following Kaggle dataset**:

**Military Aircraft Detection Dataset (Kaggle)**  
<img width="2203" height="371" alt="image" src="https://github.com/user-attachments/assets/6361547c-3d02-4f6f-a2f1-88c90a11e3cf" />

🔗 https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset/data

The dataset must follow a **class-conditional directory structure**:

```bash
dataset_root/
├── train/
│   ├── manufacturer_A/
│   │   ├── img001.jpg
│   │   ├── img002.png
│   ├── manufacturer_B/
│   └── ...
```

Class names must match entries in `manufacturers.txt` (one class per line).

---

## 🔄 Data Preprocessing

Applied transformations:

```python
transforms.Resize((64, 64))
transforms.RandomHorizontalFlip(p=0.5)
transforms.ToTensor()
transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
```

Images are normalized to **[-1, 1]**, compatible with Tanh output.

---

## 🔁 Training Configuration

| Parameter       | Value                   |
| --------------- | ----------------------- |
| Epochs          | 500                     |
| Batch Size      | 64                      |
| Learning Rate   | 0.0002                  |
| Optimizer       | Adam (β1=0.5, β2=0.999) |
| Noise Dimension | 100                     |
| Image Size      | 64 × 64                 |

---

## 📊 Evaluation Metrics

For each epoch, the following metrics are computed:

* **MSE** – pixel-wise reconstruction error
* **PSNR** – signal-to-noise ratio
* **SSIM** – perceptual structural similarity
* **FID** – distribution-level similarity (Inception-v3, 2048-d)

✔ **Best model is selected based on minimum FID**.

---

## 📈 Outputs & Visualization

* Generated image grids (per `sample_interval`)
* `metrics.json` – per-epoch quantitative metrics
* `metrics_curve.png` – MSE / PSNR / SSIM / FID curves

Example generated samples:

```bash
output_dir/images_new_test_resize/1000.png
```

---

## ▶️ How to Run

```bash
python Conditional_GAN_model_epoch500_lr0.0002.py \
  --dataroot /path/to/dataset_root \
  --manufacturer_file manufacturers.txt \
  --output_dir ./new_third \
  --epochs 500 \
  --batch_size 64 \
  --lr 0.0002 \
  --use_cuda
```

---

## 💻 Hardware & Environment

* Python ≥ 3.8
* PyTorch ≥ 1.10
* CUDA-enabled GPU recommended (multi-GPU supported via device selection)

---

## 📦 Requirements

```txt
torch
torchvision
numpy
pytorch-fid
scikit-image
matplotlib
tqdm
Pillow
```

---

## ⚠️ Notes

* GAN training is inherently unstable; monitoring FID and generated samples is strongly recommended
* Class imbalance may affect conditional generation quality
* FID computation is expensive and increases training time

---

## 📜 License

This project is intended for **research and academic use only**.

Please validate generated data carefully before using it for downstream tasks.

---

## ✉️ Experimental Results
<img width="1474" height="587" alt="image" src="https://github.com/user-attachments/assets/237c0e44-f4e6-4243-a824-e1980c40a3b1" />
<img width="1706" height="95" alt="image" src="https://github.com/user-attachments/assets/0a99954f-7ff3-433c-a298-350b413f8199" />

⭐ If you find this repository useful, consider starring it on GitHub!

