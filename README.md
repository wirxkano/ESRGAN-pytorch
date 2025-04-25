# ESRGAN: Single Image Super-Resolution

This project implements **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)** for single image super-resolution. The goal is to upscale low-resolution images (e.g., 4x) while preserving perceptual quality and fine details.

---

## 📌 Overview

Image Super-Resolution (SR) is a technique to reconstruct a high-resolution image from its low-resolution counterpart. It has applications in:

- Improving visual quality of images
- Enhancing image-based AI model accuracy
- Restoring old or compressed images

This implementation follows the paper:  
**[ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks](https://doi.org/10.48550/arXiv.1809.00219)**

---

## 🗂️ Dataset

- **Pre-training**: [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **ESRGAN training**: DF2K dataset (DIV2K + Flickr2K)
- **Testing**: Standard benchmark datasets like **Set5** and **Set14**

> Make sure to update the dataset paths in `constants.py` accordingly.

---

## 🛠️ Methodology

- **Generator**: RRDBNet (Residual-in-Residual Dense Block Network)
- **Discriminator**: VGG-based discriminator for perceptual quality
- **Losses**:
  - Adversarial loss
  - Perceptual (VGG) loss
  - Pixel-wise L1 loss
- **Training stages**:
  1. **Stage 1**: Train RRDBNet with L1 loss
  2. **Stage 2**: Fine-tune ESRGAN (GAN-based training)

- Trained up to **500 epochs** for best perceptual quality.

---

## 🚀 Usage

### ⚙️ Training

```bash
# Train RRDBNet (pretraining stage)
python main.py --mode train_rrdb --train_dir ./data/train --valid_dir ./data/valid --rrdbnet_ckpt_dir ./checkpoints/rrdb

# Train ESRGAN (adversarial stage)
python main.py --mode train_esrgan --train_dir ./data/train --valid_dir ./data/valid \
  --rrdbnet_ckpt_path ./checkpoints/rrdb/best_model.pth \
  --esrgan_ckpt_dir ./checkpoints/esrgan
```

### 🧪 Testing

```bash
python main.py --mode test \
  --test_dir ./data/test \
  --rrdbnet_ckpt_path ./checkpoints/rrdb/best_model.pth \
  --esrgan_ckpt_path ./checkpoints/esrgan/best_model.pth
```

### 🔍 Inference on a single image

```bash
python main.py --mode inference \
  --img_path ./example_image.png \
  --rrdbnet_ckpt_path ./checkpoints/rrdb/best_model.pth \
  --esrgan_ckpt_path ./checkpoints/esrgan/best_model.pth
```

Output will be saved as result.png in the current directory.

## 🖼️ Result
Example output:

<img src="./result.png" alt="Super-resolved output" width="400" />

## 📚 Notes
- This project is implemented for educational and research purposes.
- Based on the original ESRGAN paper and common training recipes.

- Checkpoints are not included in the repo.

👉 Contact me to get pretrained weights.

## 🤝 Contributing
Pull requests are welcome! If you want to contribute:

1. Fork the repo

2. Create a new branch

3. Submit a PR or open an issue

## 📬 Contact
If you have questions or need checkpoint files, feel free to contact me.

---
