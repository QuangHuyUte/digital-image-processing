<div align="center">

# 👁️ Digital Image Processing Repository
### *Algorithms · Analysis · Computer Vision*

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Headless-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=flat)

<p align="center">
  <i>A comprehensive collection of image processing algorithms and experiments,<br>
  visualizing the mathematics behind digital vision from Spatial to Frequency domains.</i>
</p>

[Explore Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage)

</div>

---

## 📘 Introduction

This repository serves as a digital laboratory for the **Digital Image Processing (DIPR)** course at **HCMUTE**.

It goes beyond simple image manipulation by implementing core algorithms from scratch (or using low-level library functions) to understand the underlying mathematical principles. The project evolves from basic pixel operations to complex segmentation and frequency analysis, packaged in an interactive **Streamlit** web interface for real-time experimentation.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Core** | ![Python](https://img.shields.io/badge/-Python_3.12-3776AB?logo=python&logoColor=white) | Primary programming language. |
| **Processing** | `opencv-python`, `numpy` | Matrix manipulation, FFT, and morphological operations. |
| **Interface** | `streamlit` | Modern, reactive web-based GUI for parameter tuning. |
| **Visualization** | `matplotlib`, `PIL` | Data plotting and image rendering. |

---

## 🚀 Key Features

This repository covers a wide spectrum of Digital Image Processing topics:

### 1. Image Enhancement (Spatial Domain)
* **Intensity Transformations:** Logarithmic, Gamma Correction, Piecewise-Linear contrast stretching.
* **Histogram Processing:** Histogram Equalization (HE) and CLAHE (Adaptive).
* **Spatial Filtering:**
    * *Smoothing:* Mean, Gaussian, Median (Linear & Non-linear).
    * *Sharpening:* Laplacian, Unsharp Masking, Sobel/Gradient operators.

### 2. Frequency Domain Analysis
* **Fourier Transform:** Implementation of FFT and IFFT.
* **Filtering:** Ideal, Gaussian, and Butterworth filters (Lowpass, Highpass, Bandpass).
* **Visualization:** Spectrum analysis and phase visualization.

### 3. Morphological Processing
* **Fundamental Operations:** Erosion, Dilation, Opening, Closing.
* **Advanced Applications:** Boundary Extraction, Region Filling, Morphological Gradient.
* **Custom Kernels:** Analysis with non-standard structuring elements (e.g., Diagonal kernels).

### 4. Image Segmentation & Object Counting
* **Thresholding:** Global, Adaptive, and **Otsu's Binarization** (Automatic).
* **Motion Detection:** Background subtraction (MOG2) for video analysis.
* **Object Analysis:** Connected Component Analysis, Contour detection, and object counting (e.g., fruit counting).

---

## 📸 Screenshots

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Frequency Domain Analysis</b></td>
      <td align="center"><b>Morphological Extraction</b></td>
    </tr>
    <tr>
      <td><img src="https://via.placeholder.com/400x250.png?text=Frequency+Filter+Demo" alt="FFT Demo" width="100%"></td>
      <td><img src="https://via.placeholder.com/400x250.png?text=Morphology+Demo" alt="Morphology Demo" width="100%"></td>
    </tr>
    <tr>
      <td align="center"><b>Object Segmentation (Otsu)</b></td>
      <td align="center"><b>Motion Detection</b></td>
    </tr>
    <tr>
      <td><img src="https://via.placeholder.com/400x250.png?text=Fruit+Counting" alt="Segmentation" width="100%"></td>
      <td><img src="https://via.placeholder.com/400x250.png?text=Motion+Extraction" alt="Motion" width="100%"></td>
    </tr>
  </table>
</div>

---

## 🔌 Installation & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/QuangHuyUte/digital-image-processing.git](https://github.com/QuangHuyUte/digital-image-processing.git)
    cd digital-image-processing
    ```

2.  **Install dependencies**
    ```bash
    Mount to each of Homework folders and:
    pip install -r requirements.txt
    ```

3.  **Run the App**
    launch the interactive Streamlit dashboard:
    ```bash
    Read README.md in each of Homework folders 
    ```
