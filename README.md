---
title: Super Resolution
emoji: 🏢
colorFrom: indigo
colorTo: red
sdk: streamlit
sdk_version: 1.29.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Image Super Resolution - Computer Vision Project

## Members
- Nguyen Ba Thiem (20214931)
- Doan The Vinh (20210940)
- Pham Quang Tung (20210919)
- Nguyen Huu Nam (20210630)

## Introduction
This project aims to develop a deep learning model for enhancing the resolution of images, known as Super Resolution. Our approach utilizes state-of-the-art techniques in deep learning to upscale images while preserving quality.

## Installation
To clone this report, make sure you follow steps below to download large files.
```
git lfs install
git clone https://github.com/thiemcun203/Image-Super-Resolution.git
```

Before running the application, ensure you have the necessary packages installed:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the web application, use the following command:

```bash
streamlit run app.py
```
## Inference and Results

To run inference and check results, follow the detailed steps provided in our Kaggle notebook:

[Infer Super Resolution](https://www.kaggle.com/code/thimcun/infer-super-resolution)

## Folders Structure

### `models/`
Contains all models with **checkpoints**, file architecture, and training files necessary for the super-resolution process.

### `images/`
Use this folder to test the application with some images. Place your low-resolution images here and run the application to see the enhanced results.

## Acknowledgements


