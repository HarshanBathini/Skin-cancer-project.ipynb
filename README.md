# Skin Cancer Prediction using Deep Learning

Classifies dermoscopic skin images into 7 lesion types (incl. melanoma) using transfer learning
(MobileNetV2) on the HAM10000 dataset, with a Streamlit web demo.

## Dataset
HAM10000 – 10,015 dermatoscopic images, 7 classes: akiec, bcc, bkl, df, mel, nv, vasc.
Download: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000 and extract into `data/`.

## Approach
- Lesion-level train/val/test split (70/15/15) using `lesion_id` to prevent data leakage
- Class-weighted loss to handle severe imbalance (~67% of images are benign nevi)
- Data augmentation (flips, rotation, zoom, contrast)
- Two-phase transfer learning: train head -> fine-tune top 40 MobileNetV2 layers
- Evaluation: per-class precision/recall/F1 and confusion matrix (recall on melanoma matters most)

## Run
```
pip install -r requirements.txt
python train.py --data_dir data
streamlit run app.py
```

## Results
(Fill in after training: test accuracy, macro F1, melanoma recall, add confusion_matrix.png)

## Disclaimer
Educational use only; not a medical device.
