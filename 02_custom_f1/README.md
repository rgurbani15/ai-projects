\# 02 Custom F1 Car Detector



Custom-trained YOLOv8n model to detect F1 toy cars (LEGO + plastic).



\## How It Works

\- Trained on 78 annotated images in Google Colab (T4 GPU)

\- 50 epochs, 640x640 input size

\- Dataset created and labeled in Roboflow



\## Run

```bash

python f1\_detector.py

## Model Weights

The trained model (`best.pt`) is too large for GitHub. Download options:

1. **Train yourself:** Follow the training steps in the main README using Google Colab
2. **Request from author:** Contact rgurbani15@gmail.com
3. **Use Roboflow dataset:** [Your Roboflow project link if public]

Place `best.pt` in this folder before running `f1_detector.py`