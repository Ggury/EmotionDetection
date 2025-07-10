# EmotionDetection
A project for **emotion recognition** using **YOLOv8 + CLIP**.

# Functionality
1) Face detection on images using **YOLOv8**  
2) Face alignment  
3) Face analysis using **CLIP_Emotion**

# Installation

This project uses a fine-tuned CLIPv14 model [CLIP_Emotions](https://huggingface.co/G1Gru/CLIP_Emotions).

Download the model:  
Clone the model from Hugging Face into your repository folder:
```bash
git clone https://huggingface.co/G1Gru/CLIP_Emotions
```

Install dependencies:

```bash
pip install -r requirements.txt
```

# Usage

1) Place an image containing the face you want to analyze into the repository folder with the name `Example.jpg`.  
2) Run `main.py` from your virtual environment with the installed dependencies.

