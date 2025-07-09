# EmotionDetection
Проект для **распознавания эмоций** с использованием **YOLOv8 + CLIP**, чтобы в реальном времени определять эмоциональное состояние водителя по изображению.

# Работа
1) Распознавание лиц на изображении с помощью **YOLOv8** 
2) Выравнивание лиц
3) Анализ лиц с помощью **CLIP_Emotion**

# Установка
В программе используется модель [Hugging Face](//huggingface.co/G1Gru/CLIP_Emotions):
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=for-the-badge)](//huggingface.co/G1Gru/CLIP_Emotions)

Установить модель: 
Клонируйте модель с Hugging Face в папку репозитория:
'git clone https://huggingface.co/G1Gru/CLIP_Emotions'

Установка зависимостей:

'''bash
pip install torch torchvision transformers huggingface_hub opencv-python numpy matplotlib ultralytics
'''

# Запуск

1) В папку репозитория поместить картинку "Example.jpg"
2) Запустить Main.py из виртуальной среды с установленными зависимостями

