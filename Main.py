
import cv2
from ultralytics import YOLO 
from huggingface_hub import hf_hub_download
#from supervision import Detections
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import torch
import numpy as np
from torchvision import transforms
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

#Модель smolLM2
# Lm2tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
# SmolLm2 = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# # YOLOv8n (nano)
# YOLOmodel = YOLO("yolov8n.pt")

YOLO8_FACE_PATH = "./model.pt"
CLIP_EMOTION_MODEL_PATH = "./clip_emotion_finetuned"
CLIP_PATCH_14_PATH = "/mnt/d/clip-vit-large-patch14"

class EmotionDetection:
    def __init__(self, image_pth: str, _device = 'cuda'):
        #self.YoloModel = YOLOmodel
        self.image = cv2.imread(image_pth)
        self.device = torch.device(device=_device)
        self.SSDmodel = ssdlite320_mobilenet_v3_large(pretrained=True)
        self.SSDmodel = self.SSDmodel.to(self.device).eval()
        #model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
        self.YOLOmodel = YOLO(YOLO8_FACE_PATH)
        #self.YOLOmodel = torch.hub.load('ultralytics/yolov5', 'yolov5s-face', pretrained=True)
        #self.YOLOmodel = self.YOLOmodel.to(self.device).eval()

        self.processor = CLIPProcessor.from_pretrained(CLIP_EMOTION_MODEL_PATH)
        self.emotion_model = CLIPModel.from_pretrained(CLIP_EMOTION_MODEL_PATH,torch_dtype = torch.float16).to(self.device)
        self.emotion_model.load_state_dict(torch.load("/mnt/d/driverassistant/clip_emotion_finetuned/pytorch_model.bin"), strict=False)
        self.emotion_labels = ["angry","disgusted","fearful","happy" ,"neutral" ,"sad","surprised"]

        self.text_inputs = self.processor(
        text=self.emotion_labels, 
        return_tensors="pt", 
        padding=True
        ).to(self.device)

        #image_inputs = self.processor(images=face_rgb, return_tensors="pt").to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((320, 320)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
    def facesdetectionHaarCascades(self):
        imagegray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(
        imagegray, 
        flags=cv2.CASCADE_SCALE_IMAGE)
        print(f"Faces on the image:{len(faces)}")
    
    def PersondetectionSSD(self):
        input_tensor = self.transform(self.image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions=self.SSDmodel(input_tensor)
        print(len(predictions))

    def FaceDetectionYOLO5(self):
        
        rgbimg = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        results = self.YOLOmodel(rgbimg, self.device)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = map(int, box)
                face_roi = rgbimg[y1:y2,x1:x2]

                h,w, _ = face_roi.shape
                img_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                
                eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                eyes = eyes_cascade.detectMultiScale(img_gray,1.1,4,flags=cv2.CASCADE_SCALE_IMAGE)

                if len(eyes)>=2:
                    eyes = sorted(eyes,key = lambda X:X[0])[:2]
                    (ex1, ey1, ew1, eh1) = eyes[0]
                    (ex2, ey2, ew2, eh2) = eyes[1]
                    left_eye_center = (ex1+ew1//2, ey1 + eh1//2)
                    right_eye_center = (ex2+ew2//2, ey2+eh2//2)
                    dx = right_eye_center[0] - left_eye_center[0]
                    dy = right_eye_center[1] - left_eye_center[1]
                    angle = np.degrees(np.arctan2(dy, dx))
                    center = (w//2, h//2)
                    rot_mat = cv2.getRotationMatrix2D(center,angle,1)
                    aligned_face = cv2.warpAffine(face_roi, rot_mat, (w,h))

                    aligned_face_bgr = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
                    aligned_face_resized = cv2.resize(aligned_face_bgr, (112, 112))
                    normalized_face = aligned_face_resized.astype('float32') / 255.0
                    
                    inputs = self.processor(images=aligned_face_resized, return_tensors="pt").to(self.device)


                    with torch.no_grad():
                        outputs = self.emotion_model(**inputs, **self.text_inputs)
                        logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
                        probs = logits_per_image.softmax(dim=1)
                        print(logits_per_image.softmax(dim=-1))
                        print(self.emotion_labels)
                        pred = torch.argmax(probs, dim=1).item()
                        emotion = self.emotion_labels[pred]

                    cv2.putText(self.image, emotion, (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # name= 'NormalizedFace'+str(center)+'.jpg'
                    # cv2.imwrite(name, aligned_face_resized)

                cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(self.image, f'{conf:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        


        cv2.imwrite('YoloV8FaceDetection.jpg', self.image)
        disp_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,10))
        plt.imshow(disp_img)
        plt.axis('off')
        plt.title('Detected Faces with Emotions')
        plt.show()


    def downloadModelYOLOV8Faces(self):
        model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")



def main():
    print("programm started")
    detector:EmotionDetection = EmotionDetection("Example.jpg")
    #detector.aceDetectionYOLO5()

    detector.FaceDetectionYOLO5()

    # messages = [{"role": "user", "content": "Столица России это?"}]
    # input_text = Lm2tokenizer.apply_chat_template(messages, tokenize=False)
    # inputs = Lm2tokenizer.encode(input_text, return_tensors="pt")

    # outputs = SmolLm2.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    # response = Lm2tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)

    # 2. Откроем веб-камеру
    # cap = cv2.imread("Skebob.jfif",1)


    # result = model(cap)
    # annotated_image = result[0].plot()
    # annotated_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    # plt.imshow(annotated_rgb)
    # plt.axis("off")
    # plt.title("YOLOv8 Detection")
    # plt.show()


if __name__ == "__main__":
    main()