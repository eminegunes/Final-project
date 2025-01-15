from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import numpy as np
from sort import *
from fer import FER

class PersonTracker:
    def __init__(self, video_path, mask_path):
        self.classNames = ["person"]
        self.mask = cv2.imread(mask_path)
        if self.mask is not None:
            self.mask = cv2.resize(self.mask, (int(self.mask.shape[1] / 2), int(self.mask.shape[0] / 2)))
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        self.cap = cv2.VideoCapture(video_path)
        success, img = self.cap.read()
        if not success:
            raise ValueError("Video could not be read. Please check the path or file format.")
        self.size = (img.shape[1], img.shape[0])
        self.model = YOLO("yolov8n.pt")
        self.emotion_model = FER(mtcnn=True)
        
        # Video yazıcı ayarları
        self.cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video = cv2.VideoWriter("SONUC5.mp4", self.cv2_fourcc, 30, self.size)
        
        # Takip değişkenleri
        self.limits = [289, 281, 590, 285]
        self.enteredIDs = set()
        self.exitedIDs = set()
        self.totalCount = []
        self.trackedObjects = {}
        self.tracked_speeds = {}
        self.tracked_emotions = {}
        
    def calculate_speed(self, previous_position, current_position, time_elapsed):
        if time_elapsed == 0:
            return 0
        distance = math.sqrt((current_position[0] - previous_position[0])**2 + 
                           (current_position[1] - previous_position[1])**2)
        speed = distance / max(time_elapsed, 0.1)  # Sıfıra bölünmeyi önle
        return min(speed, 1000)  # Makul bir üst sınır
    
    def process_frame(self, frame, prev_time):
        current_time = time.time()
        time_elapsed = current_time - prev_time
        
        # Maske kontrolü ve uygulama
        if self.mask is not None and frame.shape[:2] == self.mask.shape[:2]:
            imgRegion = cv2.bitwise_and(frame, self.mask)
        else:
            imgRegion = frame  # Maskeyi uygulayamıyorsa orijinal frame'i kullan
        
        # Nesne tespiti
        results = self.model(imgRegion, stream=True)
        
        detections = np.empty((0, 5))
        for r in results:
            boxes = r.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # Sadece insan sınıfı
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))
        
        # Tracker güncelleme
        resultsTracker = self.tracker.update(detections)
        
        # Çizgi çizme
        cv2.line(frame, (self.limits[0], self.limits[1]), 
                (self.limits[2], self.limits[3]), (0, 0, 255), 5)
        
        # Her tespit edilen kişi için işlem
        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            cx = int(x1 + (x2-x1)/2)
            cy = int(y1 + (y2-y1)/2)
            
            # Kişi takibi ve sayım
            if id not in self.trackedObjects:
                self.trackedObjects[id] = (cx, cy)
            else:
                prev_pos = self.trackedObjects[id]
                current_pos = (cx, cy)
                
                # Giriş-çıkış kontrolü
                if prev_pos[1] < self.limits[1] and cy >= self.limits[1]:
                    if id not in self.enteredIDs:
                        self.enteredIDs.add(id)
                        if id not in self.totalCount:
                            self.totalCount.append(id)
                elif prev_pos[1] >= self.limits[1] and cy < self.limits[1]:
                    if id not in self.exitedIDs:
                        self.exitedIDs.add(id)
                
                # Hız hesaplama
                speed = self.calculate_speed(prev_pos, current_pos, time_elapsed)
                self.tracked_speeds[id] = speed
                
            self.trackedObjects[id] = (cx, cy)
            
            # Duygu analizi
            try:
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    emotion_result = self.emotion_model.top_emotion(face_crop)
                    if emotion_result is not None:
                        emotion, score = emotion_result
                        self.tracked_emotions[id] = emotion
            except:
                self.tracked_emotions[id] = "Unknown"
            
            # Görselleştirme
            cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), l=9, rt=2, colorR=(255, 0, 255))
            emotion_text = self.tracked_emotions.get(id, "Unknown")
            speed_text = f"{self.tracked_speeds.get(id, 0):.1f}"
            text = f'{emotion_text} | {speed_text} px/s'
            cvzone.putTextRect(frame, text, (max(0, x1), max(35, y1)), 
                             scale=1, thickness=1, offset=5)
        
        # Sayım bilgilerini ekrana yazma
        cv2.putText(frame, f'Giris: {len(self.enteredIDs)}', (15, 50), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 2)
        cv2.putText(frame, f'Cikis: {len(self.exitedIDs)}', (15, 125), 
                   cv2.FONT_HERSHEY_PLAIN, 2, (50, 50, 255), 2)
        
        return frame, current_time
    
    def run(self):
        prev_time = time.time()
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break
            
            # Frame işleme 
            processed_frame, prev_time = self.process_frame(frame, prev_time)
            
            # Video yazma ve gösterme
            self.video.write(processed_frame)
            cv2.imshow("Image", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        self.cap.release()
        self.video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "/Users/gunes/Desktop/Okul/Bitirme_Projesi/Final-project/Videolar/son.mp4"
    mask_path = "/Users/gunes/Desktop/Okul/Bitirme_Projesi/Final-project/Maske/cepmaske.jpg"
    tracker = PersonTracker(video_path, mask_path)
    tracker.run()