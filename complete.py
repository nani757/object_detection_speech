import cv2
import torch
import pyttsx3
import threading

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open video capture device.")
    exit()

def detect_objects():
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame)

        # Print the detected object classes and convert them to speech
        for i, (class_id, score, bbox) in enumerate(zip(results.pred[0][:, 5], results.pred[0][:, 4], results.pred[0][:, :4])):
            class_name = model.names[int(class_id)]
            print(f"Detected object: {class_name}")
            text_to_speech(class_name)

        cv2.imshow('Object Detection', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def text_to_speech(text):
    class SpeechEngineThread(threading.Thread):
        def __init__(self, text):
            threading.Thread.__init__(self)
            self.text = text

        def run(self):
            engine = pyttsx3.init()
            engine.say(self.text)
            engine.runAndWait()

    speech_thread = SpeechEngineThread(text)
    speech_thread.start()

if __name__ == '__main__':
    detect_objects()
