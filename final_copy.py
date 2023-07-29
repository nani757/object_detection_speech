import cv2
import torch
import pyttsx3
import threading

# Initialize the speech synthesis engine
speech_engine = pyttsx3.init()

# Define a helper function to speak a given text
def speak(text):
    speech_thread = threading.Thread(target=speech_engine.say, args=(text,))
    speech_thread.start()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open video capture device.")
    exit()

# Main loop for object detection
while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break

    # Convert the color space from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection on the frame
    results = model(frame)

    # Loop over the detected objects
    for i, (class_id, score, bbox) in enumerate(zip(results.pred[0][:, 5], results.pred[0][:, 4], results.pred[0][:, :4])):
        class_name = model.names[int(class_id)]
        print(f"Detected object: {class_name}, score={score}")

        # Draw the box and label on the frame
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Speak the name of the detected object if it matches the desired object
        if class_name == "object_name":
            speak(f"Found {class_name}")

    # Show the annotated frame
    cv2.imshow('Object Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
