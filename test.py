
# import cv2
# import torch

# # Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# # Open the video capture device
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Failed to open video capture device.")
#     exit()

# while True:
#     # Read the next frame from the video capture device
#     ret, frame = cap.read()

#     if not ret:
#         print("Failed to capture frame.")
#         break

#     # Convert the frame from BGR to RGB
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Perform object detection on the frame
#     results = model(frame)

#     # Print the detected object classes and confidence scores to the terminal
#     for i, (class_id, score, bbox) in enumerate(zip(results.pred[0][:, 5], results.pred[0][:, 4], results.pred[0][:, :4])):
#         print(f"Detected object: class_id={class_id}, score={score}")

#         # Draw the bounding box and label on the frame
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#         cv2.putText(frame, f"Class ID: {int(class_id)}, Score: {score:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Show the resulting frame
#     cv2.imshow('Object Detection', frame)

#     # Wait for key press or exit
#     if cv2.waitKey(1) == ord('q'):
#         break

# # Clean up
# cap.release()
# cv2.destroyAllWindows()

import cv2
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open video capture device.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame.")
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)

    # Print the detected object classes and confidence scores
    for i, (class_id, score, bbox) in enumerate(zip(results.pred[0][:, 5], results.pred[0][:, 4], results.pred[0][:, :4])):
        class_name = model.names[int(class_id)]
        print(f"Detected object: {class_name}, score={score}")

        # Draw the box and label on the frame
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {score:.2f}", (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
