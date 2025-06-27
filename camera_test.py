import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    ret, frame = cap.read()
    if ret:
        print("Success: Camera is accessible and a frame was read.")
    else:
        print("Error: Could not read frame from camera.")
    cap.release()