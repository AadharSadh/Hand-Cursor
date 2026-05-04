from ultralytics import YOLO
import cv2
import pyautogui

screen_w, screen_h = pyautogui.size()

model = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(3, 320)
cap.set(4, 240)

RIGHT_WRIST = 10

prev_x, prev_y = 0, 0
smoothening = 6
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    small = cv2.resize(frame, (320, 240))

    results = model(small, conf=0.3, verbose=False)

    display = frame.copy()

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy

        for person in keypoints:
            wrist = person[RIGHT_WRIST]

            x = int(wrist[0] * (w / 320))
            y = int(wrist[1] * (h / 240))

            screen_x = int((x / w) * screen_w)
            screen_y = int((y / h) * screen_h)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            cv2.circle(display, (x, y), 10, (0, 255, 0), -1)

    cv2.imshow("Fast Cursor Control", display)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()