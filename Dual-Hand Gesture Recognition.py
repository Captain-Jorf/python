import cv2
import mediapipe as mp
import math
import pyautogui

# تنظیمات MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# دوربین
cap = cv2.VideoCapture(0)

# اندازه صفحه برای مپ کردن دست به ماوس
screen_width, screen_height = pyautogui.size()
frame_w, frame_h = int(screen_width * 0.5), int(screen_height * 0.5)

# متغیرهای کلیک
click_threshold = 30
click_active = False

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    mouse_hand_found = False
    click_hand_found = False
    index_tip = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label  # 'Left' or 'Right'

            thumb = hand_lms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index = hand_lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            cx_thumb, cy_thumb = int(thumb.x * w), int(thumb.y * h)
            cx_index, cy_index = int(index.x * w), int(index.y * h)

            cv2.circle(frame, (cx_index, cy_index), 5, (255, 0, 0), -1)

            # 🔁 جای دست چپ و راست عوض شده
            if hand_label == "Right":  # دست راست = حرکت موس
                mouse_hand_found = True
                index_tip = (cx_index, cy_index)
                cv2.circle(frame, (cx_index, cy_index), 10, (0, 255, 255), -1)

                x_mouse = int(screen_width / w * cx_index)
                y_mouse = int(screen_height / h * cy_index)
                pyautogui.moveTo(x_mouse, y_mouse)

            elif hand_label == "Left":  # دست چپ = کلیک
                click_hand_found = True
                distance = calculate_distance((cx_thumb, cy_thumb), (cx_index, cy_index))

                if distance < click_threshold:
                    cv2.line(frame, (cx_thumb, cy_thumb), (cx_index, cy_index), (0, 0, 255), 2)
                    if not click_active:
                        print("Click!")
                        pyautogui.click()
                        click_active = True
                else:
                    click_active = False

            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    # نمایش وضعیت
    status_text = "Tracking"
    if not mouse_hand_found:
        status_text += " | Mouse Hand: ❌"
    else:
        status_text += " | Mouse Hand: ✅"

    if not click_hand_found:
        status_text += " | Click Hand: ❌"
    else:
        status_text += " | Click Hand: ✅"

    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # نمایش
    cv2.imshow("Mouse Control with Right Hand | Click with Left", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# پاک‌کاری
cap.release()
cv2.destroyAllWindows()