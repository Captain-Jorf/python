import cv2
import dlib
import numpy as np
import pyautogui

# تنظیمات اولیه
EYE_POINTS_LEFT = list(range(36, 42))  # نقاط چشم چپ
MOUTH_OPEN_THRESHOLD = 8               # آستانه باز بودن دهان
SMOOTHING_WINDOW = 5                   # نرم کردن حرکت
ACTIVE_ZONE_RATIO = 0.4                # منطقه فعال
ZONE_STEP = 0.05                        # گام تغییر اندازه منطقه فعال
MIN_ZONE_RATIO = 0.1
MAX_ZONE_RATIO = 0.9

# صف برای نرم کردن حرکت
position_history = []

# بارگذاری مدل‌ها
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# دوربین
cap = cv2.VideoCapture(0)

# برای جلوگیری از کلیک مداوم
mouse_pressed = False

print("🎥 برنامه شروع شد.")
print("➕➖ برای تغییر اندازه منطقه فعال از دکمه‌های '+' و '-' استفاده کنید.")
print("🖱️ برای کلیک، دهانتان را باز کنید.")
print("❌ برای خروج 'q' را فشار دهید.")

def get_eye_center(eye_points):
    x_coords = [p.x for p in eye_points]
    y_coords = [p.y for p in eye_points]
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    return center_x, center_y

def is_mouth_open(landmarks):
    upper_lip = landmarks.part(62).y
    lower_lip = landmarks.part(66).y
    vertical_distance = abs(lower_lip - upper_lip)
    return vertical_distance > MOUTH_OPEN_THRESHOLD

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # گرفتن نقاط چشم چپ
        left_eye_points = [landmarks.part(i) for i in EYE_POINTS_LEFT]

        # محاسبه مرکز چشم چپ
        eye_center_x, eye_center_y = get_eye_center(left_eye_points)

        # ذخیره موقعیت قبلی برای نرم کردن حرکت
        position_history.append((eye_center_x, eye_center_y))
        if len(position_history) > SMOOTHING_WINDOW:
            position_history.pop(0)

        # محاسبه میانگین متحرک برای نرم کردن حرکت
        avg_pos = np.mean(position_history, axis=0)
        current_eye = avg_pos.astype(int)

        # رسم دایره روی چشم
        cv2.circle(frame, (current_eye[0], current_eye[1]), 5, (0, 255, 0), -1)

        # تعیین منطقه فعال (Active Zone)
        height, width = frame.shape[:2]
        screen_width, screen_height = pyautogui.size()

        active_zone_w = int(width * ACTIVE_ZONE_RATIO)
        active_zone_h = int(height * ACTIVE_ZONE_RATIO)

        center_x = width // 2
        center_y = height // 2

        x_start = center_x - active_zone_w // 2
        y_start = center_y - active_zone_h // 2
        x_end = x_start + active_zone_w
        y_end = y_start + active_zone_h

        # رسم مستطیل منطقه فعال
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 255, 0), 1)

        # محدود کردن موقعیت چشم به منطقه فعال
        rel_x = np.clip(current_eye[0], x_start, x_end)
        rel_y = np.clip(current_eye[1], y_start, y_end)

        # نگاشت به صفحه نمایش
        x_ratio = (rel_x - x_start) / active_zone_w
        y_ratio = (rel_y - y_start) / active_zone_h

        target_x = int(screen_width * (1 - x_ratio))  # برعکس کردن چپ/راست
        target_y = int(screen_height * y_ratio)

        # حرکت موس به صورت مطلق
        pyautogui.moveTo(target_x, target_y)

        # تشخیص باز بودن دهان و مدیریت کلیک
        if is_mouth_open(landmarks):
            if not mouse_pressed:
                pyautogui.mouseDown()
                mouse_pressed = True
                print("🖱️ کلیک فعال شد!")
        else:
            if mouse_pressed:
                pyautogui.mouseUp()
                mouse_pressed = False
                print("MouseClicked!")

    # نمایش تصویر
    cv2.imshow("Eye Controlled Mouse - Accessibility Mode", frame)

    # کنترل دکمه‌ها
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        ACTIVE_ZONE_RATIO = min(MAX_ZONE_RATIO, ACTIVE_ZONE_RATIO + ZONE_STEP)
        print(f"🟩 اندازه منطقه فعال: {int(ACTIVE_ZONE_RATIO * 100)}%")
    elif key == ord('-') or key == ord('_'):
        ACTIVE_ZONE_RATIO = max(MIN_ZONE_RATIO, ACTIVE_ZONE_RATIO - ZONE_STEP)
        print(f"🟥 اندازه منطقه فعال: {int(ACTIVE_ZONE_RATIO * 100)}%")

# پاکسازی
if mouse_pressed:
    pyautogui.mouseUp()

cap.release()
cv2.destroyAllWindows()