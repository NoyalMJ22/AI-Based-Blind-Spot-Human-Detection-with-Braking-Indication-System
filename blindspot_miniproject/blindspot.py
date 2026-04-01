# ---------------------------------------------------------
# Bus Blind Spot Human Detection System
# YOLOv8 + Arduino + Sound Alert + LED/Buzzer Control
# ALERT WHEN HUMAN WITHIN 200 CM
# MIRRORED CAMERA VIEW
# ---------------------------------------------------------

import cv2
from ultralytics import YOLO
import pygame
import serial
import time
import os

# -------------------------------
# ARDUINO SERIAL SETUP
# -------------------------------

try:
    arduino = serial.Serial('COM5', 9600)   # CHANGE THIS to your Arduino COM port
    time.sleep(2)
    print("Arduino Connected Successfully")
except:
    arduino = None
    print("Arduino Not Connected - Running in Software Mode Only")

# -------------------------------
# Load YOLOv8 model
# -------------------------------

model = YOLO("yolov8n.pt")

# -------------------------------
# Open Webcam
# -------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# -------------------------------
# Distance Calibration
# -------------------------------

KNOWN_PERSON_HEIGHT_CM = 170
FOCAL_LENGTH = 700

ALERT_DISTANCE_CM = 200

# -------------------------------
# AUDIO SETUP WITH SAFE PATH
# -------------------------------

pygame.mixer.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(BASE_DIR, "alert_final.wav")

if os.path.exists(sound_path):
    alert_sound = pygame.mixer.Sound(sound_path)
    print("Sound File Loaded Successfully")
else:
    alert_sound = None
    print("Warning: alert_final.wav NOT FOUND - Sound disabled")

sound_playing = False

# -------------------------------
# Distance Function
# -------------------------------

def estimate_distance(box_height_pixels):
    if box_height_pixels <= 0:
        return 99999

    distance_cm = (KNOWN_PERSON_HEIGHT_CM * FOCAL_LENGTH) / box_height_pixels
    return int(distance_cm)

# -------------------------------
# Display Setup
# -------------------------------

print("🚍 Blind Spot Human Detection Started")
print("🔊 ALERT when human detected WITHIN 200 cm")
print("📷 Camera view is LATERALLY INVERTED (Mirror Mode)")
print("Press 'Q' to exit")

cv2.namedWindow("Bus Blind Spot Human Detection", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(
    "Bus Blind Spot Human Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# -------------------------------
# MAIN LOOP
# -------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    # Mirror view
    frame = cv2.flip(frame, 1)

    results = model(frame)
    human_inside_range = False

    for result in results:
        for box in result.boxes:

            if int(box.cls[0]) == 0:   # Person class

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_height = y2 - y1

                distance_cm = estimate_distance(box_height)

                if distance_cm <= ALERT_DISTANCE_CM:
                    label = f"⚠ HUMAN {distance_cm} cm (ALERT)"
                    color = (0, 0, 255)
                    human_inside_range = True
                else:
                    label = f"Human {distance_cm} cm"
                    color = (0, 255, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

    # -------------------------------
    # ARDUINO ALERT CONTROL
    # -------------------------------

    if arduino is not None:
        try:
            if human_inside_range:
                arduino.write(b'1')
            else:
                arduino.write(b'0')
        except:
            pass

    # -------------------------------
    # SOFTWARE AUDIO ALERT (BACKUP)
    # -------------------------------

    if alert_sound is not None:

        if human_inside_range and not sound_playing:
            alert_sound.play(-1)
            sound_playing = True

        if not human_inside_range and sound_playing:
            alert_sound.stop()
            sound_playing = False

    cv2.imshow("Bus Blind Spot Human Detection", frame)

    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
        break

# -------------------------------
# CLEANUP
# -------------------------------

if alert_sound is not None:
    alert_sound.stop()

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()

print("System Closed Successfully")
