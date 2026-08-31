import cv2
import time
import torch
import requests
from ultralytics import YOLO


# CONFIGURATION

#Discord  Webhook URL 
DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1516846286819823679/ks8qqWL63un4gL8ObKkhnY5gWBzdWwmpZzYXJVs5ByXqA65gs-oYBWBmDK4pMHwuJCF2"  

FALL_DURATION_THRESHOLD = 3.0  # จำนวนวินาทีที่ล้มนิ่งก่อนแจ้งเตือน
COOLDOWN_TIME = 10.0           # เว้นระยะก่อนส่งแจ้งเตือนซ้ำ


# SETUP & INITIALIZATION

# ตรวจจับ NVIDIA GPU (CUDA) อัตโนมัติ หากไม่มีจะสลับไป CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Running on device: {device}")

# โหลด YOLOv8 Nano Pose Model (เบา ละเอียด เหมาะกับ Laptop)
model = YOLO("yolov8n-pose.pt")

# ตัวแปรสำหรับ ROI
roi_pts = []
drawing_roi = False
use_full_frame = True  # ค่าเริ่มต้น: ตรวจจับทั้งจอ (Full Frame)

def mouse_callback(event, x, y, flags, param):
    global roi_pts, drawing_roi, use_full_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_roi = True
        roi_pts = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing_roi:
        if len(roi_pts) > 1:
            roi_pts[1] = (x, y)
        else:
            roi_pts.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing_roi = False
        if len(roi_pts) > 1:
            roi_pts[1] = (x, y)
            use_full_frame = False  # สลับไปใช้ ROI อัตโนมัติเมื่อวาดเสร็จ
            print(f"[INFO] ROI Defined: {roi_pts}")

def is_inside_roi(box, roi):
    if use_full_frame or len(roi) < 2:
        return True
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    rx1, ry1 = min(roi[0][0], roi[1][0]), min(roi[0][1], roi[1][1])
    rx2, ry2 = max(roi[0][0], roi[1][0]), max(roi[0][1], roi[1][1])
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

def send_discord_alert(webhook_url, image_path, message):
    if webhook_url == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        print("[WARNING] กรุณาใส่ DISCORD_WEBHOOK_URL ในโค้ดก่อนใช้งานระบบแจ้งเตือน")
        return
    try:
        with open(image_path, "rb") as f:
            payload = {"content": message}
            files = {"file": (image_path, f, "image/jpeg")}
            res = requests.post(webhook_url, data=payload, files=files)
            if res.status_code in [200, 204]:
                print("[INFO] ส่งภาพแจ้งเตือนเข้า Discord สำเร็จ")
            else:
                print(f"[ERROR] ไม่สามารถส่งภาพได้ Status: {res.status_code}")
    except Exception as e:
        print(f"[ERROR] Discord Send Error: {e}")


# MAIN LOOP

cap = cv2.VideoCapture(0)  # ดึงภาพจาก Webcam
cv2.namedWindow("Park Accident Detection")
cv2.setMouseCallback("Park Accident Detection", mouse_callback)
cap = cv2.VideoCapture(0)

# กำหนดความกว้างและความสูง 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


fall_start_time = None
last_alert_time = 0

print("\n--- ปุ่มควบคุม ---")
print("คลิกลากเมาส์ : วาดพื้นที่เสี่ยง (ROI)")
print("กด 'f'       : สลับใช้ Full Frame / ROI")
print("กด 'c'       : ล้างพื้นที่ ROI")
print("กด 'q'       : ออกจากโปรแกรม\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ประมวลผลด้วย YOLOv8-Pose
    results = model(frame, device=device, verbose=False)
    current_fall_detected = False

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1

            # กรองเฉพาะจุดที่อยู่ใน ROI
            if not is_inside_roi([x1, y1, x2, y2], roi_pts):
                continue

            # Fall Logic: ความกว้างร่างกายนอนขนานกับพื้น (W > H * 1.1)
            aspect_ratio = w / float(h) if h > 0 else 0
            if aspect_ratio > 1.1:
                current_fall_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "FALL DETECTED!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # จับเวลากรณีคนล้มนิ่ง
    current_time = time.time()
    if current_fall_detected:
        if fall_start_time is None:
            fall_start_time = current_time
        
        elapsed_time = current_time - fall_start_time
        cv2.putText(frame, f"Fall Timer: {elapsed_time:.1f}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # ล้มนิ่งเกินกำหนด และพ้นช่วง Cooldown ให้ส่ง Discord
        if elapsed_time >= FALL_DURATION_THRESHOLD and (current_time - last_alert_time) > COOLDOWN_TIME:
            snapshot_path = "fall_snapshot.jpg"
            cv2.imwrite(snapshot_path, frame)
            
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            alert_msg = f"🚨 **ตรวจพบอุบัติเหตุคนหกล้มในสวน!**\n⏱ **เวลา:** {timestamp}\n⏳ **ล้มนิ่งค้างไว้:** {elapsed_time:.1f} วินาที"
            
            send_discord_alert(DISCORD_WEBHOOK_URL, snapshot_path, alert_msg)
            last_alert_time = current_time
    else:
        fall_start_time = None

    # วาดกรอบแสดงผลสถานะ ROI บนหน้าจอ
    if not use_full_frame and len(roi_pts) == 2:
        cv2.rectangle(frame, roi_pts[0], roi_pts[1], (255, 255, 0), 2)
        cv2.putText(frame, "Mode: ROI Active", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    else:
        cv2.putText(frame, "Mode: Full Frame", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Park Accident Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        use_full_frame = not use_full_frame
        print(f"[INFO] Toggled Full Frame: {use_full_frame}")
    elif key == ord('c'):
        roi_pts = []
        use_full_frame = True
        print("[INFO] Cleared ROI")

cap.release()
cv2.destroyAllWindows()