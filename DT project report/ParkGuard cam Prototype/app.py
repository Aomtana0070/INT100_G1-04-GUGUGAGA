import cv2
import time
import torch
import json
import asyncio
import requests
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO
import threading

# Configuration

DISCORD_WEBHOOK_URL = "https://discordapp.com/api/webhooks/1516846286819823679/ks8qqWL63un4gL8ObKkhnY5gWBzdWwmpZzYXJVs5ByXqA65gs-oYBWBmDK4pMHwuJCF2"

FALL_DURATION_THRESHOLD = 120.0  
COOLDOWN_TIME = 100.0           

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8n-pose.pt")

camera_lock = threading.Lock()

# Dynamic ROI & Features
roi_coords = [(200, 150), (1080, 600)]  # [(x1, y1), (x2, y2)]
use_full_frame = False                  # ใช้ ROI Zone เป็นค่าเริ่มต้น
privacy_blur = True                     # ใช้ Smart Face Blur เป็นค่าเริ่มต้น

incidents_log = []
today_incidents_count = 0
fall_start_time = None
last_alert_time = 0

app = FastAPI(title="ParkGuard Command Center")


# Privacy masking based on detected pose landmarks

def apply_smart_face_blur(frame, kpts, box):
    h_frame, w_frame = frame.shape[:2]
    face_kpts = kpts[:5] if len(kpts) >= 5 else []
    valid_pts = [pt[:2] for pt in face_kpts if len(pt) >= 2 and (len(pt) < 3 or pt[2] > 0.3)]
    
    if len(valid_pts) >= 2:
        pts = np.array(valid_pts, dtype=np.int32)
        fx1, fy1 = np.min(pts, axis=0)
        fx2, fy2 = np.max(pts, axis=0)
        
        face_w = max(fx2 - fx1, 40)
        face_h = max(fy2 - fy1, 40)
        margin_x = int(face_w * 0.9)
        margin_y = int(face_h * 1.1)
        
        bx1 = max(0, fx1 - margin_x)
        by1 = max(0, fy1 - margin_y)
        bx2 = min(w_frame, fx2 + margin_x)
        by2 = min(h_frame, fy2 + margin_y)
    else:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        bx1 = max(0, x1)
        by1 = max(0, y1)
        bx2 = min(w_frame, x2)
        by2 = min(h_frame, y1 + int(bh * 0.35))
        
    face_roi = frame[by1:by2, bx1:bx2]
    if face_roi.size > 0:
        blurred = cv2.GaussianBlur(face_roi, (99, 99), 30)
        frame[by1:by2, bx1:bx2] = blurred

def send_discord_alert(snapshot_frame, message):
    if DISCORD_WEBHOOK_URL == "YOUR_DISCORD_WEBHOOK_URL_HERE":
        return
    try:
        _, img_encoded = cv2.imencode('.jpg', snapshot_frame)
        files = {"file": ("fall_event.jpg", img_encoded.tobytes(), "image/jpeg")}
        payload = {"content": message}
        requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files, timeout=5)
    except Exception as e:
        print(f"[ERROR] Discord Send Failed: {e}")


# Video stream and detection pipeline

def generate_camera_stream():
    global fall_start_time, last_alert_time, today_incidents_count, roi_coords, use_full_frame, privacy_blur
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        with camera_lock:
            ret, frame = cap.read()
            if not ret:
                break

            current_fall_detected = False
            results = model(frame, device=device, verbose=False)

            # วาดกรอบ ROI Risk Zone
            if not use_full_frame and roi_coords and len(roi_coords) == 2:
                rx1, ry1 = min(roi_coords[0][0], roi_coords[1][0]), min(roi_coords[0][1], roi_coords[1][1])
                rx2, ry2 = max(roi_coords[0][0], roi_coords[1][0]), max(roi_coords[0][1], roi_coords[1][1])
                
                # Overlay พื้นหลังส้มโปร่งแสง
                overlay = frame.copy()
                cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 140, 255), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                
                # เส้นขอบ ROI และข้อความเตือน
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 140, 255), 2)
                cv2.putText(frame, "[CUSTOM ROI RISK ZONE ACTIVE]", (rx1 + 15, ry1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)

            for result in results:
                if result.boxes is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints_data = result.keypoints.data.cpu().numpy() if result.keypoints is not None else []

                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    w, h = x2 - x1, y2 - y1
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # ตรวจสอบว่าอยู่ภายใน ROI หรือไม่
                    if not use_full_frame and roi_coords and len(roi_coords) == 2:
                        rx1, ry1 = min(roi_coords[0][0], roi_coords[1][0]), min(roi_coords[0][1], roi_coords[1][1])
                        rx2, ry2 = max(roi_coords[0][0], roi_coords[1][0]), max(roi_coords[0][1], roi_coords[1][1])
                        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                            continue

                    # PDPA Privacy Blur
                    if privacy_blur:
                        kpts = keypoints_data[idx] if idx < len(keypoints_data) else []
                        apply_smart_face_blur(frame, kpts, (x1, y1, x2, y2))

                    # Fall Detection Logic
                    aspect_ratio = w / float(h) if h > 0 else 0
                    if aspect_ratio > 1.1:
                        current_fall_detected = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        cv2.putText(frame, "FALL DETECTED!", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # จับเวลาเหตุการณ์คนหกล้ม
            curr_time = time.time()
            if current_fall_detected:
                if fall_start_time is None:
                    fall_start_time = curr_time
                
                elapsed = curr_time - fall_start_time
                cv2.putText(frame, f"FALL TIMER: {elapsed:.1f}s", (40, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                if elapsed >= FALL_DURATION_THRESHOLD and (curr_time - last_alert_time) > COOLDOWN_TIME:
                    last_alert_time = curr_time
                    today_incidents_count += 1
                    timestamp = time.strftime('%H:%M:%S')
                    
                    incidents_log.insert(0, {
                        "time": timestamp,
                        "location": "CAM-01 (Custom ROI)",
                        "event": "Fall & Stay Detected",
                        "severity": "CRITICAL",
                        "status": "ALERT SENT"
                    })

                    alert_msg = f"🚨 **[ParkGuard Alert] ตรวจพบคนหกล้มในพื้นที่เสี่ยง!**\n⏱ **เวลา:** {timestamp}\n⏳ **ล้มนิ่งค้างไว้:** {elapsed:.1f} วินาที"
                    threading.Thread(target=send_discord_alert, args=(frame.copy(), alert_msg)).start()
            else:
                fall_start_time = None

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# API routes and monitoring dashboard

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_camera_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/settings")
async def update_settings(data: dict):
    global use_full_frame, privacy_blur, roi_coords
    if "use_full_frame" in data:
        use_full_frame = data["use_full_frame"]
    if "privacy_blur" in data:
        privacy_blur = data["privacy_blur"]
    if "roi" in data:
        # รับค่าพิกัดพิกเซล [(x1, y1), (x2, y2)] จากหน้าเว็บ
        roi_coords = [(int(data["roi"][0][0]), int(data["roi"][0][1])), 
                      (int(data["roi"][1][0]), int(data["roi"][1][1]))]
    return {"status": "success", "roi": roi_coords}

@app.get("/api/data")
async def get_dashboard_data():
    return {
        "today_count": today_incidents_count,
        "device": device.upper(),
        "incidents": incidents_log[:10]
    }

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ParkGuard | Command Center</title>
    <style>
        :root {
            --bg: #0b1020;
            --bg-elevated: #111827;
            --panel: #151d2a;
            --panel-strong: #1c2434;
            --border: rgba(148, 163, 184, 0.18);
            --text: #e5edf7;
            --muted: #8aa0ba;
            --orange: #f59e0b;
            --orange-soft: rgba(245, 158, 11, 0.12);
            --green: #22c55e;
            --green-soft: rgba(34, 197, 94, 0.12);
            --red: #ef4444;
            --red-soft: rgba(239, 68, 68, 0.12);
            --blue: #38bdf8;
            --shadow: 0 18px 40px rgba(15, 23, 42, 0.32);
        }

        * { box-sizing: border-box; }
        html, body {
            margin: 0;
            padding: 0;
            background: radial-gradient(circle at top, #101a2d 0%, var(--bg) 40%, #090d18 100%);
            color: var(--text);
            font-family: "Segoe UI", Tahoma, sans-serif;
        }
        body {
            min-height: 100vh;
            padding: 22px;
        }

        .topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 16px 20px;
            background: rgba(21, 29, 42, 0.82);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
        }

        .brand { display: flex; align-items: center; gap: 14px; }
        .brand-mark {
            width: 40px; height: 40px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg, #f59e0b, #ea580c);
            color: #fff;
            font-weight: 800;
            letter-spacing: 0.08em;
        }
        .brand h1 {
            margin: 0;
            font-size: 1.1rem;
            font-weight: 700;
        }
        .brand p {
            margin: 3px 0 0;
            color: var(--muted);
            font-size: 0.74rem;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(34, 197, 94, 0.08);
            border: 1px solid rgba(34, 197, 94, 0.25);
            color: #a7f3d0;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }
        .status-dot {
            width: 8px; height: 8px;
            border-radius: 50%;
            background: var(--green);
            box-shadow: 0 0 12px rgba(34, 197, 94, 0.8);
        }

        .stats {
            margin-top: 18px;
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
        }
        .card {
            background: rgba(21, 29, 42, 0.85);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: var(--shadow);
        }
        .metric {
            border-left: 4px solid var(--orange);
            padding-left: 14px;
        }
        .metric.success { border-left-color: var(--green); }
        .metric.alert { border-left-color: var(--red); }
        .metric-label {
            font-size: 0.68rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 700;
        }
        .metric-value {
            margin-top: 8px;
            font-size: clamp(1.1rem, 1.5vw, 1.65rem);
            font-weight: 800;
            letter-spacing: -0.03em;
            line-height: 1.2;
        }

        .content {
            margin-top: 18px;
            display: grid;
            grid-template-columns: 1.7fr 1fr;
            gap: 18px;
        }
        .panel {
            background: rgba(21, 29, 42, 0.85);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--shadow);
            padding: 16px;
        }

        .panel-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 14px;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--muted);
            font-weight: 700;
        }

        .video-box {
            position: relative;
            width: 100%;
            aspect-ratio: 16 / 9;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        .video-box img {
            width: 100%;
            height: 100%;
            display: block;
            object-fit: contain;
            background: #050b12;
        }
        #roiCanvas {
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 2;
        }
        #roiCanvas.drawing-mode { pointer-events: auto; cursor: crosshair; }

        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
        }
        .btn {
            border: 1px solid rgba(148, 163, 184, 0.26);
            border-radius: 10px;
            background: var(--panel-strong);
            color: var(--text);
            padding: 10px 14px;
            font-size: 0.8rem;
            font-weight: 700;
            transition: 0.2s ease;
            cursor: pointer;
        }
        .btn:hover { transform: translateY(-1px); }
        .btn.active {
            background: rgba(245, 158, 11, 0.16);
            border-color: rgba(245, 158, 11, 0.5);
            color: #fcd34d;
        }
        .btn.draw {
            background: rgba(56, 189, 248, 0.12);
            border-color: rgba(56, 189, 248, 0.35);
            color: #bae6fd;
        }
        .btn.draw.active {
            background: rgba(239, 68, 68, 0.12);
            border-color: rgba(239, 68, 68, 0.5);
            color: #fecaca;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.78rem;
        }
        th {
            text-align: left;
            padding: 10px 8px;
            color: var(--muted);
            font-size: 0.66rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(148, 163, 184, 0.15);
        }
        td {
            padding: 10px 8px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            color: var(--text);
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.66rem;
        }
        .badge-red { background: var(--red-soft); color: #fca5a5; }
        .badge-green { background: var(--green-soft); color: #bbf7d0; }
        .empty {
            text-align: center;
            color: var(--muted);
            padding: 18px 8px;
        }

        @media (max-width: 960px) {
            .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .content { grid-template-columns: 1fr; }
        }
        @media (max-width: 560px) {
            body { padding: 14px; }
            .topbar { flex-direction: column; align-items: flex-start; }
            .stats { grid-template-columns: 1fr; }
            .controls { flex-direction: column; }
            .btn { width: 100%; }
        }
    </style>
</head>
<body>
    <header class="topbar">
        <div class="brand">
            <div class="brand-mark">PG</div>
            <div>
                <h1>ParkGuard Command Center</h1>
                <p>Site monitoring and incident response</p>
            </div>
        </div>
        <div class="status-pill"><span class="status-dot"></span>System online</div>
    </header>

    <section class="stats">
        <div class="card metric">
            <div class="metric-label">Camera</div>
            <div class="metric-value">CAM-01</div>
        </div>
        <div class="card metric success">
            <div class="metric-label">Processing Unit</div>
            <div class="metric-value" id="deviceVal">GPU</div>
        </div>
        <div class="card metric alert">
            <div class="metric-label">Today Alerts</div>
            <div class="metric-value" id="todayIncidents">0</div>
        </div>
        <div class="card metric">
            <div class="metric-label">Alert Window</div>
            <div class="metric-value">3.0s</div>
        </div>
    </section>

    <main class="content">
        <section class="panel">
            <div class="panel-header">
                <span>Live feed</span>
                <span id="roiStatusText" style="color: #fbbf24;">Monitoring area</span>
            </div>

            <div class="video-box">
                <img id="videoStream" src="/video_feed" alt="Security camera live feed">
                <canvas id="roiCanvas"></canvas>
            </div>

            <div class="controls">
                <button class="btn draw" id="btnDraw" onclick="toggleDrawMode()">Draw risk zone</button>
                <button class="btn active" id="btnRoiMode" onclick="toggleRoiMode()">Monitoring area</button>
                <button class="btn active" id="btnPDPA" onclick="togglePDPA()">Privacy masking: ON</button>
            </div>
        </section>

        <aside class="panel">
            <div class="panel-header">
                <span>Incident log</span>
            </div>
            <div style="overflow-y:auto; max-height: 420px;">
                <table>
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Location</th>
                            <th>Event</th>
                            <th>Severity</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="logTable">
                        <tr><td colspan="5" class="empty">Waiting for monitoring data...</td></tr>
                    </tbody>
                </table>
            </div>
        </aside>
    </main>

    <script>
        let isFullFrame = false;
        let isPdpa = true;
        let isDrawingMode = false;
        let isMouseDown = false;
        let startX = 0, startY = 0;

        const canvas = document.getElementById('roiCanvas');
        const ctx = canvas.getContext('2d');
        const img = document.getElementById('videoStream');

        function syncCanvasSize() {
            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
        }

        window.addEventListener('resize', syncCanvasSize);
        img.onload = syncCanvasSize;

        function toggleDrawMode() {
            syncCanvasSize();
            isDrawingMode = !isDrawingMode;
            const btn = document.getElementById('btnDraw');

            if (isDrawingMode) {
                canvas.classList.add('drawing-mode');
                btn.innerText = 'Cancel selection';
                btn.classList.add('active');
            } else {
                canvas.classList.remove('drawing-mode');
                btn.innerText = 'Draw risk zone';
                btn.classList.remove('active');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }

        canvas.addEventListener('mousedown', (e) => {
            if (!isDrawingMode) return;
            const rect = canvas.getBoundingClientRect();
            startX = e.clientX - rect.left;
            startY = e.clientY - rect.top;
            isMouseDown = true;
        });

        canvas.addEventListener('mousemove', (e) => {
            if (!isMouseDown || !isDrawingMode) return;
            const rect = canvas.getBoundingClientRect();
            const currX = e.clientX - rect.left;
            const currY = e.clientY - rect.top;

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = 'rgba(245, 158, 11, 0.22)';
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 2;

            const w = currX - startX;
            const h = currY - startY;
            ctx.fillRect(startX, startY, w, h);
            ctx.strokeRect(startX, startY, w, h);
        });

        canvas.addEventListener('mouseup', (e) => {
            if (!isMouseDown || !isDrawingMode) return;
            isMouseDown = false;

            const rect = canvas.getBoundingClientRect();
            const endX = e.clientX - rect.left;
            const endY = e.clientY - rect.top;

            const scaleX = 1280 / canvas.width;
            const scaleY = 720 / canvas.height;
            const x1 = Math.round(Math.min(startX, endX) * scaleX);
            const y1 = Math.round(Math.min(startY, endY) * scaleY);
            const x2 = Math.round(Math.max(startX, endX) * scaleX);
            const y2 = Math.round(Math.max(startY, endY) * scaleY);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            toggleDrawMode();

            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    use_full_frame: false,
                    roi: [[x1, y1], [x2, y2]]
                })
            });

            isFullFrame = false;
            document.getElementById('btnRoiMode').innerText = 'Custom area';
            document.getElementById('btnRoiMode').classList.add('active');
        });

        async function updateDashboardData() {
            try {
                const res = await fetch('/api/data');
                const data = await res.json();
                document.getElementById('todayIncidents').innerText = data.today_count;
                document.getElementById('deviceVal').innerText = data.device;

                if (data.incidents && data.incidents.length > 0) {
                    const tbody = document.getElementById('logTable');
                    tbody.innerHTML = data.incidents.map(i => `
                        <tr>
                            <td>${i.time}</td>
                            <td>${i.location}</td>
                            <td>${i.event}</td>
                            <td><span class="badge badge-red">${i.severity}</span></td>
                            <td><span class="badge badge-green">${i.status}</span></td>
                        </tr>
                    `).join('');
                }
            } catch (e) {}
        }

        setInterval(updateDashboardData, 1500);

        function toggleRoiMode() {
            isFullFrame = !isFullFrame;
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ use_full_frame: isFullFrame })
            });
            const btn = document.getElementById('btnRoiMode');
            btn.innerText = isFullFrame ? 'Full frame' : 'Custom area';
            btn.classList.toggle('active', !isFullFrame);
        }

        function togglePDPA() {
            isPdpa = !isPdpa;
            fetch('/api/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ privacy_blur: isPdpa })
            });
            const btn = document.getElementById('btnPDPA');
            btn.innerText = isPdpa ? 'Privacy masking: ON' : 'Privacy masking: OFF';
            btn.classList.toggle('active', isPdpa);
        }
    </script>
</body>
</html>
    """