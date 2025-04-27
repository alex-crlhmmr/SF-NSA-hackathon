import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from ultralytics import YOLO
import base64
import time
import threading
import serial


# Load YOLOv8 model for human detection (COCO class 0 = person)
model = YOLO('yolov8m.pt').to('cuda')

# Initialize FastAPI app and camera
app = FastAPI()
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
camera.set(cv2.CAP_PROP_FPS, 30)

# GPS setup
SERIAL_PORT = '/dev/ttyACM0'  # Change to 'COM3' or similar if on Windows
BAUDRATE = 9600
HDOP_THRESHOLD = 4.0  # Horizontal accuracy (speed, course)
VDOP_THRESHOLD = 4.0  # Vertical accuracy (altitude)

# Global frame buffer and lock for thread-safe access
latest_frame = None
frame_lock = threading.Lock()
camera_running = True

# Global GPS data and lock for thread-safe access
gps_data = {}
gps_lock = threading.Lock()
gps_running = True
serial_connection = None

# Background thread to continuously read camera frames
def camera_reader():
    global latest_frame, camera_running
    while camera_running:
        success, frame = camera.read()
        if success:
            with frame_lock:
                latest_frame = frame.copy()
        time.sleep(0.01)  # Prevent excessive CPU usage
camera_thread = threading.Thread(target=camera_reader, daemon=True)
camera_thread.start()

# GPS parsing functions
def parse_GGA(parts):
    """Parse $GPGGA sentence for position and time data."""
    try:
        if len(parts) < 15 or parts[0] != '$GPGGA':
            return {}
        utc_time = parts[1]
        lat = parts[2]
        lat_dir = parts[3]
        lon = parts[4]
        lon_dir = parts[5]
        fix = parts[6]
        num_satellites = parts[7]
        altitude = parts[9]

        # Convert latitude
        lat_deg = int(float(lat) / 100)
        lat_min = float(lat) - lat_deg * 100
        latitude = lat_deg + lat_min / 60.0
        if lat_dir == 'S':
            latitude = -latitude

        # Convert longitude
        lon_deg = int(float(lon) / 100)
        lon_min = float(lon) - lon_deg * 100
        longitude = lon_deg + lon_min / 60.0
        if lon_dir == 'W':
            longitude = -longitude

        time_str = f"{utc_time[0:2]}:{utc_time[2:4]}:{utc_time[4:6]}" if utc_time else "--"
        fix_status = "Yes" if fix in ['1', '2'] else "No"

        return {
            'UTC Time': time_str,
            'Latitude': round(latitude, 6),
            'Longitude': round(longitude, 6),
            'Fix': fix_status,
            'Satellites Used': num_satellites,
            'Altitude (m)': altitude if altitude else "--"
        }
    except Exception:
        return {}

def parse_RMC(parts):
    """Parse $GPRMC sentence for speed, course, and date."""
    try:
        if len(parts) < 10 or parts[0] != '$GPRMC':
            return {}
        status = "Valid" if parts[2] == 'A' else "Invalid"
        speed_knots = float(parts[7]) if parts[7] else 0.0
        course = float(parts[8]) if parts[8] else 0.0
        date = parts[9]
        date_formatted = f"{date[0:2]}/{date[2:4]}/20{date[4:6]}" if date else "--"
        return {
            'Status': status,
            'Speed (km/h)': round(speed_knots * 1.852, 2),  # Convert knots to km/h
            'Course (deg)': round(course, 1),
            'Date': date_formatted
        }
    except Exception:
        return {}

def parse_VTG(parts):
    """Parse $GPVTG sentence for speed and course."""
    try:
        if len(parts) < 8 or parts[0] != '$GPVTG':
            return {}
        course = float(parts[1]) if parts[1] else 0.0
        speed_kph = float(parts[7]) if parts[7] else 0.0
        return {
            'Course (deg)': round(course, 1),
            'Speed (km/h)': round(speed_kph, 2)
        }
    except Exception:
        return {}

def parse_GSA(parts):
    """Parse $GPGSA sentence for fix type and DOP values."""
    try:
        if len(parts) < 18 or parts[0] != '$GPGSA':
            return {}
        fix_type = {'1': "No Fix", '2': "2D", '3': "3D"}.get(parts[2], "Unknown")
        pdop = parts[15] if parts[15] else "--"
        hdop = parts[16] if parts[16] else "--"
        vdop = parts[17] if parts[17] else "--"
        try:
            pdop_num = float(parts[15]) if parts[15] else float('inf')
        except (ValueError, TypeError):
            pdop_num = float('inf')
        try:
            hdop_num = float(parts[16]) if parts[16] else float('inf')
        except (ValueError, TypeError):
            hdop_num = float('inf')
        try:
            vdop_num = float(parts[17]) if parts[17] else float('inf')
        except (ValueError, TypeError):
            vdop_num = float('inf')
        return {
            'Fix Type': fix_type,
            'PDOP': pdop,
            'HDOP': hdop,
            'VDOP': vdop,
            'PDOP_num': pdop_num,
            'HDOP_num': hdop_num,
            'VDOP_num': vdop_num
        }
    except Exception:
        return {}

def assess_corrected_values(gps_data):
    """Assess and return corrected values based on DOP thresholds."""
    hdop = gps_data.get('HDOP_num', float('inf'))
    vdop = gps_data.get('VDOP_num', float('inf'))
    fix_status = gps_data.get('Fix', 'No')
    satellites = int(gps_data.get('Satellites Used', 0))

    # Check reliability
    is_reliable = (
        hdop <= HDOP_THRESHOLD and
        vdop <= VDOP_THRESHOLD and
        fix_status == 'Yes' and
        satellites >= 4
    )

    # Corrected values
    corrected = {
        'Altitude (m)': gps_data.get('Altitude (m)', '--') if is_reliable and vdop <= VDOP_THRESHOLD else '--',
        'Speed (km/h)': gps_data.get('Speed (km/h)', '--') if is_reliable and hdop <= HDOP_THRESHOLD else '--',
        'Course (deg)': gps_data.get('Course (deg)', '--') if is_reliable and hdop <= HDOP_THRESHOLD else '--'
    }

    return corrected

# Background thread to continuously read GPS data
def gps_reader():
    global gps_data, gps_running, serial_connection
    fix_acquired = False

    while gps_running:
        try:
            if serial_connection is None or not serial_connection.is_open:
                print(f"Attempting GPS connection on {SERIAL_PORT}...")
                serial_connection = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
                print(f"✅ Connected to GPS on {SERIAL_PORT} at {BAUDRATE} baud.")

            line = serial_connection.readline().decode('utf-8', errors='ignore').strip()
            if line:
                parts = line.split(',')
                with gps_lock:
                    if line.startswith('$GPGGA'):
                        gps_data.update(parse_GGA(parts))
                    elif line.startswith('$GPRMC'):
                        gps_data.update(parse_RMC(parts))
                    elif line.startswith('$GPVTG'):
                        gps_data.update(parse_VTG(parts))
                    elif line.startswith('$GPGSA'):
                        gps_data.update(parse_GSA(parts))

                # Check fix status and print once
                with gps_lock:
                    fix_status = gps_data.get('Fix', 'No')
                    num_satellites = int(gps_data.get('Satellites Used', 0))
                    if fix_status == 'Yes' and num_satellites >= 4:
                        if not fix_acquired:
                            print(f"✅ GPS Fix Acquired! Satellites: {num_satellites}")
                            fix_acquired = True
                    else:
                        if fix_acquired:
                            print("⚠️  GPS Fix Lost.")
                            fix_acquired = False

        except serial.SerialException as e:
            print(f"⚠️ GPS SerialException: {e}. Retrying in 5 seconds...")
            time.sleep(5)
            serial_connection = None  # Force reconnect

        except Exception as e:
            print(f"❌ Unexpected GPS reader error: {e}")
            time.sleep(1)



def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_height, frame_width = frame.shape[:2]
        center_of_screen = (frame_width // 2, frame_height // 2)

        # Run YOLO detection on 'person' class only (class 0)
        results = model.predict(source=frame, classes=[0], verbose=False)
        annotated_frame = frame.copy()

        # Draw green crosshairs at screen center
        cv2.line(annotated_frame, (center_of_screen[0], 0), (center_of_screen[0], frame_height), (0, 255, 0), 1)
        cv2.line(annotated_frame, (0, center_of_screen[1]), (frame_width, center_of_screen[1]), (0, 255, 0), 1)

        # Draw box and laser for ALL detected humans
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box[:4]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.line(annotated_frame, center_of_screen, (cx, cy), (0, 0, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/take_picture")
def take_picture():
    with frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=500, detail="No frame available")
        frame = latest_frame.copy()
    
    ret, buffer = cv2.imencode('.jpg', frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
    
    return JSONResponse(content={"image": frame_base64})

@app.get("/take_n_pictures")
def take_n_pictures(
    count: int = Query(10, ge=1, le=20),
    interval: float = Query(2.0, ge=0.1, le=10.0),
    annotated: bool = Query(False),
    quality: int = Query(75, ge=10, le=100)
):
    pictures = []
    for _ in range(count):
        with frame_lock:
            if latest_frame is None:
                raise HTTPException(status_code=500, detail="No frame available")
            frame = latest_frame.copy()
        
        encode_frame = frame
        if annotated:
            results = model.predict(source=frame, classes=[0], verbose=False)
            encode_frame = frame.copy()
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                center_of_screen = (frame.shape[1] // 2, frame.shape[0] // 2)
                for box in boxes:
                    x1, y1, x2, y2 = box[:4]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.rectangle(encode_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.line(encode_frame, center_of_screen, (cx, cy), (0, 0, 255), 2)
                    cv2.circle(encode_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        ret, buffer = cv2.imencode('.jpg', encode_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode image")
        frame_bytes = buffer.tobytes()
        frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
        pictures.append(frame_base64)
        
        time.sleep(interval)
    
    return JSONResponse(content={"pictures": pictures})

@app.get("/take_n_pictures_with_info")
def take_n_pictures_with_info(
    count: int = Query(10, ge=1, le=20),
    interval: float = Query(2.0, ge=0.1, le=10.0),
    include_image: bool = Query(True),
    annotated: bool = Query(True),
    quality: int = Query(75, ge=10, le=100)
):
    captures = []
    for _ in range(count):
        with frame_lock:
            if latest_frame is None:
                raise HTTPException(status_code=500, detail="No frame available")
            frame = latest_frame.copy()
        
        # Run YOLO detection only if needed (for annotations or detection data)
        detected_data = []
        encode_frame = frame
        if annotated or include_image or detected_data:  # Always run for detected_data
            results = model.predict(source=frame, classes=[0], verbose=False)
            if annotated:
                encode_frame = frame.copy()
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                center_of_screen = (frame.shape[1] // 2, frame.shape[0] // 2)
                for j, box in enumerate(boxes):
                    x1, y1, x2, y2 = box[:4]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    person_info = {
                        "position": {"x": cx, "y": cy},
                        "box_size": {"width": float(x2 - x1), "height": float(y2 - y1)},
                        "confidence": float(confidences[j])
                    }
                    detected_data.append(person_info)
                    
                    if annotated:
                        cv2.rectangle(encode_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.line(encode_frame, center_of_screen, (cx, cy), (0, 0, 255), 2)
                        cv2.circle(encode_frame, (cx, cy), 5, (0, 0, 255), -1)
        
        if include_image:
            ret, buffer = cv2.imencode('.jpg', encode_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ret:
                raise HTTPException(status_code=500, detail="Failed to encode image")
            frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        else:
            frame_base64 = None
        
        captures.append({
            "detected_data": detected_data,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "image": frame_base64 if include_image else None
        })
        
        time.sleep(interval)
    
    return JSONResponse(content={"captures": captures})

@app.get("/take_picture_with_info")
def take_picture_with_info(include_image: bool = Query(True)):
    with frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=500, detail="No frame available")
        frame = latest_frame.copy()
    
    results = model.predict(source=frame, classes=[0], verbose=False)
    annotated_frame = frame.copy()
    detected_data = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            person_info = {
                "position": {"x": cx, "y": cy},
                "box_size": {"width": float(x2 - x1), "height": float(y2 - y1)},
                "confidence": float(confidences[i])
            }
            detected_data.append(person_info)
    
    ret, buffer = cv2.imencode('.jpg', annotated_frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode image")
    frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
    
    response = {"detected_data": detected_data}
    if include_image:
        response["image"] = frame_base64
    
    return JSONResponse(content=response)

@app.get("/latest_detection")
def latest_detection():
    with frame_lock:
        if latest_frame is None:
            raise HTTPException(status_code=500, detail="No frame available")
        frame = latest_frame.copy()
    
    results = model.predict(source=frame, classes=[0], verbose=False)
    detected_data = []
    
    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box[:4]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            person_info = {
                "position": {"x": cx, "y": cy},
                "box_size": {"width": float(x2 - x1), "height": float(y2 - y1)},
                "confidence": float(confidences[i])
            }
            detected_data.append(person_info)
    
    return JSONResponse(content={"latest_detection": detected_data})


# Start GPS threads
gps_thread = threading.Thread(target=gps_reader, daemon=True)
gps_thread.start()


def sanitize_for_json(obj):
    """Replace inf, -inf, and NaN with None to make JSON-safe."""
    if isinstance(obj, float):
        if np.isinf(obj) or np.isnan(obj):
            return None
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj


@app.get("/gps_info")
def get_gps_info():
    with gps_lock:
        current_gps_data = gps_data.copy()

    if not current_gps_data:
        raise HTTPException(status_code=500, detail="No GPS data available")

    corrected = assess_corrected_values(current_gps_data)
    sanitized_data = sanitize_for_json({
        "gps_data": current_gps_data,
        "corrected_values": corrected
    })

    return JSONResponse(content=sanitized_data)



@app.get("/reset_camera")
def reset_camera():
    global camera, camera_running
    camera_running = False
    camera.release()
    
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise HTTPException(status_code=500, detail="Failed to reopen camera")
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera.set(cv2.CAP_PROP_FPS, 30)
    
    camera_running = True
    camera_thread = threading.Thread(target=camera_reader, daemon=True)
    camera_thread.start()
    
    return JSONResponse(content={"status": "Camera reset successful"})



@app.on_event("shutdown")
def shutdown_event():
    global camera_running
    camera_running = False
    camera.release()