import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import time
import json
import uuid
import threading
from auto_push import git_push
from datetime import datetime

# -------- GPS SIMU ----------
with open("gps_simu.json", "r") as f:
    GPS_POINTS = json.load(f)

gps_index = 0

def git_push_async():
    t = threading.Thread(target=git_push, daemon=True)
    t.start()


def get_gps_mock():
    global gps_index
    point = GPS_POINTS[gps_index]
    gps_index = (gps_index + 1) % len(GPS_POINTS)
    return point["lat"], point["lon"]


def save_pothole(frame, lat, lon, conf):
    pid = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    img_path = f"images/{pid}.jpg"
    cv2.imwrite(img_path, frame)   # ← LƯU ẢNH GỐC

    record = {
        "id": pid,
        "lat": lat,
        "lon": lon,
        "confidence": float(conf),
        "image": img_path,
        "time": timestamp
    }

    with open("potholes_data.json", "a") as f:
        f.write(json.dumps(record) + "\n")



TRT_LOGGER = trt.Logger(trt.Logger.INFO)
ENGINE_PATH = "yolov5n.engine"  
CONF_THR = 0.25                     # confidence threshold
IOU_THR = 0.45                      # NMS IoU threshold

# ---------- Load TensorRT engine ----------
def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()
print("Engine loaded")

# ---------- Prepare buffers ----------
inputs, outputs, bindings = [], [], []

for i, binding in enumerate(engine):
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    dtype = trt.nptype(engine.get_binding_dtype(binding))
    host_mem = cuda.pagelocked_empty(size, dtype)
    device_mem = cuda.mem_alloc(host_mem.nbytes)
    bindings.append(int(device_mem))
    if engine.binding_is_input(binding):
        inputs.append({'host': host_mem, 'device': device_mem})
        input_shape = engine.get_binding_shape(binding)
    else:
        outputs.append({'host': host_mem, 'device': device_mem})
        output_shape = engine.get_binding_shape(binding)

input_h, input_w = input_shape[2], input_shape[3]
stream = cuda.Stream()
print(f"Input: {input_shape}, Output: {output_shape}")

# ---------- Preprocess ----------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_w, input_h))
    img = img.astype(np.float32)/255.0
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, axis=0)
    return np.ascontiguousarray(img)

# ---------- Map engine output to frame ----------
def map_box_to_frame(box, frame_w, frame_h):
    x_c, y_c, w, h = box[0:4]
    conf = box[4]

    scale_x = frame_w / input_w
    scale_y = frame_h / input_h

    x1 = int((x_c - w/2) * scale_x)
    y1 = int((y_c - h/2) * scale_y)
    x2 = int((x_c + w/2) * scale_x)
    y2 = int((y_c + h/2) * scale_y)

    # clamp
    x1 = max(0, min(frame_w-1, x1))
    y1 = max(0, min(frame_h-1, y1))
    x2 = max(0, min(frame_w-1, x2))
    y2 = max(0, min(frame_h-1, y2))

    return x1, y1, x2, y2, conf

# ---------- Non-Maximum Suppression ----------
def nms_boxes_safe(boxes, iou_threshold=0.45):
    if boxes is None or len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# ---------- Camera ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 448)

# tao cua so
cv2.namedWindow("YOLOv5 TensorRT Pothole", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv5 TensorRT Pothole", 800, 448)

prev_time = time.time()

last_save_time = 0
SAVE_INTERVAL = 2.0

last_push_time = 0
PUSH_INTERVAL = 60   # giây (1 phút)


while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue

    H, W = frame.shape[:2]

    try:
        img = preprocess(frame)
        np.copyto(inputs[0]['host'], img.ravel())

        # inference
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
        stream.synchronize()

        # reshape output
        output = np.array(outputs[0]['host']).reshape(-1,6)
        boxes = output[output[:,4] > CONF_THR]

        # map boxes
        mapped_boxes = []
        for b in boxes:
            x1, y1, x2, y2, conf = map_box_to_frame(b, W, H)
            mapped_boxes.append([x1, y1, x2, y2, conf])

        mapped_boxes = np.array(mapped_boxes, dtype=np.float32)
        keep = nms_boxes_safe(mapped_boxes, iou_threshold=IOU_THR)
        final_boxes = mapped_boxes[keep] if len(keep) > 0 else []
        
        # ================= GPS MOCK + SAVE =================
        now = time.time()
        if len(final_boxes) > 0 and now - last_save_time > SAVE_INTERVAL:
            lat, lon = get_gps_mock()

            # lấy box có confidence cao nhất
            best_box = final_boxes[np.argmax(final_boxes[:,4])]
            conf = best_box[4]

            save_pothole(frame, lat, lon, conf)
            last_save_time = now
        if now - last_push_time > PUSH_INTERVAL:
            git_push_async()
            last_push_time = now    
        # draw
        draw_frame = frame.copy()
        for b in final_boxes:
            x1, y1, x2, y2, conf = b
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(draw_frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(draw_frame,f"Pothole:{conf:.2f}",(x1,max(15,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # FPS
        curr_time = time.time()
        fps = 1/(curr_time-prev_time + 1e-6)
        prev_time = curr_time
        cv2.putText(draw_frame, f"FPS:{fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        cv2.imshow("YOLOv5 TensorRT Pothole", draw_frame)

    except Exception as e:
        print("Warning: Exception during frame processing:", e)
        continue
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Nhan Q de thoat")
        break
cap.release()
cv2.destroyAllWindows()

