from ultralytics import YOLO
import cv2
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

CONFIDENCE = 0.2
# Load a pre-trained YOLOv10n model
model = YOLO("yolov10m.pt")

object_tracker = DeepSort(
    max_age=20,
    n_init=2,
    nms_max_overlap=0.3,
    max_cosine_distance=0.8,
    nn_budget=None,
    override_track_class=None,
    embedder="mobilenet",
    half=True,
    bgr=True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon=False,
    today=None
)

classList = ["person"]


def make_detections(boxes):
    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        clsNum = int(box.cls[0])

        if result.names[clsNum] not in classList:
            continue
        conf = box.conf[0]
        # print(box)
        detections.append((([x1, y1, w, h]), clsNum, conf))
        # detections.append(((int(box.xyxy[0][0]), int(box.xyxy[0][1])),))
        # cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
        #               (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
        # cv2.putText(frame, f"{result.names[int(box.cls[0])]}",
        #             (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
        #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    return detections


# Open the video file
cap = cv2.VideoCapture("football.mp4")

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# for _ in range(500):
#     ret, frame = cap.read()
#     if not ret:
#         break

# Read frames one by one
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # height, width, _ = frame.shape

    # # Calculate the new dimensions (half the original size)
    # new_height, new_width = height // 2, width // 2

    # # Resize the image
    # frame = cv2.resize(frame, (new_width, new_height))

    startTime = time.perf_counter()
    results = model.predict(frame, conf=CONFIDENCE)
    result = results[0]
    detections = make_detections(result.boxes)
    tracks = object_tracker.update_tracks(detections, frame=frame)
    # print(result.boxes)

    for index, track in enumerate(tracks):
        if not track.is_confirmed():
            continue
        trackId = track.track_id
        ltrb = track.to_ltrb()
        bbox = ltrb
        # print(
        #     f"{index} and {track} and {track.is_confirmed()}")
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(
            bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(frame, f"{str(trackId)}", (int(bbox[0]), int(
            bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)
    # for box in result.boxes:
    #     # print(box)
    #     cv2.rectangle(frame, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
    #                   (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
    #     cv2.putText(frame, f"{result.names[int(box.cls[0])]}",
    #                 (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
    #                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    # Process the frame here
    cv2.imshow("Frame", frame)
    endTime = time.perf_counter()
    fps = 1 / (endTime - startTime)
    print(fps)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
