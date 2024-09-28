
import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

MODEL_PATH = "models/yolov10m.pt"
VIDEO_PATH = "assets/football.mp4"


def main():
  detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
  tracker = Tracker()

  cap = cv2.VideoCapture(VIDEO_PATH)

  if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    start_time = time.perf_counter()
    detections = detector.detect(frame)
    tracking_ids, boxes = tracker.track(detections, frame)

    for tracking_id, bounding_box in zip(tracking_ids, boxes):
      cv2.rectangle(frame, (int(bounding_box[0]), int(bounding_box[1])), (int(
          bounding_box[2]), int(bounding_box[3])), (0, 0, 255), 2)
      cv2.putText(frame, f"{str(tracking_id)}", (int(bounding_box[0]), int(
          bounding_box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    end_time = time.perf_counter()
    fps = 1 / (end_time - start_time)
    print(f"Current fps: {fps}")

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
      break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
