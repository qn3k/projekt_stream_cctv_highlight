import cv2
import numpy as np
from ultralytics import YOLO
import argparse

class VideoMomentSearcher:
    def __init__(self, video_path, clip_len_sec=3, fps_sample=2):
        self.video_path = video_path
        self.clip_len_sec = clip_len_sec
        self.fps_sample = fps_sample
        self.model = YOLO("yolov8n.pt")
        self.index = []

    def iter_clips(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, int(fps / self.fps_sample))

        frames, times = [], []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % step == 0:
                frames.append(frame)
                times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)

            frame_id += 1

            if len(frames) >= self.clip_len_sec * self.fps_sample:
                yield frames, times
                frames, times = [], []

        cap.release()

    def detect_objects(self, frames):
        results = self.model(frames, verbose=False)
        objs = set()

        for r in results:
            for c in r.boxes.cls:
                objs.add(self.model.names[int(c)])

        return list(objs)

    def motion_score(self, frames):
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i - 1], frames[i])
            diffs.append(np.mean(diff))
        return float(np.mean(diffs)) if diffs else 0.0

    def interesting_score(self, objects, motion):
        score = 0

        if motion > 15:
            score += 1

        important = {"person", "car", "truck", "bicycle", "vase"}
        score += len(set(objects) & important)

        return score

    def build_index(self):
        print("[INFO] Buduję indeks wideo...")
        for frames, times in self.iter_clips():
            objects = self.detect_objects(frames)
            motion = self.motion_score(frames)
            score = self.interesting_score(objects, motion)

            self.index.append({
                "start": times[0],
                "end": times[-1],
                "objects": objects,
                "motion": motion,
                "score": score
            })

        print(f"[INFO] Gotowe. Fragmentów: {len(self.index)}")

    def search(self, query, min_score=1):
        keywords = query.lower().split()
        results = []

        for clip in self.index:
            text = " ".join(clip["objects"]).lower()
            if any(k in text for k in keywords) and clip["score"] >= min_score:
                results.append(clip)

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def show_results(self, results, limit=10):
        for r in results[:limit]:
            print(
                f"[{r['start']:.1f}s – {r['end']:.1f}s] "
                f"score={r['score']} objects={r['objects']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Wyszukiwarka momentów w wideo (MVP)")
    parser.add_argument("video", help="ścieżka do pliku wideo")
    parser.add_argument("--query", help="zapytanie (np. 'car person')", required=True)
    args = parser.parse_args()

    engine = VideoMomentSearcher(args.video)
    engine.build_index()

    results = engine.search(args.query)
    engine.show_results(results)


if __name__ == "__main__":
    main()
