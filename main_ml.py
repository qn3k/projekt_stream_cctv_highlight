import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import argparse
import os

class VideoMomentSearcherML:
    def __init__(self, video_path, clip_len_sec=3, fps_sample=2):
        self.video_path = video_path
        self.clip_len_sec = clip_len_sec
        self.fps_sample = fps_sample
        self.model = YOLO("yolov8n.pt")
        self.clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

    # -------- feature extraction --------

    def iter_clips(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        step = max(1, int(fps / self.fps_sample))

        frames = []
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_id % step == 0:
                frames.append(frame)

            frame_id += 1

            if len(frames) >= self.clip_len_sec * self.fps_sample:
                yield frames
                frames = []

        cap.release()

    def motion_score(self, frames):
        diffs = []
        for i in range(1, len(frames)):
            diffs.append(np.mean(cv2.absdiff(frames[i-1], frames[i])))
        return np.mean(diffs) if diffs else 0.0

    def object_features(self, frames):
        results = self.model(frames, verbose=False)

        counts = {"person": 0, "car": 0}
        total = 0

        for r in results:
            for c in r.boxes.cls:
                name = self.model.names[int(c)]
                total += 1
                if name in counts:
                    counts[name] += 1

        return total, counts["person"], counts["car"]

    def extract_features(self, frames):
        motion = self.motion_score(frames)
        total, persons, cars = self.object_features(frames)

        return [
            motion,
            total,
            persons,
            cars
        ]

    # -------- training --------

    def build_dataset(self, labels):
        X, y = [], []
        for frames, label in zip(self.iter_clips(), labels):
            X.append(self.extract_features(frames))
            y.append(label)
        return np.array(X), np.array(y)

    def train(self, X, y):
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.clf.fit(Xtr, ytr)
        preds = self.clf.predict(Xte)

        print(classification_report(yte, preds))

    def save(self, path="model.joblib"):
        joblib.dump(self.clf, path)

    def load(self, path="model.joblib"):
        self.clf = joblib.load(path)

    # -------- inference --------

    def search(self, threshold=0.6):
        results = []
        t = 0

        for frames in self.iter_clips():
            feats = self.extract_features(frames)
            prob = self.clf.predict_proba([feats])[0][1]

            if prob >= threshold:
                results.append((t, t + self.clip_len_sec, prob))

            t += self.clip_len_sec

        return results


# -------- CLI --------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--labels", help="plik .npy z etykietami")
    args = parser.parse_args()

    engine = VideoMomentSearcherML(
    args.video,
    clip_len_sec=5,
    fps_sample=2    
    )


    if args.train:
        labels = np.load(args.labels)
        X, y = engine.build_dataset(labels)
        engine.train(X, y)
        engine.save()
    else:
        engine.load()
        hits = engine.search()
        for s, e, p in hits:
            print(f"[{s:.1f}-{e:.1f}s] P(ciekawy)={p:.2f}")

if __name__ == "__main__":
    main()

# 100 fragmentow
#recznie oznaczamy fragmenty ciekawe
#
#labels = [0,0,1]
#np.save("labels.npy", labels)
#trening
#python main_ml.py video.mp4 --train --labels labels.npy
#wyszkukiwanie
#python main_search_ml.py video.mp4

