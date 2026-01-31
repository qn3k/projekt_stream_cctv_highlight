import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import librosa
import os
import subprocess
import whisper

class VideoMomentSearcher:
    def __init__(self, video_path, clip_len_sec=30, fps_sample=2):
        self.video_path = video_path
        self.clip_len_sec = clip_len_sec
        self.fps_sample = fps_sample
        self.model = YOLO("yolov8n.pt")
        self.index = []
        self.asr = None

    def audio_score(self, start, duration):
        try:
            y, sr = librosa.load(
                self.video_path,
                offset=start,
                duration=duration,
                mono=True
            )
            rms = np.mean(librosa.feature.rms(y=y))
            return rms
        except Exception:
            return 0.0

    def adaptive_threshold(self, percentile=90):
        if not self.index:
            return 0.0
        scores = [c["score"] for c in self.index]
        return np.percentile(scores, percentile)


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
        objs = []
        results = self.model(frames, verbose=False)

        for r in results:
            for c in r.boxes.cls:
                objs.append(self.model.names[int(c)])
        return objs

    def motion_score(self, frames):
        diffs = []
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i - 1], frames[i])
            diffs.append(np.mean(diff))
        return float(np.mean(diffs)) if diffs else 0.0
    
    def interesting_score(self, objects, motion, audio):
        score = 0

        # gracze
        score += objects.count("person") * 2

        # ruch
        if motion > 20:
            score += 2

        # dźwięk
        if audio > 0.1:
            score += 3

        #important = {"person", "fire", "gun", "smoke"}
        #score += len(set(objects) & important)

        return score



    def build_index(self):
        print("[INFO] Buduję indeks wideo...")
        for frames, times in self.iter_clips():
            start = times[0]
            duration = self.clip_len_sec

            objects = self.detect_objects(frames)
            motion = self.motion_score(frames)
            audio = self.audio_score(start, duration)
            #audio_thr = np.percentile([c["audio"] for c in self.index], 90)

            score = self.interesting_score(objects, motion, audio)
            #score = self.interesting_score(objects, motion)

            self.index.append({
                "start": start,
                "end": start + duration,
                "objects": objects,
                "motion": motion,
                "audio": audio,
                "raw_score": score
            })

        self.normalize_scores()
        print(f"[INFO] Gotowe. Fragmentów: {len(self.index)}")

    def normalize_scores(self):
        raw_scores = [c["raw_score"] for c in self.index]
        max_s = max(raw_scores) if raw_scores else 1

        for clip in self.index:
            clip["score"] = clip["raw_score"] / max_s

    def search(self, query, percentile=90):
        keywords = query.lower().split()
        threshold = self.adaptive_threshold(percentile)

        results = []
        for clip in self.index:
            text = " ".join(clip["objects"]).lower()
            if any(k in text for k in keywords) and clip["score"] >= threshold:
                results.append(clip)

        return sorted(results, key=lambda x: x["score"], reverse=True)


    def show_results(self, results, limit=10):
        for r in results[:limit]:
            print(
                f"[{r['start']:.1f}s – {r['end']:.1f}s] "
                f"score={r['score']} objects={r['objects']}"
            )

    def save_top_clips(self, percentile=90, out_dir="highlights"):
        os.makedirs(out_dir, exist_ok=True)

        threshold = self.adaptive_threshold(percentile)
        selected = [c for c in self.index if c["score"] >= threshold]

        for i, clip in enumerate(selected):
            out = os.path.join(out_dir, f"highlight_{i+1}.mp4")
            self.save_clip(clip["start"], self.clip_len_sec, out)
            print(f"[SAVED] {out} score={clip['score']:.2f}")

    #zapisywanie do mp4
    def save_clip(self, start, duration, out_path):
        cmd = [
            "./ffmpeg",
            "-y",
            "-ss", str(start),
            "-i", self.video_path,
            "-t", str(duration),
            "-c", "copy",
            out_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    #napisy
    def transcribe_clip(self, clip_path):
        if self.asr is None:
            self.asr = whisper.load_model("base")
        result = self.asr.transcribe(clip_path, fp16=False)
        return result["segments"]

    def save_subtitles_txt(self, text, out_path):
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    #fajny format dla davinci
    def save_subtitles_srt(self, segments, out_path):
        def ts(t):
            h = int(t // 3600)
            m = int((t % 3600) // 60)
            s = t % 60
            return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")

        with open(out_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments, 1):
                f.write(f"{i}\n")
                f.write(f"{ts(seg['start'])} --> {ts(seg['end'])}\n")
                f.write(f"{seg['text'].strip()}\n\n")

    def save_top_clips_with_subs(self, percentile=90, out_dir="highlights"):
        os.makedirs(out_dir, exist_ok=True)

        threshold = self.adaptive_threshold(percentile)
        selected = [c for c in self.index if c["score"] >= threshold]

        for i, clip in enumerate(selected):
            video_out = os.path.join(out_dir, f"highlight_{i+1}.mp4")
            subs_out = os.path.join(out_dir, f"highlight_{i+1}.txt")

            self.save_clip(clip["start"], self.clip_len_sec, video_out)

            text = self.transcribe_clip(video_out)
            self.save_subtitles_srt(text, subs_out)
            #self.save_subtitles_txt(text, subs_out)

            print(f"[SAVED] {video_out} + napisy")



def main():
    parser = argparse.ArgumentParser(description="Wyszukiwarka momentów w wideo (MVP)")
    parser.add_argument("video", help="ścieżka do pliku wideo")
    parser.add_argument("--query", help="zapytanie (np. 'car person')",default="person")
    parser.add_argument("--save", action="store_true", help="zapisz highlighty")
    parser.add_argument("--subs", action="store_true", help="zapisz highlighty z napisami")

    args = parser.parse_args()

    engine = VideoMomentSearcher(args.video)
    engine.build_index()

    if args.subs:
        engine.save_top_clips_with_subs()
    elif args.save:
        engine.save_top_clips()

    results = engine.search(args.query)
    engine.show_results(results)
    #engine.save_top_clips(top_n=3)


if __name__ == "__main__":
    main()
