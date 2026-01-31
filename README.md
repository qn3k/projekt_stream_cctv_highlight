# projekt_stream_cctv_highlight

1. Jak używać?

python main.py video.mp4 --query car

python main.py test.mp4 --query person

lub z pomocą aplikacji Streamlit

streamlit run app.py

2. Co robi?

Wybiera najlepsze kawałki video z pomocą YOLO (obiekty oraz ruch) oraz librosa (audio), zapisuje je do pliku.
Działa w wersji zarówno konsolowej oraz aplikacji webowej. Pokazuje nam znormalizowany score naszego fragmentu.
Jeśli chcemy może zapisać nasze fragmenty.

3. Co potrzeba?

Do działania - obiekty i ruch:
pip install ultralytics opencv-python numpy librosa

Do audio:
pip install librosa soundfile

Do zapisu fragmentu:
pip install ffmpeg

Do zapisu napisów:
pip install openai-whisper

Do zapisu video z linku:
pip install streamlit yt-dlp
