import streamlit as st
import os
from main import VideoMomentSearcher

st.set_page_config(page_title="Highlight Finder")

st.title("Highlight Finder")

url = st.text_input("Link do wideo (YouTube / Twitch / plik lokalny)")
clip_len = st.slider("Długość fragmentu (sekundy)", 3, 30, 5)
use_audio = st.checkbox("Audio", value=True)
use_yolo = st.checkbox("YOLO", value=True)

if st.button("Start"):
    with st.spinner("Pobieram wideo..."):
        if url.startswith("http"):
            os.system(f"yt-dlp -f mp4 -o input.mp4 {url}")
            video_path = "input.mp4"
        else:
            video_path = url

    engine = VideoMomentSearcher(
        video_path,
        clip_len_sec=clip_len
    )

    st.info("Analizuję wideo...")
    engine.build_index()

    st.success("Gotowe!")

    st.write("Najlepsze fragmenty:")
    for clip in sorted(engine.index, key=lambda x: x["score"], reverse=True)[:5]:
        st.write(
            f"{clip['start']:.1f}s – {clip['end']:.1f}s | score={clip['score']}"
        )
