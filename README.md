
# Agentic AI Podcast Insight Generator

This Streamlit app allows users to analyze podcast episodes for business consulting insights using AI.

## 🚀 Features
- Transcribe any podcast episode (via public URL)
- Summarize the transcript using Hugging Face models
- Extract business needs, goals, and stakeholders
- Customize analysis based on uploaded resume or LinkedIn profile
- Visualize stakeholder mentions

## 📂 Files
- `streamlit_app.py` – Main app file
- `requirements.txt` – All necessary dependencies

## 🔧 How to Deploy on Streamlit Cloud
1. Clone this repo to GitHub
2. Visit https://streamlit.io/cloud
3. Connect your GitHub and deploy this repo
4. Set `streamlit_app.py` as your entry point

## 📝 Notes
- Uses Whisper for transcription via `transformers`
- Uses `pydub` for MP3 audio loading (ffmpeg not needed if installed via Streamlit Cloud)
