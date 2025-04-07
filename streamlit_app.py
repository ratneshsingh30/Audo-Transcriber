
# Streamlit Web App for Agentic AI Consulting Demo (no ffmpeg dependency)
import streamlit as st
import os
import subprocess
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
import torch
import requests
from bs4 import BeautifulSoup
from docx import Document
from collections import Counter
from pydub import AudioSegment
import numpy as np
import shutil

# --- Helper Functions ---
def extract_profile_from_resume(file):
    doc = Document(file)
    text = " ".join([para.text for para in doc.paragraphs])
    background = "Business Analytics professional" if "analytics" in text.lower() else "Tech Consultant"
    interests = [kw for kw in ["AI", "data", "remote", "decision", "collaboration"] if kw.lower() in text.lower()]
    return {
        "name": "Resume User",
        "background": background,
        "interests": interests or ["AI", "Analytics"]
    }

def extract_profile_from_linkedin(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        text = soup.get_text().lower()
        interests = [kw.capitalize() for kw in ["analytics", "consulting", "ai", "data"] if kw in text]
        return {
            "name": "LinkedIn User",
            "background": "Business Consultant",
            "interests": interests or ["AI"]
        }
    except:
        return {"name": "User", "background": "Professional", "interests": ["AI"]}

def summarize_transcript(transcript):
    summarizer = pipeline("summarization")
    return summarizer(transcript[:1000], max_length=130, min_length=30, do_sample=False)[0]['summary_text']

def generate_insights(summary, profile):
    return {
        "business_needs": ["Faster analytics", "Cost optimization", "KPI alignment", "User engagement"],
        "stakeholder_focus": ["Marketing", "Finance", "Product"],
        "goals": {"user_engagement_target": "25%", "timeline": "2 quarters"},
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "personal_reflection": f"As a {profile['background']}, this aligns with my interests in {', '.join(profile['interests'])}."
    }

def visualize_stakeholders(data):
    df = pd.DataFrame(Counter(data).items(), columns=["Department", "Mentions"])
    fig, ax = plt.subplots()
    ax.bar(df['Department'], df['Mentions'], color='skyblue')
    ax.set_title("Stakeholder Mentions")
    return fig

def download_audio(url, output_file="podcast_audio.webm"):
    try:
        if shutil.which("yt-dlp") is None:
            st.error("yt-dlp is not installed. Please install it with: pip install yt-dlp")
            return False

        result = subprocess.run(
            ["yt-dlp", "-f", "bestaudio", "-o", output_file, url],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            st.error(f"Failed to download audio: {result.stderr}")
            return False

        if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
            st.error("Downloaded file is empty or does not exist.")
            return False

        return True
    except Exception as e:
        st.error(f"Error downloading audio: {str(e)}")
        return False

# --- Streamlit UI ---
st.set_page_config(page_title="Agentic AI Consulting Demo", layout="centered")
st.title("🎧 AI-Powered Podcast Insight Generator")

podcast_url = st.text_input("Enter Podcast URL", "https://shows.acast.com/design-talk/episodes/software-architecture")
resume_file = st.file_uploader("Upload your Resume (.docx)", type="docx")
linkedin_url = st.text_input("Or paste your LinkedIn URL")

if st.button("Generate Report"):
    audio_file = "podcast_audio.webm"
    if os.path.exists(audio_file):
        os.remove(audio_file)

    with st.spinner("Downloading audio..."):
        download_success = download_audio(podcast_url, audio_file)
        if not download_success:
            st.stop()

    with st.spinner("Transcribing audio..."):
        try:
            audio = AudioSegment.from_file(audio_file)
            audio = audio.set_channels(1).set_frame_rate(16000)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
            waveform = torch.tensor(samples).unsqueeze(0)

            processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            input_features = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_features
            predicted_ids = model.generate(input_features)
            transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except Exception as e:
            st.error(f"Error during transcription: {str(e)}")
            st.stop()

    with st.spinner("Summarizing and generating insights..."):
        try:
            summary = summarize_transcript(transcript)
            profile = extract_profile_from_resume(resume_file) if resume_file else extract_profile_from_linkedin(linkedin_url)
            insights = generate_insights(summary, profile)

            st.subheader("🔍 Summary")
            st.write(insights['summary'])

            st.subheader("📌 Business Needs")
            st.write("\n".join([f"- {x}" for x in insights['business_needs']]))

            st.subheader("👥 Stakeholder Focus")
            st.pyplot(visualize_stakeholders(insights['stakeholder_focus']))

            st.subheader("🎯 Goals")
            st.write(insights['goals'])

            st.subheader("💬 Personal Reflection")
            st.write(insights['personal_reflection'])

            st.success("✅ Insight report generated!")
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
