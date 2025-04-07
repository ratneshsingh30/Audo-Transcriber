
# Streamlit App: Podcast Insight Generator using OpenAI Whisper API
import streamlit as st
import os
import openai
import tempfile
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from docx import Document
from collections import Counter
import requests
from bs4 import BeautifulSoup

# Set your API key from environment variable or input
openai.api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else st.text_input("Enter your OpenAI API Key", type="password")

# Helper Functions
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

def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
        return transcript["text"]

def summarize_transcript(text):
    summarizer = pipeline("summarization")
    return summarizer(text[:1000], max_length=130, min_length=30, do_sample=False)[0]['summary_text']

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

# Streamlit UI
st.set_page_config(page_title="OpenAI Whisper AI Consulting Demo")
st.title("🎧 Podcast Insight Generator (via OpenAI Whisper API)")

audio_url = st.text_input("Paste podcast audio URL (MP3)", "")
resume_file = st.file_uploader("Upload your Resume (.docx)", type="docx")
linkedin_url = st.text_input("Or paste your LinkedIn URL")

if st.button("Generate Report"):
    if not audio_url:
        st.error("Please enter a podcast MP3 URL.")
        st.stop()

    with st.spinner("Downloading and transcribing audio..."):
        response = requests.get(audio_url)
        if response.status_code != 200:
            st.error("Failed to fetch audio file.")
            st.stop()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name

        try:
            transcript = transcribe_audio(tmp_path)
        except Exception as e:
            st.error(f"OpenAI Whisper API failed: {str(e)}")
            st.stop()

    with st.spinner("Summarizing and generating insights..."):
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
