import streamlit as st
import tempfile
import os
import time
import requests
from streamlit_lottie import st_lottie

from qa_engine import answer_question
from voice_input import voice_to_text
from hf_utils import granite_summarize, granite_rephrase  # Granite stubs

st.set_page_config(page_title="📄 StudyMate AI", layout="wide", page_icon="📘")

# -------------------------------
# Helper functions
# -------------------------------
def load_lottieurl(url):
    try:
        r = requests.get(url)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Failed to load animation from {url}: {e}")
        return None

def typewriter(text, placeholder):
    output = ""
    for char in text:
        output += char
        placeholder.text(output)
        time.sleep(0.01)

def hover_card(title, content, key=None):
    """Display content in a card with hover effect and flash highlight."""
    st.markdown(f"""
        <style>
        .card-{key} {{
            background-color:#f0f0f0; 
            padding:20px; 
            border-radius:15px; 
            transition: transform 0.3s, box-shadow 0.3s, background-color 0.3s;
        }}
        .card-{key}:hover {{
            transform: scale(1.03);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }}
        .flash-{key} {{
            animation: flash 1s ease;
        }}
        @keyframes flash {{
            0% {{background-color: #ffff99;}}
            100% {{background-color: #f0f0f0;}}
        }}
        </style>
        <div class="card-{key} flash-{key}">
            <h4>{title}</h4>
            <p>{content}</p>
        </div>
    """, unsafe_allow_html=True)

# -------------------------------
# Load Lottie animations
# -------------------------------
background_lottie = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_tutvdkg0.json")
loading_animation = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_usmfx6bp.json")
success_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_HpFqiS.json")  # Working success animation

# -------------------------------
# Display Background & Header
# -------------------------------
if background_lottie:
    st_lottie(background_lottie, height=300, key="background_main", quality="low")

st.markdown(
    """
    <h1 style='text-align:center; 
               background: linear-gradient(to right, #4B0082, #8A2BE2); 
               -webkit-background-clip: text;
               -webkit-text-fill-color: transparent;
               font-size: 50px;'>
        📄 StudyMate AI
    </h1>
    <p style='text-align:center; font-size:18px;'>Upload PDFs and ask questions. Voice input supported!</p>
    """, unsafe_allow_html=True
)

# -------------------------------
# Tabs for Summarization & Q&A
# -------------------------------
tab1, tab2 = st.tabs(["Summarize Document", "Ask Questions"])

# -------------------------------
# Tab 1: Summarization
# -------------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload PDF to Summarize", type=["pdf"])
    if uploaded_file is not None and st.button("📄 Summarize Document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            if loading_animation:
                st_lottie(loading_animation, height=150, key="loading_summarize")
            summary_text = granite_summarize(tmp_path)
            st.success("✅ Document Summary:")
            placeholder = st.empty()
            typewriter(summary_text, placeholder)
            if success_animation:
                st_lottie(success_animation, height=150, key="success_summarize")
        except Exception as e:
            st.error(f"Error summarizing: {str(e)}")
        finally:
            os.remove(tmp_path)

# -------------------------------
# Tab 2: Question-Answering
# -------------------------------
with tab2:
    uploaded_file_qna = st.file_uploader("Upload PDF for Q&A", type=["pdf"], key="qna_uploader")

    mode = st.radio("Question input mode:", ["Text", "Voice"])
    question = ""

    # Custom CSS for text input box color
    def hover_card(title, content, key=None):
        st.markdown(f"""
            <div style="
                background-color:#2b2b2b; 
                color: #f1f1f1; 
                padding:20px; 
                border-radius:15px; 
                transition: transform 0.3s, box-shadow 0.3s;">
                <h4>{title}</h4>
                <p>{content}</p>
            </div>
        """, unsafe_allow_html=True)


    if mode == "Text":
        question = st.text_input("Ask a question about the document")
    else:
        if "voice_text" not in st.session_state:
            st.session_state.voice_text = ""
        if st.button("🎙️ Start Voice Recording (5 sec)"):
            try:
                with st.spinner("Listening..."):
                    st.session_state.voice_text = voice_to_text(duration=5)
                if st.session_state.voice_text.strip():
                    st.success(f"Voice converted to text: {st.session_state.voice_text}")
                else:
                    st.warning("No voice detected. Please try again.")
            except Exception as e:
                st.error(f"Voice recognition failed: {str(e)}")
                st.session_state.voice_text = ""
        question = st.session_state.voice_text

    if st.button("💡 Get Answer"):
        if uploaded_file_qna is None:
            st.warning("Please upload a document first.")
        elif not question.strip():
            st.warning("Enter a question or use voice input.")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file_qna.read())
                tmp_path = tmp.name
            try:
                with st.spinner("Fetching answer..."):
                    result = answer_question(tmp_path, question)
                
                original_answer = result["answers"][0]
                simplified_answer = granite_rephrase(original_answer)

                # Display side by side with hover and flash effects
                col1, col2 = st.columns(2)
                with col1:
                    hover_card("Original Answer", original_answer, key="orig")
                with col2:
                    hover_card("Granite Simplified Answer", simplified_answer, key="simp")

                # Trigger success animation
                if success_animation:
                    st_lottie(success_animation, height=150, key="success_qna")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
            finally:
                os.remove(tmp_path)
