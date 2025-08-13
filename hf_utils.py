import requests
from PyPDF2 import PdfReader
from transformers import pipeline

# -------------------------------
# HuggingFace pipelines (CPU)
# -------------------------------
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
simplifier = pipeline("text2text-generation", model="t5-small", device=-1)

# -------------------------------
# Internal functions
# -------------------------------
def hf_summarize(text, max_tokens=1024):
    """Summarize text in chunks to prevent index errors"""
    if not text.strip():
        return "No text to summarize."

    words = text.split()
    outputs = []
    chunk = []
    count = 0

    for word in words:
        chunk.append(word)
        count += 1
        if count >= max_tokens:
            chunk_text = " ".join(chunk)
            summary = summarizer(chunk_text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
            outputs.append(summary)
            chunk = []
            count = 0

    if chunk:
        chunk_text = " ".join(chunk)
        summary = summarizer(chunk_text, max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
        outputs.append(summary)

    return " ".join(outputs)

def hf_simplify(text):
    """Simplify/rephrase text safely"""
    if not text.strip():
        return text
    result = simplifier("simplify: " + text, max_length=200, do_sample=False)
    return result[0]["generated_text"]

# -------------------------------
# Granite stub functions
# -------------------------------
def granite_summarize(pdf_path):
    """Read PDF and summarize using HuggingFace as Granite stub"""
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            full_text += page_text + "\n"

    if not full_text.strip():
        return "PDF is empty or text could not be extracted."

    return hf_summarize(full_text)

def granite_rephrase(text):
    """Rephrase/simplify answer as Granite stub"""
    return hf_simplify(text)
