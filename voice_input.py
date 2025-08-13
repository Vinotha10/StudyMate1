import sounddevice as sd
import vosk
import queue
import json

# -------------------------------
# Load Vosk model once (download small English model first)
# e.g., https://alphacephei.com/vosk/models
# -------------------------------
model = vosk.Model("vosk-model-small-en-us-0.15")  # Ensure folder exists

def voice_to_text(duration=5, samplerate=16000):
    """Record voice for `duration` seconds and convert to text using Vosk"""
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status)
        q.put(bytes(indata))

    with sd.RawInputStream(samplerate=samplerate, blocksize=8000, dtype='int16',
                           channels=1, callback=callback):
        rec = vosk.KaldiRecognizer(model, samplerate)
        print("Recording voice...")
        for _ in range(int(samplerate / 8000 * duration)):
            data = q.get()
            if rec.AcceptWaveform(data):
                pass
        final_result = rec.FinalResult()
        text = json.loads(final_result).get("text", "")
        return text
