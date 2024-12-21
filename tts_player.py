import time
import threading
import os
import subprocess
from gtts import gTTS

# Global variable to track the last audio playback time
last_audio_time = 0

def play_gtts_text(text, cooldown=10, speed=1.0):
    """
    Generate and play speech from text using Google TTS in a non-blocking manner.
    text: The text to be spoken.
    cooldown: Minimum number of seconds before another audio can be played.
    speed: Playback speed (1.0 is normal, >1.0 is faster, <1.0 is slower).
    """
    global last_audio_time
    current_time = time.time()

    # Respect cooldown to avoid frequent audio playback
    if current_time - last_audio_time < cooldown:
        return

    # Generate audio file
    original_audio = "detection_sound_original.mp3"
    tts = gTTS(text=text, lang="pt-br")
    tts.save(original_audio)

    # Modify playback speed using ffmpeg
    adjusted_audio = "detection_sound_adjusted.mp3"
    subprocess.run(
        [
            "ffmpeg", "-i", original_audio, "-filter:a", f"atempo={speed}",
            "-vn", adjusted_audio, "-y", "-loglevel", "quiet"
        ]
    )

    # Play audio in a background thread
    def play_audio_file(file_path):
        subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", file_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT
        ).wait()
        # Cleanup after playback
        os.remove(file_path)

    threading.Thread(target=play_audio_file, args=(adjusted_audio,), daemon=True).start()

    # Cleanup original audio file
    os.remove(original_audio)

    # Update the last audio time
    last_audio_time = current_time
