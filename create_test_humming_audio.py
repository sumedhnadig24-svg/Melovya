"""
Creates a sample synthetic humming WAV audio file with multiple lines
separated by natural breath silences (> 500ms) to test long-term silence detection.
"""

import math
import wave
import numpy as np


def generate_humming_phrase(note_pitches_durations, sample_rate=44100):
    audio = []
    for freq, dur_s in note_pitches_durations:
        num_samples = int(dur_s * sample_rate)
        t = np.linspace(0, dur_s, num_samples, endpoint=False)
        # Humming voice harmonics (fundamental + soft 2nd/3rd harmonics)
        # Envelope to avoid click onsets/offsets
        envelope = np.sin(np.pi * np.linspace(0, 1, num_samples)) ** 0.5
        waveform = (
            0.60 * np.sin(2 * np.pi * freq * t) +
            0.25 * np.sin(2 * np.pi * (freq * 2) * t) +
            0.15 * np.sin(2 * np.pi * (freq * 3) * t)
        ) * envelope
        audio.append(waveform)
    return np.concatenate(audio)


def generate_silence(duration_s, sample_rate=44100):
    num_samples = int(duration_s * sample_rate)
    # Very faint background room noise (realistic)
    return np.random.normal(0, 0.002, num_samples)


def make_test_wav(filename="sample_humming_4lines.wav"):
    sr = 44100
    bpm = 83.0
    t_laghu = 60.0 / (bpm * 2.0)  # ~0.361 s

    # 10-note phrase (14 matras)
    # [2, 1, 2, 1, 1, 2, 1, 2, 1, 1]
    phrase1_notes = [
        (130.81, t_laghu * 2),  # C3 Guru (2)
        (146.83, t_laghu * 1),  # D3 Laghu (1)
        (164.81, t_laghu * 2),  # E3 Guru (2)
        (174.61, t_laghu * 1),  # F3 Laghu (1)
        (196.00, t_laghu * 1),  # G3 Laghu (1)
        (220.00, t_laghu * 2),  # A3 Guru (2)
        (196.00, t_laghu * 1),  # G3 Laghu (1)
        (174.61, t_laghu * 2),  # F3 Guru (2)
        (164.81, t_laghu * 1),  # E3 Laghu (1)
        (130.81, t_laghu * 1)   # C3 Laghu (1)
    ]

    p1 = generate_humming_phrase(phrase1_notes, sr)
    silence1 = generate_silence(0.70, sr)  # 700ms silence (Line 1 ends)

    p2 = generate_humming_phrase(phrase1_notes, sr)
    silence2 = generate_silence(0.85, sr)  # 850ms silence (Line 2 ends)

    p3 = generate_humming_phrase(phrase1_notes, sr)
    silence3 = generate_silence(0.75, sr)  # 750ms silence (Line 3 ends)

    p4 = generate_humming_phrase(phrase1_notes, sr)

    full_song = np.concatenate([p1, silence1, p2, silence2, p3, silence3, p4])

    # Convert to 16-bit PCM
    int_samples = (full_song * 32767).astype(np.int16)

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())

    print(f"Generated test humming audio: {filename} (Duration: {len(full_song)/sr:.2f}s, 4 lines separated by silence)")


if __name__ == '__main__':
    make_test_wav()
