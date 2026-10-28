"""
Generates the User's Exact 10-Note Humming WAV File:
- Note 1 (4m) + Note 2 (2m)
- Breath pause (300ms)
- Note 3 (5m) + Note 4 (2m) + Note 5 (1m)
- Longer gap (800ms) -> Line break
- Note 6 (3m) + Note 7 (2m) + Note 8 (1m)
- Breath pause (320ms)
- Note 9 (2m) + Note 10 (2m)
"""

import math
import wave
import numpy as np


def generate_note(freq_hz, duration_s, sample_rate=44100):
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
    # Voice envelope: fast 8ms attack and 8ms decay for full acoustic fidelity
    fade_len = min(int(0.008 * sample_rate), num_samples // 4)
    envelope = np.ones(num_samples)
    if fade_len > 0:
        envelope[:fade_len] = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
        envelope[-fade_len:] = np.sin(np.linspace(np.pi / 2, 0, fade_len)) ** 2

    waveform = (
        0.60 * np.sin(2 * np.pi * freq_hz * t) +
        0.25 * np.sin(2 * np.pi * (freq_hz * 2) * t) +
        0.15 * np.sin(2 * np.pi * (freq_hz * 3) * t)
    ) * envelope
    return waveform


def generate_silence(duration_s, sample_rate=44100):
    num_samples = int(duration_s * sample_rate)
    return np.random.normal(0, 0.001, num_samples)


def make_user_10note_audio(filename="sample_user_10notes_breaths.wav"):
    sr = 44100
    bpm = 83.0
    t_laghu = 60.0 / (bpm * 2.8)  # ~0.2582 s

    # Line 1, Word 1: Note 1 (4m), Note 2 (2m)
    n1 = generate_note(220.00, t_laghu * 4, sr)  # A3 (4m)
    n2 = generate_note(146.83, t_laghu * 2, sr)  # D3 (2m)
    w1 = np.concatenate([n1, n2])

    # Breath Pause 1 (300ms)
    b1 = generate_silence(0.30, sr)

    # Line 1, Word 2: Note 3 (5m), Note 4 (2m), Note 5 (1m)
    n3 = generate_note(174.61, t_laghu * 5, sr)  # F3 (5m)
    n4 = generate_note(196.00, t_laghu * 2, sr)  # G3 (2m)
    n5 = generate_note(220.00, t_laghu * 1, sr)  # A3 (1m)
    w2 = np.concatenate([n3, n4, n5])

    # Longer Gap (800ms) -> Line break!
    longer_gap = generate_silence(0.80, sr)

    # Line 2, Word 3: Note 6 (3m), Note 7 (2m), Note 8 (1m)
    n6 = generate_note(196.00, t_laghu * 3, sr)  # G3 (3m)
    n7 = generate_note(164.81, t_laghu * 2, sr)  # E3 (2m)
    n8 = generate_note(146.83, t_laghu * 1, sr)  # D3 (1m)
    w3 = np.concatenate([n6, n7, n8])

    # Breath Pause 2 (320ms)
    b2 = generate_silence(0.32, sr)

    # Line 2, Word 4: Note 9 (2m), Note 10 (2m)
    n9 = generate_note(130.81, t_laghu * 2, sr)  # C3 (2m)
    n10 = generate_note(146.83, t_laghu * 2, sr) # D3 (2m)
    w4 = np.concatenate([n9, n10])

    full_audio = np.concatenate([w1, b1, w2, longer_gap, w3, b2, w4])

    int_samples = (full_audio * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())

    total_dur = len(full_audio) / sr
    print(f"Generated User 10-Note Audio: {filename} ({total_dur:.2f}s)")
    print(f"  • Line 1: Word 1 [4m, 2m] -> Breath (0.30s) -> Word 2 [5m, 2m, 1m]")
    print(f"  • Longer Gap (0.80s)")
    print(f"  • Line 2: Word 3 [3m, 2m, 1m] -> Breath (0.32s) -> Word 4 [2m, 2m]")


if __name__ == '__main__':
    make_user_10note_audio()
