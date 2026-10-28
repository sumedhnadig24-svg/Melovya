"""
Generates audio demonstrating User's Exact Repetition Scenario:
- A3 to G#3 and again come to A3 (slight up and down) -> Repeated notes -> 2 words with different letters!
- Single note extended for a long time (D3 for 1040ms) -> 1 letter only!
- A3 repeated with slight gap (50ms) -> Repeated notes -> 2 words with different letters!
"""

import numpy as np
import wave

def generate_note(freq_hz, duration_s, sample_rate=44100):
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False)
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
    return np.random.normal(0, 0.0005, num_samples)

def create_audio(filename="sample_user_repeated_and_extended_notes.wav"):
    sr = 44100

    # 1. Repeated note via slight note shift: A3 (220 Hz) -> G#3 (207.65 Hz dip) -> A3 (220 Hz)
    n1 = generate_note(220.00, 0.40, sr)
    dip = generate_note(207.65, 0.06, sr)
    n2 = generate_note(220.00, 0.40, sr)
    b1 = generate_silence(0.30, sr)

    # 2. Extended note: D3 (146.83 Hz) held for 1040ms (4 matras at 83 BPM)
    n3 = generate_note(146.83, 1.04, sr)
    longer_gap = generate_silence(0.80, sr)

    # 3. Repeated note via slight gap: A3 (220 Hz) -> 50ms gap -> A3 (220 Hz)
    n4 = generate_note(220.00, 0.40, sr)
    micro_gap = generate_silence(0.05, sr)
    n5 = generate_note(220.00, 0.40, sr)
    b2 = generate_silence(0.30, sr)

    # 4. Standard 2-note word: C3 (130.81 Hz) -> D3 (146.83 Hz)
    n6 = generate_note(130.81, 0.50, sr)
    n7 = generate_note(146.83, 0.50, sr)

    full_audio = np.concatenate([n1, dip, n2, b1, n3, longer_gap, n4, micro_gap, n5, b2, n6, n7])

    int_samples = (full_audio * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    print(f"Generated {filename} ({len(full_audio)/sr:.2f}s)")

if __name__ == '__main__':
    create_audio()
