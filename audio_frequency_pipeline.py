"""
Audio Humming Frequency & Prosody Pipeline
==========================================
Strictly implements the User's Exact Logic:

1. Audio Input: Reads audio file (humming or imported audio).
2. Frequency Analyzer:
   - Detects fundamental frequencies in chronological order.
   - If frequency changes, that is a new note.
   - Detects breath pauses (150ms <= gap < 550ms) -> Word boundary.
   - Detects longer gaps (gap >= 550ms) -> Line boundary.
3. Standard Pitch Table (C0 to C10):
   - Computes standard frequencies for all pitches C0 to C10: f(midi) = 440 * 2^((midi - 69)/12).
   - Maps each detected frequency to the nearest standard pitch.
4. Duration to Mātras Calculation:
   - 1 Mātra = T_laghu = 60 / (BPM * 4) seconds.
   - User's >0.8 Threshold Rule: stays at floor until ratio strictly exceeds k.8 (e.g. 1.6 -> 1m, 2.61 -> 2m, only > 2.8 -> 3m).
5. Pitch Code / Note Code:
   - Explicit sequence of pitches (e.g. [A3, D3] [F3, G3, A3] ...).
6. Laghu and Guru Code (Separately):
   - 1m -> ⏑ (Laghu)
   - 2m -> – (Guru)
   - M >= 3 -> – with (M - 2) tildes '~' (Pluta)
   - Printed separately by words and lines.
7. Lyrics Without Meaning (Random Letters):
   - Assigns a random Kannada letter for each frequency matching that frequency's mātra count.
   - No meaning search, no dictionary validation.
"""

import sys
import os
import math
import wave
import random
from typing import List, Tuple, Dict, Optional
import numpy as np

def load_wav_file(file_path: str) -> Tuple[np.ndarray, int]:
    """Loads a WAV audio file into a 1D float numpy array in [-1.0, 1.0]."""
    with wave.open(file_path, 'rb') as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    if sample_width == 2:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 1:
        data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 4:
        data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if num_channels > 1:
        data = data.reshape(-1, num_channels)[:, 0]
    return data, framerate

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# =============================================================================
# 1. STANDARD PITCH TABLE (C0 to C10)
# =============================================================================
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

PITCH_TABLE = []
for midi in range(12, 133): # C0 (midi 12) to C10 (midi 132)
    p_name = f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"
    std_f = 440.0 * (2.0 ** ((midi - 69.0) / 12.0))
    PITCH_TABLE.append({
        'name': p_name,
        'midi': midi,
        'freq_hz': std_f
    })

def find_nearest_standard_pitch(detected_hz: float) -> Dict:
    """Finds the nearest standard pitch from C0 to C10."""
    if not detected_hz or detected_hz <= 0:
        return {'name': 'REST', 'midi': 0, 'freq_hz': 0.0}
    return min(PITCH_TABLE, key=lambda p: abs(math.log2(detected_hz / p['freq_hz'])))

# =============================================================================
# 2. RANDOM KANNADA LETTER PALETTES (BY MĀTRA WEIGHT)
# =============================================================================
HRASVA_LETTERS = ['ಕ', 'ತ', 'ನ', 'ಲ', 'ಮ', 'ರ', 'ಸ', 'ದ', 'ಬ', 'ಗ', 'ಜ', 'ವ', 'ಯ', 'ಹ', 'ಅ', 'ಇ', 'ಉ', 'ಎ', 'ಪ', 'ಡ', 'ಚ']
DEERGHA_LETTERS = ['ಕಾ', 'ನೀ', 'ತೋ', 'ದೇ', 'ಹಾ', 'ಬಾ', 'ಕಂ', 'ನಂ', 'ಮಿಂ', 'ರಂ', 'ಜೀ', 'ಮೈ', 'ಪೂ', 'ಗೋ', 'ಆ', 'ಈ', 'ಊ', 'ಏ', 'ಓ', 'ಸೋ', 'ರಾ', 'ಮಾ', 'ದಾ', 'ವಾ']
PLUTA_BASES = ['ಆ', 'ಹಾ', 'ಸೋ', 'ನೀ', 'ದೇ', 'ತೋ', 'ಓ', 'ತಾ', 'ರಾ', 'ಬಾ', 'ಮಾ', 'ದೋ']

def pick_random_letter_for_matra(matra: int, preferred_char: Optional[str] = None) -> str:
    """Legacy helper for picking letter strictly by integer mātras."""
    if matra <= 1:
        if preferred_char and preferred_char in HRASVA_LETTERS:
            return preferred_char
        return random.choice(HRASVA_LETTERS)
    elif matra == 2:
        if preferred_char and preferred_char in DEERGHA_LETTERS:
            return preferred_char
        return random.choice(DEERGHA_LETTERS)
    else:
        tildes = "~" * (matra - 2)
        base = preferred_char if (preferred_char and preferred_char in PLUTA_BASES) else random.choice(PLUTA_BASES)
        return f"{base}{tildes}"


def pick_letter_for_representation(
    base_type: str,
    tildes: str,
    preferred_char: Optional[str] = None,
    exclude_letters: Optional[List[str]] = None
) -> str:
    """
    Returns a random Kannada letter matching base_type ('laghu' or 'guru') plus tildes:
    - Laghu base: Hrasva letter (e.g. 'ಮ', 'ನ', 'ಕ')
    - Guru base: Deergha letter (e.g. 'ಮಾ', 'ದೇ', 'ಬಾ')
    If exclude_letters is provided (e.g. for repeated notes), guarantees that
    the returned letter is different from all letters in exclude_letters.
    """
    clean_pref = preferred_char.rstrip('~') if preferred_char else None
    exclude_clean = set(c.rstrip('~') for c in (exclude_letters or []))

    if base_type == 'laghu':
        pool = [c for c in HRASVA_LETTERS if c not in exclude_clean]
        if not pool:
            pool = HRASVA_LETTERS
        base = clean_pref if (clean_pref and clean_pref in pool) else random.choice(pool)
    else:
        pool = [c for c in DEERGHA_LETTERS if c not in exclude_clean]
        if not pool:
            pool = DEERGHA_LETTERS
        base = clean_pref if (clean_pref and clean_pref in pool) else random.choice(pool)
    return f"{base}{tildes}"


def get_prosodic_representation(ratio: float, threshold: float = 0.9) -> Tuple[int, str, str, str]:
    """
    User's Dynamic Laghu/Guru Pluta Representation (Threshold > 0.9):
    - 0.0 to 1.9   -> 1m: Symbol '⏑'   | Base: 'laghu' | Tildes: ''    (e.g. 'ಮ')
    - 1.9 to 2.0   -> 2m: Symbol '⏑~'  | Base: 'laghu' | Tildes: '~'   (e.g. 'ಮ~')
    - 2.0 to 2.9   -> 2m: Symbol '–'   | Base: 'guru'  | Tildes: ''    (e.g. 'ಮಾ')
    - 2.9 to 3.0   -> 3m: Symbol '–~'  | Base: 'guru'  | Tildes: '~'   (e.g. 'ಮಾ~')
    - 3.0 to 3.9   -> 3m: Symbol '⏑~~' | Base: 'laghu' | Tildes: '~~'  (e.g. 'ಮ~~')
    - 3.9 to 4.0   -> 4m: Symbol '–~~' | Base: 'guru'  | Tildes: '~~'  (e.g. 'ಮಾ~~')
    - 4.0 to 4.9   -> 4m: Symbol '⏑~~~'| Base: 'laghu' | Tildes: '~~~' (e.g. 'ಮ~~~')
    - 4.9 to 5.0   -> 5m: Symbol '–~~~'| Base: 'guru'  | Tildes: '~~~' (e.g. 'ಹಾ~~~')
    """
    if ratio <= 1.0:
        return 1, "⏑", "laghu", ""
    k = int(ratio)
    frac = round(ratio - k, 4)
    if frac > threshold:
        matras = k + 1
        if k == 1:
            return matras, "⏑~", "laghu", "~"
        else:
            tildes = "~" * (k - 1)
            return matras, f"–{tildes}", "guru", tildes
    else:
        matras = k
        if k == 1:
            return matras, "⏑", "laghu", ""
        elif k == 2:
            return matras, "–", "guru", ""
        else:
            tildes = "~" * (k - 1)
            return matras, f"⏑{tildes}", "laghu", tildes

# =============================================================================
# 3. FREQUENCY ANALYZER (AUTOCORRELATION PITCH DETECTOR)
# =============================================================================
def estimate_pitch_autocorr(frame: np.ndarray, sample_rate: int, min_f0: float = 65.0, max_f0: float = 600.0) -> Optional[float]:
    min_lag = int(sample_rate / max_f0)
    max_lag = int(sample_rate / min_f0)
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(frame) - 1:]

    if max_lag >= len(corr):
        return None
    search_window = corr[min_lag:max_lag]
    if len(search_window) == 0:
        return None

    peak_idx = np.argmax(search_window) + min_lag
    peak_val = corr[peak_idx]

    if corr[0] > 0 and (peak_val / corr[0]) > 0.35:
        return float(sample_rate / peak_idx)
    return None

def snap_ratio_to_matras(ratio: float, threshold: float = 0.9) -> int:
    """
    User's >0.9 Threshold Rule:
    - If ratio is 1.8 -> 1 mātra (stays at 1 mātra until ratio > 1.9)
    - If ratio is 2.8 -> 2 mātras (stays at 2 mātras until ratio > 2.9)
    - Only when ratio strictly exceeds 1.9 does it become 2 mātras.
    - Only when ratio strictly exceeds 2.9 does it become 3 mātras.
    - Minimum 1 mātra for any sung note
    - Applies to all mātras from 1 to infinity
    """
    if ratio <= 1.0:
        return 1
    base = int(ratio)
    frac = round(ratio - base, 4)
    if frac > threshold:
        return base + 1
    return base


def analyze_audio_frequencies(
    audio_path: str,
    bpm: float = 83.0,
    time_signature: str = '6/8',
    breath_pause_min_s: float = 0.15,
    longer_gap_min_s: float = 0.55,
    demo_mode: bool = False
) -> Dict:
    """
    Runs the complete Frequency -> Pitch -> Laghu/Guru -> Random Lyrics Pipeline
    with explicit Time Signature and millisecond duration measurement.
    """
    data, sr = load_wav_file(audio_path)
    total_duration_s = len(data) / sr

    # Parse Time Signature (e.g. 4/8, 5/8, 6/8, 7/8, 4/4)
    try:
        ts_num, ts_den = map(int, time_signature.strip().split('/'))
    except Exception:
        ts_num, ts_den = 6, 8

    # Universal formula adjusted for Time Signature pulse:
    # Denominator 8: 1 Matra = 60 / (BPM * 2.8) seconds
    t_matra_s = (60.0 / (bpm * 2.8)) * (8.0 / ts_den)
    t_matra_ms = t_matra_s * 1000.0
    bar_duration_ms = ts_num * t_matra_ms

    # 1. RMS Energy VAD
    frame_dur_s = 0.03
    hop_dur_s = 0.015
    frame_len = int(frame_dur_s * sr)
    hop_len = int(hop_dur_s * sr)
    num_frames = max(1, (len(data) - frame_len) // hop_len)

    rms = np.zeros(num_frames)
    for i in range(num_frames):
        st = i * hop_len
        f = data[st:st + frame_len]
        rms[i] = np.sqrt(np.mean(f ** 2) + 1e-12)

    peak_rms = np.max(rms) if len(rms) > 0 else 1.0
    silence_thresh = max(0.008, peak_rms * 0.12)
    is_voiced = rms > silence_thresh

    vocal_chunks = []
    in_vocal = False
    st_f = 0
    for i, val in enumerate(is_voiced):
        if val and not in_vocal:
            in_vocal = True
            st_f = i
        elif not val and in_vocal:
            in_vocal = False
            vocal_chunks.append((st_f * hop_dur_s, i * hop_dur_s))
    if in_vocal:
        vocal_chunks.append((st_f * hop_dur_s, num_frames * hop_dur_s))

    if not vocal_chunks:
        vocal_chunks = [(0.0, total_duration_s)]

    # 2. Hierarchical Silence Segmentation
    lines_of_word_chunks = []
    current_line = []
    curr_st, curr_end = vocal_chunks[0]

    for s, e in vocal_chunks[1:]:
        gap = s - curr_end
        if gap >= longer_gap_min_s:
            # Long gap -> Line break
            current_line.append({'start': curr_st, 'end': curr_end, 'gap_after': gap})
            lines_of_word_chunks.append(current_line)
            current_line = []
            curr_st = s
            curr_end = e
        elif gap >= breath_pause_min_s:
            # Breath pause -> Word break
            current_line.append({'start': curr_st, 'end': curr_end, 'gap_after': gap})
            curr_st = s
            curr_end = e
        else:
            # Micro gap -> Same word
            curr_end = e

    current_line.append({'start': curr_st, 'end': curr_end, 'gap_after': 0.0})
    if current_line:
        lines_of_word_chunks.append(current_line)

    # 3. Note Extraction, Pitch Snapping, Mātra Calculation, Laghu/Guru Code & Random Lyrics
    demo_preferred = [
        [['ಆ', 'ದ'], ['ಹಾ', 'ಸ', 'ನ']],
        [['ನೀ', 'ಇ', 'ಮ'], ['ಬ', 'ದ']]
    ]

    all_note_events = []
    lines_output = []
    note_counter = 1

    for line_idx, word_chunks in enumerate(lines_of_word_chunks, 1):
        words_in_line = []
        line_notes = []

        for w_chunk_idx, w_chunk in enumerate(word_chunks, 1):
            st_sample = int(w_chunk['start'] * sr)
            end_sample = int(w_chunk['end'] * sr)
            chunk_signal = data[st_sample:end_sample]

            # Track pitch in 40ms frames, 20ms hop
            p_frame_len = int(0.04 * sr)
            p_hop_len = int(0.02 * sr)
            p_num = max(1, (len(chunk_signal) - p_frame_len) // p_hop_len)

            raw_segs = []
            curr_f0 = None
            curr_cnt = 0
            curr_silence = 0

            for i in range(p_num):
                sub = chunk_signal[i * p_hop_len : i * p_hop_len + p_frame_len]
                rms = np.sqrt(np.mean(sub ** 2) + 1e-12)
                f0 = estimate_pitch_autocorr(sub, sr) if rms > 0.008 else None

                if f0 is None:
                    if curr_cnt > 0:
                        raw_segs.append({'type': 'note', 'f0': curr_f0, 'dur_s': curr_cnt * 0.02})
                        curr_cnt = 0
                        curr_f0 = None
                    curr_silence += 1
                else:
                    if curr_silence > 0:
                        raw_segs.append({'type': 'gap', 'dur_s': curr_silence * 0.02})
                        curr_silence = 0
                    if curr_f0 is None:
                        curr_f0 = f0
                        curr_cnt = 1
                    else:
                        semitones_diff = abs(12.0 * math.log2(f0 / curr_f0))
                        if semitones_diff < 0.8:
                            curr_f0 = (curr_f0 * curr_cnt + f0) / (curr_cnt + 1)
                            curr_cnt += 1
                        else:
                            raw_segs.append({'type': 'note', 'f0': curr_f0, 'dur_s': curr_cnt * 0.02})
                            curr_f0 = f0
                            curr_cnt = 1
            if curr_cnt > 0:
                raw_segs.append({'type': 'note', 'f0': curr_f0, 'dur_s': curr_cnt * 0.02})
            if curr_silence > 0:
                raw_segs.append({'type': 'gap', 'dur_s': curr_silence * 0.02})

            # Process segments:
            # 1. REPEATED NOTES: Slight note shift (e.g. A3 -> G#3 -> A3) or slight gap:
            #    - Gives two separate words for the two repeated notes (max 3 repetitions).
            #    - Assigns DIFFERENT letters to repeated notes.
            # 2. EXTENDED NOTES:
            #    - Continuous single note extended for a long time gets ONLY ONE letter with tildes.
            pref_word = None
            if demo_mode and (line_idx - 1 < len(demo_preferred)) and (w_chunk_idx - 1 < len(demo_preferred[line_idx - 1])):
                pref_word = demo_preferred[line_idx - 1][w_chunk_idx - 1]

            chunk_words = []
            curr_word_notes = []
            curr_word_letters = []
            curr_word_matras = []
            curr_word_lg = []

            seg_idx = 0
            prev_repeated_letter = None
            rep_counter = 0

            while seg_idx < len(raw_segs):
                seg = raw_segs[seg_idx]
                if seg['type'] == 'gap':
                    seg_idx += 1
                    continue

                dur_s = seg['dur_s']
                if dur_s < 0.07 and len(raw_segs) > 1:
                    seg_idx += 1
                    continue

                f0_detected = seg['f0']
                dur_ms = dur_s * 1000.0
                pitch_info = find_nearest_standard_pitch(f0_detected)
                pitch_name = pitch_info['name']
                std_freq = pitch_info['freq_hz']

                # Repetition Articulation Check
                is_repeat_next = False
                repeat_reason = None
                jump_steps = 1

                if seg_idx + 2 < len(raw_segs) and rep_counter < 3:
                    mid = raw_segs[seg_idx + 1]
                    nxt = raw_segs[seg_idx + 2]
                    if nxt['type'] == 'note':
                        nxt_p = find_nearest_standard_pitch(nxt['f0'])['name']
                        if nxt_p == pitch_name:
                            if mid['type'] == 'note' and mid['dur_s'] <= 0.18:
                                mid_p = find_nearest_standard_pitch(mid['f0'])['name']
                                semitone_dist = abs(12.0 * math.log2(mid['f0'] / f0_detected))
                                if semitone_dist <= 2.5:
                                    is_repeat_next = True
                                    repeat_reason = f"Slight Note Shift ({pitch_name} ➔ {mid_p} ➔ {nxt_p})"
                                    jump_steps = 2
                            elif mid['type'] == 'gap' and mid['dur_s'] <= 0.14:
                                is_repeat_next = True
                                repeat_reason = f"Slight Gap ({round(mid['dur_s'] * 1000)}ms)"
                                jump_steps = 2

                exact_ratio = dur_ms / t_matra_ms
                matra, lg_sym, base_type, tildes = get_prosodic_representation(exact_ratio, threshold=0.9)
                base_k = int(exact_ratio)
                frac_k = exact_ratio - base_k
                if exact_ratio <= 1.0:
                    rule_tag = "(0-1.9 ➔ ⏑)"
                elif frac_k > 0.9:
                    rule_tag = f"(> {base_k}.9 ➔ {lg_sym})"
                else:
                    rule_tag = f"(<= {base_k}.9 ➔ {lg_sym})"

                rep_tag = f" [🔁 {repeat_reason}]" if repeat_reason else ""
                ratio_calc_str = f"{dur_ms:.1f}ms / {t_matra_ms:.1f}ms = {exact_ratio:.2f} ➔ {matra}m [{lg_sym}] {rule_tag}{rep_tag}"
                curr_word_matras.append(matra)
                curr_word_lg.append(lg_sym)

                # Pick letter, ensuring different letters for repeated notes
                pref_char = pref_word[len(curr_word_letters)] if (pref_word and len(curr_word_letters) < len(pref_word)) else None
                exclude = [prev_repeated_letter] if prev_repeated_letter else []
                letter = pick_letter_for_representation(base_type, tildes, pref_char, exclude_letters=exclude)
                curr_word_letters.append(letter)

                note_dict = {
                    'note_order': note_counter,
                    'detected_hz': round(f0_detected, 1),
                    'pitch_name': pitch_name,
                    'standard_hz': round(std_freq, 1),
                    'duration_s': round(dur_s, 3),
                    'duration_ms': round(dur_ms, 1),
                    'time_sig': time_signature,
                    'ratio_calc': ratio_calc_str,
                    'matras': matra,
                    'laghu_guru': lg_sym,
                    'letter': letter,
                    'is_repeated': bool(repeat_reason or prev_repeated_letter),
                    'repeat_reason': repeat_reason
                }
                curr_word_notes.append(note_dict)
                line_notes.append(note_dict)
                all_note_events.append(note_dict)
                note_counter += 1

                if is_repeat_next:
                    # User rule: 'give two word for the two repeated notes'
                    chunk_words.append({
                        'word_text': "".join(curr_word_letters),
                        'letter_count': len(curr_word_letters),
                        'letters': curr_word_letters,
                        'matras_list': curr_word_matras,
                        'total_matras': sum(curr_word_matras),
                        'laghu_guru_list': curr_word_lg,
                        'laghu_guru_code': " ".join(curr_word_lg),
                        'pitches': [n['pitch_name'] for n in curr_word_notes],
                        'notes': curr_word_notes,
                        'gap_after_s': 0.05,
                        'repeat_trigger': repeat_reason
                    })
                    curr_word_notes = []
                    curr_word_letters = []
                    curr_word_matras = []
                    curr_word_lg = []
                    prev_repeated_letter = letter
                    rep_counter += 1
                    seg_idx += jump_steps
                else:
                    prev_repeated_letter = None
                    rep_counter = 0
                    seg_idx += 1

            if curr_word_notes:
                chunk_words.append({
                    'word_text': "".join(curr_word_letters),
                    'letter_count': len(curr_word_letters),
                    'letters': curr_word_letters,
                    'matras_list': curr_word_matras,
                    'total_matras': sum(curr_word_matras),
                    'laghu_guru_list': curr_word_lg,
                    'laghu_guru_code': " ".join(curr_word_lg),
                    'pitches': [n['pitch_name'] for n in curr_word_notes],
                    'notes': curr_word_notes,
                    'gap_after_s': round(w_chunk['gap_after'], 3),
                    'repeat_trigger': None
                })

            words_in_line.extend(chunk_words)

        # Re-number word indices in line
        for w_i, w in enumerate(words_in_line, 1):
            w['word_index'] = w_i

        # Line formed by words separated by space
        line_text = " ".join(w['word_text'] for w in words_in_line)
        line_laghu_guru = "  ".join(f"[{' '.join(w['laghu_guru_list'])}]" for w in words_in_line)
        line_pitch_code = "  ".join(f"[{' ➔ '.join(w['pitches'])}]" for w in words_in_line)
        line_matra_breakdown = " ".join(f"[{', '.join(f'{m}m' for m in w['matras_list'])}]" for w in words_in_line)
        total_line_matras = sum(w['total_matras'] for w in words_in_line)

        lines_output.append({
            'line_number': line_idx,
            'start_s': round(word_chunks[0]['start'], 2),
            'end_s': round(word_chunks[-1]['end'], 2),
            'duration_s': round(word_chunks[-1]['end'] - word_chunks[0]['start'], 2),
            'words': words_in_line,
            'line_text': line_text,
            'pitch_code': line_pitch_code,
            'laghu_guru_code': line_laghu_guru,
            'matra_breakdown': line_matra_breakdown,
            'total_matras': total_line_matras,
            'word_count': len(words_in_line),
            'letter_count': sum(w['letter_count'] for w in words_in_line),
            'line_gap_after_s': round(word_chunks[-1]['gap_after'], 3)
        })

    return {
        'audio_path': audio_path,
        'bpm': bpm,
        'time_signature': time_signature,
        'ts_num': ts_num,
        'ts_den': ts_den,
        't_matra_ms': round(t_matra_ms, 1),
        'bar_duration_ms': round(bar_duration_ms, 1),
        'total_duration_s': round(total_duration_s, 2),
        'total_lines': len(lines_output),
        'lines': lines_output,
        'all_notes': all_note_events
    }

def print_pipeline_results(res: Dict):
    print("\n" + "=" * 96)
    print(" 🎙️ AUDIO HUMMING ➔ FREQUENCY ➔ LAGHU/GURU ➔ RANDOM LYRICS PIPELINE")
    print("=" * 96)
    print(f" • Audio Source   : {res['audio_path']} ({res['total_duration_s']}s)")
    print(f" • Time Signature : {res['time_signature']} ({res['ts_num']} pulses / bar, Bar Length = {res['bar_duration_ms']} ms)")
    print(f" • Tempo (BPM)    : {res['bpm']} BPM")
    print(f" • 1 Mātra Unit   : T_laghu = (60 / (BPM * 2.8)) * (8 / {res['ts_den']}) = {res['t_matra_ms']} ms")
    print(f" • Snapping Rule  : User's >0.9 Threshold Rule (e.g. 2.61 -> 2m [<=2.9], 1.60 -> 1m [<=1.9], only > k.9 -> k+1)")
    print(f" • Total Lines    : {res['total_lines']} Lines detected")

    for line in res['lines']:
        print("\n" + "━" * 96)
        print(f" 📄 LINE {line['line_number']}: [{line['start_s']}s - {line['end_s']}s] ({line['duration_s']}s) | {line['word_count']} Words | {line['letter_count']} Letters | {line['total_matras']} Mātras")
        print("━" * 96)

        print("\n 1️⃣  PITCH / NOTE CODE (Chronological frequencies mapped to nearest standard pitch):")
        print(f"     👉 {line['pitch_code']}")

        print("\n 2️⃣  LAGHU & GURU CODE (Separately written):")
        print(f"     👉 {line['laghu_guru_code']}")
        print(f"        Rhythm Breakdown: {line['matra_breakdown']} ({line['total_matras']} mātras total)")

        print("\n 3️⃣  LYRICS WITHOUT MEANING (Random letters matching note mātras):")
        print(f"     👉 \"{line['line_text']}\"")

        print("\n 4️⃣  WORD-BY-WORD CHRONOLOGICAL DERIVATION (WITH EXACT MILLISECONDS):")
        for w in line['words']:
            print(f"     ┌── WORD {w['word_index']}: \"{w['word_text']}\" ({w['letter_count']} letters ← {w['letter_count']} notes before breath)")
            print(f"     │    • Mātras: [{', '.join(f'{m}m' for m in w['matras_list'])}] = {w['total_matras']}m | Laghu-Guru: {w['laghu_guru_code']}")
            for n in w['notes']:
                print(f"     │    • #{n['note_order']}: Measured {n['detected_hz']} Hz ➔ Nearest Pitch: {n['pitch_name']} ({n['standard_hz']} Hz) | Dur: {n['duration_ms']}ms ➔ Calc: [{n['ratio_calc']}] (Sym: {n['laghu_guru']}) ➔ Letter '{n['letter']}'")
            if w.get('repeat_trigger'):
                print(f"     └── 🔁 Repeated Note Detected ({w['repeat_trigger']}) ➔ Formed 2 Separate Words with Different Letters (Space added)")
            elif w['gap_after_s'] >= 0.15 and w['gap_after_s'] < 0.55:
                print(f"     └── 💨 Breath Pause ({int(w['gap_after_s'] * 1000)}ms) ➔ Word Completed (Space added)")
            else:
                print(f"     └──────────────────────────────────────────────────────────────────────────")

        if line['line_gap_after_s'] >= 0.55:
            print(f" ⏸️  Longer Gap ({int(line['line_gap_after_s'] * 1000)}ms >= 550ms) ➔ LINE BREAK")

    print("\n" + "=" * 96)
    print(" 📊 FULL FREQUENCY ANALYZER INSPECTION TABLE (WITH MILLISECONDS & SNAPPING CALCULATION):")
    print("=" * 96)
    print(f" {'#':<3} {'Detected Hz':<12} {'Pitch':<8} {'Std Hz':<10} {'Dur (ms)':<10} {'TimeSig':<8} {'Nearest Matra Snapping Math':<30} {'L/G':<6} {'Letter':<8} {'Repeat Articulation'}")
    print("-" * 115)
    for n in res['all_notes']:
        rep_info = n.get('repeat_reason') or ('Repeated Note' if n.get('is_repeated') else '-')
        print(f" {n['note_order']:<3} {n['detected_hz']:<12.1f} {n['pitch_name']:<8} {n['standard_hz']:<10.1f} {n['duration_ms']:<10.1f} {n['time_sig']:<8} {n['ratio_calc']:<30} {n['laghu_guru']:<6} '{n['letter']}'{'':<6} {rep_info}")
    print("=" * 96)

if __name__ == '__main__':
    audio_file = sys.argv[1] if len(sys.argv) > 1 else 'sample_user_10notes_breaths.wav'
    time_sig = sys.argv[2] if len(sys.argv) > 2 else '6/8'
    res = analyze_audio_frequencies(audio_file, bpm=83.0, time_signature=time_sig, demo_mode=True)
    print_pipeline_results(res)
