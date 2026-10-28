"""
Breath-Based Prosody & Word Formation Analyzer
==============================================
Implements the User's Exact Breath-to-Word & Note-to-Letter Algorithm:

1. Breath Segmentation:
   - A singer sings notes continuously, then takes a breath (short silence).
   - Each group of continuous notes sung before taking a breath forms EXACTLY ONE WORD.
   - The number of notes sung before the breath = THE NUMBER OF LETTERS IN THAT WORD.
   - For example:
     - 2 notes sung before breath -> Word has 2 letters.
     - 3 notes sung before breath -> Word has 3 letters.

2. Note-to-Letter Matra Matching:
   - For each note in that word, find its matra weight (T_laghu = 60 / (BPM * 2.8)).
   - Letter 1 of the word has the exact matras of Note 1.
   - Letter 2 of the word has the exact matras of Note 2.
   - Letter k of the word has the exact matras of Note k.
   - For example:
     - Note 1 is 4 matras, Note 2 is 2 matras -> Word has 2 letters: Letter 1 (4m), Letter 2 (2m)
       e.g. 'ಆ~~' (4m) + 'ದೇ' (2m) = 'ಆ~~ದೇ'
     - Note 1 is 5 matras, Note 2 is 2 matras, Note 3 is 1 matra -> Word has 3 letters:
       Letter 1 (5m), Letter 2 (2m), Letter 3 (1m)
       e.g. 'ಹಾ~~~' (5m) + 'ಸೋ' (2m) + 'ನ' (1m) = 'ಹಾ~~~ಸೋನ'

3. Hierarchical Silence Hierarchy:
   - Breath Pause (e.g. 150ms <= silence < 550ms): marks the end of a WORD (adds a space).
   - Longer Gap (e.g. silence >= 550ms): marks the end of a LINE / SENTENCE (adds a newline).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

from .musical_quantizer import MusicalQuantizer, NoteEvent


# Open Kannada Akshara Palettes by Matra Weight
HRASVA_LETTERS = ['ಕ', 'ತ', 'ನ', 'ಲ', 'ಮ', 'ರ', 'ಸ', 'ದ', 'ಬ', 'ಗ', 'ಜ', 'ವ', 'ಯ', 'ಹ', 'ಅ', 'ಇ', 'ಉ', 'ಎ', 'ಪ', 'ಡ', 'ಚ']
DEERGHA_LETTERS = ['ಕಾ', 'ನೀ', 'ತೋ', 'ದೇ', 'ಹಾ', 'ಬಾ', 'ಕಂ', 'ನಂ', 'ಮಿಂ', 'ರಂ', 'ಜೀ', 'ಮೈ', 'ಪೂ', 'ಗೋ', 'ಆ', 'ಈ', 'ಊ', 'ಏ', 'ಓ', 'ಸೋ', 'ರಾ', 'ಮಾ', 'ದಾ', 'ವಾ']
PLUTA_BASES = ['ಆ', 'ಹಾ', 'ಸೋ', 'ನೀ', 'ದೇ', 'ತೋ', 'ಓ', 'ತಾ', 'ರಾ', 'ಬಾ', 'ಮಾ', 'ದೋ']


def assign_akshara_for_matra(matra: int, preferred_char: Optional[str] = None) -> str:
    """
    Returns a single Kannada akshara having EXACTLY `matra` matras:
    - 1 matra: Hrasva (Laghu) single letter (e.g. 'ಕ', 'ನ', 'ಲ', 'ಮ')
    - 2 matras: Deergha (Guru) single letter (e.g. 'ಕಾ', 'ನೀ', 'ದೇ', 'ಬಾ')
    - 3 matras: Pluta 3 (e.g. 'ಸೋ~', 'ಹಾ~', 'ನೀ~')
    - 4 matras: Pluta 4 (e.g. 'ಆ~~', 'ನೀ~~', 'ಹಾ~~')
    - 5 matras: Pluta 5 (e.g. 'ಹಾ~~~', 'ಆ~~~', 'ಸೋ~~~')
    - M matras (M >= 3): Base + (M - 2) tildes
    """
    if matra <= 1:
        if preferred_char and preferred_char in HRASVA_LETTERS:
            return preferred_char
        return random.choice(HRASVA_LETTERS)
    elif matra == 2:
        if preferred_char and preferred_char in DEERGHA_LETTERS:
            return preferred_char
        return random.choice(DEERGHA_LETTERS)
    else:
        # Pluta: M >= 3
        tildes = "~" * (matra - 2)
        base = preferred_char if (preferred_char and preferred_char in PLUTA_BASES) else random.choice(PLUTA_BASES)
        return f"{base}{tildes}"


@dataclass
class BreathNote:
    note_index: int       # 1-based index within song/line
    pitch_hz: Optional[float]
    note_name: str        # e.g. "A3", "C3"
    duration_ms: float
    matras: int           # e.g. 4
    letter: str           # e.g. "ಆ~~"


@dataclass
class BreathWord:
    word_index: int            # 1-based index within line
    notes: List[BreathNote]    # notes sung before this breath
    letter_count: int          # number of notes = number of letters
    matra_pattern: List[int]   # e.g. [4, 2]
    total_matras: int          # e.g. 6
    letters: List[str]         # e.g. ['ಆ~~', 'ದೇ']
    word_text: str             # e.g. 'ಆ~~ದೇ'


@dataclass
class BreathLine:
    line_index: int            # 1-based index
    words: List[BreathWord]
    line_text: str             # e.g. 'ಆ~~ದೇ ಹಾ~~~ಸೋನ'
    total_matras: int
    word_count: int
    letter_count: int
    matra_breakdown: str       # e.g. '[4m, 2m] [5m, 2m, 1m]'


class BreathProsodyAnalyzer:
    """
    Hierarchical Analyzer for:
    1. Detecting breath pauses (inter-word) vs longer gaps (inter-line).
    2. Grouping continuous notes between breath pauses into single words.
    3. Constructing words where:
       - Word length (in letters) = number of notes in breath group.
       - Letter k has the exact matras of Note k.
    """

    def __init__(
        self,
        bpm: float = 83.0,
        time_signature: str = '6/8',
        scale_root: str = 'C',
        scale_type: str = 'major',
        breath_pause_min_s: float = 0.15,  # Breath pause between words: 150ms - 550ms
        longer_gap_min_s: float = 0.55     # Longer gap between lines: >= 550ms
    ):
        self.bpm = bpm
        self.time_signature = time_signature
        self.scale_root = scale_root
        self.scale_type = scale_type
        self.breath_pause_min_s = breath_pause_min_s
        self.longer_gap_min_s = longer_gap_min_s

        self.quantizer = MusicalQuantizer(bpm=bpm, scale_root=scale_root, scale_type=scale_type, time_signature=time_signature)
        self.t_laghu_ms = self.quantizer.t_laghu_ms

    def form_word_from_notes(
        self,
        word_idx: int,
        note_events: List[NoteEvent],
        preferred_letters: Optional[List[str]] = None
    ) -> BreathWord:
        """
        Takes N notes sung before a breath and forms 1 word of N letters.
        Letter k has the exact matras of Note k.
        """
        breath_notes = []
        letters = []
        matras_list = []

        for idx, ev in enumerate(note_events, 1):
            dur_ms = ev.duration_ms
            # Quantize matras: M = round(dur_ms / T_laghu)
            m = max(1, int(round(dur_ms / self.t_laghu_ms)))
            matras_list.append(m)

            # Snap pitch
            note_name = "REST"
            hz = ev.pitch_hz
            if hz and hz > 0:
                _, note_name, hz = self.quantizer.snap_pitch(hz, None, None)

            # Assign single letter with matching matras
            pref = preferred_letters[idx - 1] if (preferred_letters and idx - 1 < len(preferred_letters)) else None
            akshara = assign_akshara_for_matra(m, pref)
            letters.append(akshara)

            breath_notes.append(BreathNote(
                note_index=idx,
                pitch_hz=hz,
                note_name=note_name,
                duration_ms=dur_ms,
                matras=m,
                letter=akshara
            ))

        # Combine letters into single word (no spaces between letters of the same word)
        word_text = "".join(letters)

        return BreathWord(
            word_index=word_idx,
            notes=breath_notes,
            letter_count=len(breath_notes),
            matra_pattern=matras_list,
            total_matras=sum(matras_list),
            letters=letters,
            word_text=word_text
        )

    def form_lines_from_breath_groups(
        self,
        lines_of_words: List[List[List[NoteEvent]]],
        custom_letters: Optional[List[List[List[str]]]] = None
    ) -> List[BreathLine]:
        """
        Constructs lines and words from structured note groups:
        lines_of_words = [
            [ [note1, note2], [note3, note4, note5] ],   # Line 1: Word 1 (2 notes), Word 2 (3 notes)
            [ [note6, note7, note8], [note9, note10] ]   # Line 2: Word 3 (3 notes), Word 4 (2 notes)
        ]
        """
        lines = []

        for line_idx, word_note_groups in enumerate(lines_of_words, 1):
            words = []
            for word_idx, notes in enumerate(word_note_groups, 1):
                pref_letters = None
                if custom_letters and line_idx - 1 < len(custom_letters):
                    l_prefs = custom_letters[line_idx - 1]
                    if word_idx - 1 < len(l_prefs):
                        pref_letters = l_prefs[word_idx - 1]

                word = self.form_word_from_notes(word_idx, notes, pref_letters)
                words.append(word)

            line_text = " ".join(w.word_text for w in words)
            total_matras = sum(w.total_matras for w in words)
            total_letters = sum(w.letter_count for w in words)
            breakdown = " ".join(f"[{', '.join(f'{m}m' for m in w.matra_pattern)}]" for w in words)

            lines.append(BreathLine(
                line_index=line_idx,
                words=words,
                line_text=line_text,
                total_matras=total_matras,
                word_count=len(words),
                letter_count=total_letters,
                matra_breakdown=breakdown
            ))

        return lines

    def segment_audio_by_breaths_and_gaps(
        self,
        signal: np.ndarray,
        sample_rate: int,
        frame_duration_s: float = 0.03,
        hop_duration_s: float = 0.015,
        energy_threshold_ratio: float = 0.12
    ) -> List[List[Tuple[float, float]]]:
        """
        Hierarchical silence segmentation:
        1. Computes short-time RMS energy.
        2. Detects silent intervals.
        3. Silent interval >= longer_gap_min_s -> Splits into a new LINE.
        4. Breath pause_min_s <= Silent interval < longer_gap_min_s -> Splits into a new WORD within that line.
        Returns:
            List of Lines, where each Line is a List of Word audio intervals (start_s, end_s).
        """
        frame_len = int(frame_duration_s * sample_rate)
        hop_len = int(hop_duration_s * sample_rate)
        total_len = len(signal)
        num_frames = max(1, (total_len - frame_len) // hop_len)

        rms = np.zeros(num_frames)
        for i in range(num_frames):
            start = i * hop_len
            f = signal[start:start + frame_len]
            rms[i] = np.sqrt(np.mean(f ** 2) + 1e-12)

        peak_rms = np.max(rms) if len(rms) > 0 else 1.0
        thresh = max(0.008, peak_rms * energy_threshold_ratio)
        is_voiced = rms > thresh

        # Extract continuous vocal chunks
        vocal_chunks = []
        in_vocal = False
        start_f = 0
        for i, val in enumerate(is_voiced):
            if val and not in_vocal:
                in_vocal = True
                start_f = i
            elif not val and in_vocal:
                in_vocal = False
                vocal_chunks.append((start_f * hop_duration_s, i * hop_duration_s))
        if in_vocal:
            vocal_chunks.append((start_f * hop_duration_s, num_frames * hop_duration_s))

        if not vocal_chunks:
            return [[(0.0, total_len / sample_rate)]]

        # Classify gaps into Words vs Lines
        lines = []
        current_line_words = []
        curr_start, curr_end = vocal_chunks[0]

        for s, e in vocal_chunks[1:]:
            gap = s - curr_end
            if gap >= self.longer_gap_min_s:
                # Long gap -> Line break
                current_line_words.append((curr_start, curr_end))
                lines.append(current_line_words)
                current_line_words = []
                curr_start = s
                curr_end = e
            elif gap >= self.breath_pause_min_s:
                # Breath pause -> Word break
                current_line_words.append((curr_start, curr_end))
                curr_start = s
                curr_end = e
            else:
                # Micro hesitation (< breath_pause_min_s) -> Same word
                curr_end = e

        current_line_words.append((curr_start, curr_end))
        if current_line_words:
            lines.append(current_line_words)

        return lines
