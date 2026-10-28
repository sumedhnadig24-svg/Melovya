"""
Open Prosody Synthesizer & Dynamic Lyrical Generator
Implements the User-Specified 2-Stage Algorithm:

1. Pitch Snapping & Sub-unit/Long-Note Decomposition:
   - Snaps pitch to nearest frequency using trend context.
   - If 2 sub-laghu notes occur, merges them into 1 Laghu.
   - If a note extends for 4 matras, assigns a 4-matra letter (e.g. ಆ~~) or Laghu/Guru/Pluta.
   - If a note extends long (e.g., 6 to 30 matras), decomposes 50%-75% into Laghus and Gurus,
     and keeps the tail as Pluta sustain.
   - Preserves musical gaps as word/phrase pauses.

2. STAGE 1: Raw Note-to-Single-Letter Prototype (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ):
   - For every single note, assigns a single Kannada letter/akshara of matching matra weight.
   - Unconstrained by dictionary: forms raw phonetic patterns (e.g., "ಆ~~ ಸೋ~ ...").
   - Verifies matra pattern match.

3. STAGE 2: Meaningful Lyrical Transformation (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ):
   - Takes the raw syllable blueprint and generates genuine, grammatically coherent Kannada words.
   - Conditioned on style: Mass (punchy/swagger) vs Melody (poetic/romantic) vs Janapada vs Sad.
   - Open vocabulary: NO repetitive fixed word lists! Uses rich non-repeating lexicon.
   - Enforces Antyaprāsa (end-rhyme) between paired lines.

4. Gatekeeper Verification:
   - Audits matra alignment, word-boundary beat lock, and sentence meaning.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .kannada_prosody import (
    KannadaProsodyEngine,
    extract_primary_consonant,
    extract_vowel,
    calculate_consonant_phonic_score,
    check_swara_similarity
)
from .kannada_semantics import KannadaSemanticEngine
from .musical_quantizer import MusicalQuantizer, NoteEvent


# Open Kannada Syllable Palette by Matra Weight (for Stage 1 Raw Note-to-Letter Mapping)
RAW_SYLLABLE_PALETTE = {
    1: ['ಕ', 'ತ', 'ನ', 'ಲ', 'ಮ', 'ರ', 'ಸ', 'ದ', 'ಬ', 'ಗ', 'ಜ', 'ವ', 'ಯ', 'ಹ', 'ಅ', 'ಇ', 'ಉ', 'ಎ', 'ಪ', 'ಡ', 'ಚ'],
    2: ['ಕಾ', 'ನೀ', 'ತೋ', 'ದೇ', 'ಹಾ', 'ಬಾ', 'ಕಂ', 'ನಂ', 'ಮಿಂ', 'ರಂ', 'ಜೀ', 'ಮೈ', 'ಪೂ', 'ಗೋ', 'ಆ', 'ಈ', 'ಊ', 'ಏ', 'ಓ', 'ಸೋ', 'ರಾ', 'ಮಾ'],
    3: ['ಸೋ~', 'ಹಾ~', 'ಬಾ~', 'ಕೇ~', 'ದೇ~', 'ನೀ~', 'ತೋ~', 'ಮೀ~', 'ಓ~', 'ಆ~', 'ರಾ~', 'ಲಾ~'],
    4: ['ಆ~~', 'ನೀ~~', 'ಓ~~', 'ತಾ~~', 'ದೇ~~', 'ಹಾ~~', 'ಸೋ~~', 'ಕಾಂ~~', 'ರಾ~~', 'ಬಾ~~']
}

# Rich Vocabulary categorized by Matra Shape & Genre (NO repetitive fixed 4-word lists)
GENRE_VOCABULARY = {
    'mass': {
        (1,): ['ಗೆ', 'ನು', 'ಲಿ', 'ದು', 'ರು', 'ವು'],
        (2,): ['ನಾ', 'ಬಾ', 'ಹವಾ', 'ಪಟಾ', 'ಜೋರಾ', 'ಮಾಸ್ಸು'],
        (3,): ['ಹಾ~', 'ಬಾ~', 'ಜೋ~'],
        (4,): ['ಹಾ~~', 'ತಾ~~'],
        (1, 1): ['ಜನ', 'ಬಲ', 'ಗತಿ', 'ರಣ', 'ಪದ', 'ಲಯ'],
        (2, 1): [
            'ನೋಡು', 'ಹಾಡು', 'ಕಿಕ್ಕು', 'ಜೋರು', 'ಪವರ್', 'ಖದರ್', 'ಬೆಂಕಿ', 'ಗತ್ತು',
            'ಧೂಳು', 'ಶಕ್ತಿ', 'ದಮ್ಮು', 'ಹವಾ', 'ಹುಲಿ', 'ಸಿಂಹ', 'ಸ್ಟೈಲು', 'ಜಿಲ್ಲೆ',
            'ರೌಂಡ್ಸ್', 'ಮೆಚ್ಚು', 'ಗೆದ್ದು', 'ಗುಮ್ಮು', 'ಕಿಲ್ಲರ್', 'ಕಿಂಗ್', 'ಗುಟುಕು', 'ರಂಪ'
        ],
        (2, 1, 1): [
            'ಸ್ಟೈಲಲಿ', 'ರಂಗನು', 'ಸಡಗರ', 'ಹಳ್ಳಿಯ', 'ಜಾತ್ರೆಯ', 'ಹುಮ್ಮಸ್ಸು', 'ಶಕ್ತಿಯ',
            'ಗತ್ತಲಿ', 'ಕೇರಿಯ', 'ಊರಲಿ', 'ಕಿಕ್ಕನು', 'ಬೆಂಕಿಯ', 'ದಮ್ಮನು', 'ಮಜವನು', 'ಗುಡುಗಿನ'
        ],
        (1, 2): ['ಬರೋ', 'ಗುರೂ', 'ದಳಾ', 'ಹವಾ', 'ಪಟಾ'],
        (2, 2): ['ರಂಗು', 'ಬೆಂಕಿ', 'ಪಕ್ಕಾ', 'ಜೋರಾ', 'ಮಾಸ್ಸು', 'ಗತ್ತೇ', 'ಖದರೇ', 'ಧೂಳೇ', 'ಕಿಕ್ಕೇ'],
        (2, 1, 2): [
            'ಚೆಲ್ಲಿದೆ', 'ಹತ್ತಿದೆ', 'ಗೆದ್ದಿದೆ', 'ನಿಂತಿದೆ', 'ಬಂದಿದೆ', 'ಕಂಡಿದೆ',
            'ಮೆಚ್ಚಿದೆ', 'ಬಿಟ್ಟಿದೆ', 'ಕೊಟ್ಟಿದೆ', 'ತಂದಿದೆ', 'ಸುಟ್ಟಿದೆ', 'ಎದ್ದಿದೆ', 'ಮುಟ್ಟಿದೆ'
        ],
        (1, 1, 2): ['ಮನವೇ', 'ಅವನೇ', 'ಗುರುವೇ', 'ಜನರೇ', 'ನಗರೇ', 'ದಳವೇ'],
        (1, 1, 1, 1): ['ಕುಣಿಕುಣಿದು', 'ನಗುನಗುತ', 'ಮಿಂಚುತಿದೆ', 'ಸದ್ದುಗದ್ದು', 'ಅಬ್ಬರದಿ', 'ಗುಡುಗುಡುತ']
    },
    'melody': {
        (1,): ['ಗೆ', 'ಲಿ', 'ನು', 'ದು', 'ರು', 'ವು'],
        (2,): ['ನಾ', 'ನೀ', 'ಹಾ', 'ಬಾ', 'ಮೈ', 'ಕೈ'],
        (3,): ['ಹಾ~', 'ಬಾ~', 'ನೀ~', 'ತೋ~'],
        (4,): ['ಆ~~', 'ನೀ~~', 'ಓ~~'],
        (1, 1): ['ಮನ', 'ಹಿತ', 'ಸದಾ', 'ಇದು', 'ಅದು', 'ಸುಖ', 'ಜತೆ', 'ಕರೆ', 'ಸ್ವರ', 'ಲಯ', 'ಪದ'],
        (2, 1): [
            'ನಿನ್ನ', 'ನನ್ನ', 'ಕಣ್ಣ', 'ಬಾಳ', 'ಜೀವ', 'ನೋಡಿ', 'ಕಂಡೆ', 'ತಂದೆ',
            'ಹಾಡು', 'ಬಂತು', 'ಹಾಗೆ', 'ಮೈಯ', 'ಮನ', 'ಪ್ರೀತಿ', 'ಪ್ರೇಮ', 'ಚೆಂದ',
            'ರೂಪ', 'ನೆನಪು', 'ತೀರ', 'ಅಲೆ', 'ಗಾನ', 'ಸ್ವರ', 'ಭಾವ', 'ಹೂವು', 'ಸೋಲು',
            'ಗೆಲವು', 'ಮೌನ', 'ತಂಪು', 'ಕಾವ', 'ಕಾಂತಿ', 'ಶಾಂತಿ', 'ಕಾಡು', 'ಬೆಟ್ಟ', 'ಜೇನು'
        ],
        (2, 1, 1): [
            'ಕಣ್ಣಲಿ', 'ಜೀವಕೆ', 'ತೇಲುತ', 'ಲವ್ವನು', 'ಮಿಂಚನು', 'ಚೇತನ', 'ಮನಸಲಿ',
            'ಕನಸಲಿ', 'ಪ್ರೀತಿಯ', 'ರೂಪನು', 'ನೆನಪನು', 'ಉಸಿರಲಿ', 'ಹೃದಯದ', 'ಸೊಬಗನು',
            'ಒಲವನು', 'ಗೆಳತಿಯ', 'ಗೆಳೆಯನ', 'ಅಲೆಯಲಿ', 'ತೀರದ', 'ಮೌನವ', 'ಚಂದನ', 'ನೋಟದ'
        ],
        (2, 2): ['ಬಾಳೇ', 'ಕಣ್ಣೇ', 'ಜೀವ್ವೇ', 'ಚೆಂದಾ', 'ಒಲವೇ', 'ಚೆಲುವೇ', 'ಕನಸೇ', 'ಪ್ರೀತೀ', 'ಗಾನಾ', 'ಭಾವಾ'],
        (2, 1, 2): [
            'ಒಲವಿನ', 'ಚೆಲುವಿನ', 'ಕಂಡಿದೆ', 'ಬಂದಿದೆ', 'ತುಂಬಿದೆ', 'ಬೆಂದಿದೆ',
            'ಮಿಂಚಿದೆ', 'ಚೆಲ್ಲಿದೆ', 'ಕಾದಿದೆ', 'ಸೇರಿದೆ', 'ಹಾರಿದೆ', 'ತೋರಿದೆ', 'ಮೂಡಿದೆ'
        ],
        (1, 1, 1, 1): [
            'ಮರೆತೆನು', 'ಅರಿತೆನು', 'ಬರೆದೆನು', 'ತಿಳಿದೆನು', 'ಕರೆದೆನು', 'ನೆನೆದೆನು',
            'ಬೆರೆತೆನು', 'ತೇಲಿಸು', 'ಸೇರಿಸು', 'ಹರಸಿದು', 'ಮುಳುಗಿದೆ'
        ],
        (1, 2): ['ಸದಾ', 'ನಿಜಾ', 'ಒಲಿದು', 'ಸಖೀ', 'ಪ್ರಿಯಾ'],
        (1, 1, 2): ['ಮನವೇ', 'ಒಲವೇ', 'ಚೆಲುವೇ', 'ಗೆಳತೀ', 'ಸಖಿಯೇ']
    },
    'janapada': {
        (1,): ['ಗೆ', 'ಲಿ', 'ನು', 'ದು'],
        (2,): ['ನಾ', 'ಬಾ', 'ಹಾ'],
        (1, 1): ['ಮಣ್ಣ', 'ಹೊಲ', 'ತೇರು', 'ಊರು'],
        (2, 1): ['ಚಿನ್ನ', 'ಬೆಣ್ಣೆ', 'ಮಣ್ಣು', 'ಕೆನ್ನೆ', 'ಹಳ್ಳಿ', 'ಹೊಲ', 'ತೇರು', 'ಜಾತ್ರೆ', 'ಕುಣಿತ', 'ಬಂಡಿ', 'ಹಾಡು'],
        (2, 1, 1): ['ಮಣ್ಣಲಿ', 'ಕೆನ್ನೆಯ', 'ಚಿನ್ನದ', 'ಬೆಣ್ಣೆಯ', 'ಹಳ್ಳಿಯ', 'ಜಾತ್ರೆಯ', 'ತೇರಲಿ', 'ಬಂಡಿಯ', 'ಊರಲಿ'],
        (2, 1, 2): ['ಚೆಲ್ಲಿದೆ', 'ಬಂದಿದೆ', 'ತುಂಬಿದೆ', 'ನಿಂತಿದೆ', 'ಹಾಡಿದೆ', 'ಕುಣಿದಿದೆ'],
        (2, 2): ['ಚೆಂದಾ', 'ಮಣ್ಣೇ', 'ಚಿನ್ನಾ', 'ಹಾಡೇ', 'ಊರೇ']
    },
    'sad': {
        (1,): ['ಗೆ', 'ಲಿ', 'ನು', 'ದು'],
        (2,): ['ನಾ', 'ನೋ'],
        (1, 1): ['ನೋವು', 'ಭಾರ', 'ದೂರ', 'ಸಾವು'],
        (2, 1): ['ನೋವು', 'ಮೌನ', 'ಭಾರ', 'ಕಣ್ಣೀರು', 'ಕತ್ತಲೆ', 'ದೂರ', 'ದಾರಿ', 'ಗೋಳು', 'ಸೋಲು', 'ಬೆಂಕಿ'],
        (2, 1, 1): ['ನೋವಲಿ', 'ಮೌನದ', 'ಭಾರಕೆ', 'ಕಣ್ಣೀರ', 'ವಿರಹದ', 'ಗೋಳಲಿ', 'ಮನಸಲಿ', 'ಕತ್ತಲೆ'],
        (2, 1, 2): ['ಬೆಂದಿದೆ', 'ತುಂಬಿದೆ', 'ಮುಗಿದಿದೆ', 'ಸೋತಿದೆ', 'ಕಳೆದಿದೆ', 'ಉಳಿದಿದೆ', 'ತೀರಿತೆ'],
        (1, 1, 1, 1): ['ಮರೆತೆನು', 'ಅಳಿಸುತ', 'ಕರಗುತ', 'ನೊಂದೆನು', 'ಸೋತೆನು']
    }
}


@dataclass
class DecomposedNote:
    note_name: str
    freq_hz: float
    mātras: int
    prosody_symbol: str
    is_rest: bool
    raw_letter: str  # Stage 1: Single letter assigned to this note


@dataclass
class StageResults:
    phrase_id: int
    decomposed_notes: List[DecomposedNote]
    stage1_raw_lyrics: str           # e.g. "ಆ~~ ಸೋ~ ಕ ತ ನ ..."
    stage1_note_letter_pairs: List[Tuple[str, str, int]] # [("A3", "ಆ~~", 4), ("C3", "ಸೋ~", 3), ...]
    stage1_syllable_count: int
    stage1_total_matras: int
    stage2_meaningful_lyrics: str    # e.g. "ನಿನ್ನ ಕಣ್ಣಲಿ ನೋಡಿ ಒಲವನು"
    stage2_meaning_score: float
    antya_rhyme_info: Optional[str]
    is_meter_valid: bool


class OpenProsodySynthesizer:
    """
    Core Engine for:
    1. Note & Matra Decomposition (splitting long sustained notes into Laghu/Guru + Pluta).
    2. Stage 1: Note-to-Single-Letter Prototype (unconstrained acoustic akshara mapping).
    3. Stage 2: Meaningful Lyrical Transformation (rich open vocabulary with style conditioning).
    """

    def __init__(self, bpm: float = 83.0, scale_root: str = 'C', scale_type: str = 'major', style: str = 'melody'):
        self.bpm = bpm
        self.scale_root = scale_root
        self.scale_type = scale_type
        self.style = style.lower()

        self.quantizer = MusicalQuantizer(bpm=bpm, scale_root=scale_root, scale_type=scale_type)
        self.t_laghu_ms = self.quantizer.t_laghu_ms
        self.prosody_engine = KannadaProsodyEngine()
        self.semantic_engine = KannadaSemanticEngine()

        # Track used words across lines to prevent repetitive lyric generation
        self.used_words_memory = set()

    def snap_and_decompose_notes(self, raw_events: List[NoteEvent]) -> List[DecomposedNote]:
        """
        Processes humming note events:
        1. Sub-laghu merging: If 2 consecutive notes have duration < 0.75 * T_laghu, merges into 1 Laghu.
        2. Long-note decomposition: If a note extends >= 5 matras (or up to 30 matras),
           converts 50%-75% into random Laghus (1) and Gurus (2), leaving the tail as Pluta (-~ / -~~).
        3. Assigns 1 single letter per resulting note (Stage 1).
        """
        decomposed: List[DecomposedNote] = []
        n = len(raw_events)
        i = 0

        while i < n:
            ev = raw_events[i]
            is_rest = (ev.pitch_hz is None or ev.pitch_hz <= 0)

            if is_rest:
                units = max(1, round(ev.duration_ms / self.t_laghu_ms))
                decomposed.append(DecomposedNote(
                    note_name="REST",
                    freq_hz=0.0,
                    mātras=0,
                    prosody_symbol=f"REST({units})",
                    is_rest=True,
                    raw_letter=" "
                ))
                i += 1
                continue

            # Pitch Snapping with Trend Logic
            prev_hz = raw_events[i - 1].pitch_hz if i > 0 else None
            next_hz = raw_events[i + 1].pitch_hz if i < n - 1 else None
            midi, note_name, std_hz = self.quantizer.snap_pitch(ev.pitch_hz, prev_hz, next_hz)

            dur_ms = ev.duration_ms

            # Rule 1: Sub-laghu check (If 2 notes come in half laghu, merge to 1 Laghu)
            if dur_ms < (self.t_laghu_ms * 0.75) and (i + 1 < n) and (raw_events[i + 1].pitch_hz is not None and raw_events[i + 1].pitch_hz > 0):
                next_dur = raw_events[i + 1].duration_ms
                if next_dur < (self.t_laghu_ms * 0.75):
                    # Merge 2 half-laghus into 1 Laghu
                    merged_dur = dur_ms + next_dur
                    units = max(1, round(merged_dur / self.t_laghu_ms))
                    raw_char = random.choice(RAW_SYLLABLE_PALETTE[1])
                    decomposed.append(DecomposedNote(
                        note_name=note_name,
                        freq_hz=std_hz,
                        mātras=1,
                        prosody_symbol="⏑",
                        is_rest=False,
                        raw_letter=raw_char
                    ))
                    i += 2
                    continue

            # Compute initial matra units
            total_units = max(1, round(dur_ms / self.t_laghu_ms))

            # Rule 2: Note <= 4 matras: keep intact as Laghu (1), Guru (2), Pluta 3 (-~), or Pluta 4 (-~~)
            if total_units <= 4:
                symbol = "⏑" if total_units == 1 else ("-" if total_units == 2 else ("-~" if total_units == 3 else "-~~"))
                raw_char = random.choice(RAW_SYLLABLE_PALETTE.get(total_units, RAW_SYLLABLE_PALETTE[2]))
                decomposed.append(DecomposedNote(
                    note_name=note_name,
                    freq_hz=std_hz,
                    mātras=total_units,
                    prosody_symbol=symbol,
                    is_rest=False,
                    raw_letter=raw_char
                ))
                i += 1
                continue

            # Rule 3: Long Note Decomposition (e.g., 5 to 30 matras)
            # "convert half of it, or three-quarters of it, into laghu and gurus randomly.
            # Then the remaining, you'll just do it as a pluta of the remaining notes."
            split_ratio = random.uniform(0.60, 0.75)
            rhythmic_units = max(2, int(round(total_units * split_ratio)))
            tail_pluta_units = total_units - rhythmic_units

            curr_accum = 0
            while curr_accum < rhythmic_units:
                rem = rhythmic_units - curr_accum
                if rem == 1:
                    m = 1
                elif rem == 2:
                    m = 2
                else:
                    m = random.choice([1, 2])

                curr_accum += m
                sym = "⏑" if m == 1 else "-"
                char = random.choice(RAW_SYLLABLE_PALETTE[m])
                decomposed.append(DecomposedNote(
                    note_name=note_name,
                    freq_hz=std_hz,
                    mātras=m,
                    prosody_symbol=sym,
                    is_rest=False,
                    raw_letter=char
                ))

            # Tail sustain Pluta
            tail_m = max(2, tail_pluta_units)
            tildes = "~" * max(1, tail_m - 2)
            pluta_sym = f"-{tildes}"
            pluta_char = random.choice(RAW_SYLLABLE_PALETTE.get(min(4, tail_m), RAW_SYLLABLE_PALETTE[4]))
            decomposed.append(DecomposedNote(
                note_name=note_name,
                freq_hz=std_hz,
                mātras=tail_m,
                prosody_symbol=pluta_sym,
                is_rest=False,
                raw_letter=pluta_char
            ))

            i += 1

        return decomposed

    def generate_stage1_raw_prototype(self, notes: List[DecomposedNote]) -> str:
        """
        Stage 1: Generates the Raw Single-Letter Prototype.
        For each note, outputs its assigned letter. Gaps are preserved as spaces.
        Example: 'ಆ~~ ಸೋ~ ಕ ತ ನ ಲಾ~'
        """
        letters = []
        for n in notes:
            if n.is_rest:
                letters.append(" ")
            else:
                letters.append(n.raw_letter)
        # Clean consecutive spaces
        raw_str = " ".join(letters).strip()
        while "   " in raw_str:
            raw_str = raw_str.replace("   ", " ")
        return raw_str

    def _get_active_lexicon(self) -> Dict:
        for k in ['mass', 'janapada', 'sad']:
            if k in self.style:
                return GENRE_VOCABULARY[k]
        return GENRE_VOCABULARY['melody']

    def generate_stage2_meaningful_lyrics(
        self,
        notes: List[DecomposedNote],
        line_num: int,
        target_rhyme_ending: Optional[str] = None
    ) -> str:
        """
        Stage 2: Translates the acoustic prosodic pattern into authentic, meaningful Kannada lyrics.
        - Partitions notes into word chunks whose syllable matras match note matras.
        - Open dynamic sampling (NO repeating fixed words).
        - If line_num is even, locks Antyaprāsa with target_rhyme_ending.
        """
        lex = self._get_active_lexicon()
        phrase_matras = [n.mātras for n in notes if not n.is_rest]
        total_p_matras = sum(phrase_matras)

        words = []
        i = 0
        n = len(phrase_matras)

        while i < n:
            rem_notes = n - i
            chosen_word = None

            # Try chunks of 3 notes, 2 notes, or 1 note
            for chunk_size in [4, 3, 2, 1]:
                if chunk_size <= rem_notes:
                    pattern = tuple(phrase_matras[i:i + chunk_size])
                    candidates = lex.get(pattern, [])

                    # Filter out words already used in previous lines of this song to guarantee variety
                    fresh_candidates = [c for c in candidates if c not in self.used_words_memory]
                    pool = fresh_candidates if fresh_candidates else candidates

                    if pool:
                        # If this chunk reaches the end of the line, prioritize Antyaprāsa!
                        if (i + chunk_size == n) and (line_num % 2 == 0) and target_rhyme_ending:
                            c1 = extract_primary_consonant(target_rhyme_ending)
                            v1 = extract_vowel(target_rhyme_ending)

                            def rhyme_fitness(candidate: str) -> float:
                                c_syls = self.prosody_engine.syllabify_word(candidate)
                                if not c_syls:
                                    return 0.0
                                last_s = c_syls[-1]
                                c2 = extract_primary_consonant(last_s)
                                v2 = extract_vowel(last_s)
                                c_sc, _ = calculate_consonant_phonic_score(c1, c2)
                                v_sc, _ = check_swara_similarity(v1, v2)
                                return (c_sc / 20.0) * 0.70 + v_sc * 0.30

                            scored_cands = [(w, rhyme_fitness(w)) for w in pool]
                            scored_cands.sort(key=lambda x: x[1], reverse=True)
                            if scored_cands and scored_cands[0][1] >= 0.70:
                                top_rhymes = [w for w, sc in scored_cands if sc >= 0.70]
                                chosen_word = random.choice(top_rhymes)
                                i += chunk_size
                                break

                        chosen_word = random.choice(pool)
                        i += chunk_size
                        break

            if chosen_word:
                words.append(chosen_word)
                self.used_words_memory.add(chosen_word)
            else:
                # Pluta sustain or generic vowel extension
                m = phrase_matras[i]
                if m >= 3:
                    ext = "~" * (m - 2)
                    chosen_word = f"ಹಾ{ext}"
                elif m == 2:
                    chosen_word = "ನಾ"
                else:
                    chosen_word = "ಗೆ"
                words.append(chosen_word)
                i += 1

        return " ".join(words)

    def process_melody_phrase(
        self,
        phrase_id: int,
        raw_events: List[NoteEvent],
        preceding_lyrics: Optional[str] = None
    ) -> StageResults:
        """
        End-to-End Processing for one Phrase:
        1. Subdivide / Decompose notes.
        2. Stage 1: Generate Raw Note-to-Letter Prototype.
        3. Stage 2: Transform into Meaningful Lyrical Sentence.
        4. Audit matras, meaning, and Antyaprāsa.
        """
        # Step 1: Note decomposition
        notes = self.snap_and_decompose_notes(raw_events)

        # Step 2: Stage 1 Raw Prototype
        stage1_text = self.generate_stage1_raw_prototype(notes)
        voiced_notes = [n for n in notes if not n.is_rest]
        total_matras = sum(n.mātras for n in voiced_notes)

        # Step 3: Stage 2 Meaningful Lyrics
        target_rhyme_syl = None
        if preceding_lyrics and (phrase_id % 2 == 0):
            p_words = preceding_lyrics.strip().split()
            if p_words:
                syls = self.prosody_engine.syllabify_word(p_words[-1])
                if syls:
                    target_rhyme_syl = syls[-1]

        stage2_text = self.generate_stage2_meaningful_lyrics(
            notes,
            line_num=phrase_id,
            target_rhyme_ending=target_rhyme_syl
        )

        # Step 4: Verification & Audit
        expected_matras = [n.mātras for n in voiced_notes]
        meter_audit = self.prosody_engine.verify_line_against_prosodic_mask(
            stage2_text,
            expected_mātras=expected_matras
        )

        sem_audit = self.semantic_engine.evaluate_line_meaning(stage2_text, style_genre=self.style)

        rhyme_str = None
        if preceding_lyrics and (phrase_id % 2 == 0):
            p_res = self.prosody_engine.check_prasa(preceding_lyrics, stage2_text)
            rhyme_str = f"Antyaprāsa: {p_res['antyaprasa_score']}/20 ({p_res['antyaprasa_reason']})"

        stage1_pairs = [(n.note_name, n.raw_letter, n.mātras) for n in voiced_notes]

        return StageResults(
            phrase_id=phrase_id,
            decomposed_notes=notes,
            stage1_raw_lyrics=stage1_text,
            stage1_note_letter_pairs=stage1_pairs,
            stage1_syllable_count=len(voiced_notes),
            stage1_total_matras=total_matras,
            stage2_meaningful_lyrics=stage2_text,
            stage2_meaning_score=sem_audit['line_score'],
            antya_rhyme_info=rhyme_str,
            is_meter_valid=meter_audit['is_valid']
        )
