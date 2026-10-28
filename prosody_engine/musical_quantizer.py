"""
Musical Quantizer Module
Implements:
1. Universal Tempo Formula: T_laghu = 60 / (BPM * 4) (seconds)
2. Note duration to Laghu / Guru / Pluta mapping
3. Frequency-to-MIDI pitch snapping with trend-based tie breaking and scale awareness
"""

import math
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# Standard note names in an octave
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Common musical scales (semitone intervals from root)
SCALES: Dict[str, List[int]] = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
    'mohana': [0, 2, 4, 7, 9],            # Mohana / Bhoop (Pentatonic Major)
    'kalyani': [0, 2, 4, 6, 7, 9, 11],    # Lydian
    'shankarabharana': [0, 2, 4, 5, 7, 9, 11] # Major
}


@dataclass
class NoteEvent:
    pitch_hz: Optional[float]  # None or <=0 indicates REST / Silence
    duration_ms: float


@dataclass
class QuantizedNote:
    note_name: str             # e.g., 'C3', 'C#3', or 'REST'
    midi_note: Optional[int]
    frequency_hz: Optional[float]
    duration_ms: float
    units: int                 # Number of atomic units (Laghu = 1)
    prosody_symbol: str        # '⏑' (Laghu), '-' (Guru), '-~' (Pluta), or 'REST'
    mātras: int


class MusicalQuantizer:
    def __init__(self, bpm: float = 120.0, scale_root: str = 'C', scale_type: str = 'major', time_signature: str = '6/8'):
        self.bpm = bpm
        self.scale_root = scale_root.upper()
        self.scale_type = scale_type.lower()
        self.time_signature = time_signature
        try:
            self.ts_num, self.ts_den = map(int, time_signature.strip().split('/'))
        except Exception:
            self.ts_num, self.ts_den = 6, 8
        self.scale_pitch_classes = self._get_scale_pitch_classes()

    @property
    def t_laghu_sec(self) -> float:
        """Base formula: T_laghu = (60 / (BPM * 2.8)) * (8 / Denominator)"""
        return (60.0 / (self.bpm * 2.8)) * (8.0 / self.ts_den)

    @property
    def t_laghu_ms(self) -> float:
        """Atomic time unit in milliseconds"""
        return self.t_laghu_sec * 1000.0

    @property
    def t_matra_sec(self) -> float:
        return self.t_laghu_sec

    @property
    def t_matra_ms(self) -> float:
        return self.t_laghu_ms

    @property
    def bar_duration_ms(self) -> float:
        """Bar / Cycle duration in milliseconds: N * T_matra"""
        return self.ts_num * self.t_matra_ms

    @staticmethod
    def snap_ratio_to_matras(ratio: float, threshold: float = 0.9) -> int:
        """
        User's >0.9 Threshold Mātra Snapping Rule:
        - If ratio is 1.8 -> 1 mātra (stays at floor until fractional part goes above 0.9)
        - If ratio is 2.8 -> 2 mātras (stays at 2 mātras until > 2.9)
        - Only when ratio strictly exceeds 1.9 does it become 2 mātras.
        - Only when ratio strictly exceeds 2.9 does it become 3 mātras.
        - Applies to all mātras from 1 to infinity.
        """
        if ratio <= 1.0:
            return 1
        base = int(ratio)
        frac = round(ratio - base, 4)
        if frac > threshold:
            return base + 1
        return base

    @staticmethod
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

    def snap_duration_to_matras(self, duration_ms: float) -> Tuple[int, float]:
        """
        Calculates the mātra value for a given duration in milliseconds:
        ratio = duration_ms / t_matra_ms
        matras = snap_ratio_to_matras(ratio, threshold=0.9)
        """
        ratio = duration_ms / self.t_matra_ms
        matras, symbol, base_type, tildes = self.get_prosodic_representation(ratio, threshold=0.9)
        return matras, ratio

    def _get_scale_pitch_classes(self) -> List[int]:
        root_idx = NOTE_NAMES.index(self.scale_root) if self.scale_root in NOTE_NAMES else 0
        intervals = SCALES.get(self.scale_type, SCALES['major'])
        return [(root_idx + interval) % 12 for interval in intervals]

    @staticmethod
    def hz_to_midi(hz: float) -> float:
        """Convert frequency in Hz to fractional MIDI note number."""
        if hz <= 0:
            return 0.0
        return 69.0 + 12.0 * math.log2(hz / 440.0)

    @staticmethod
    def midi_to_hz(midi: float) -> float:
        """Convert MIDI note number to frequency in Hz."""
        return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))

    @staticmethod
    def midi_to_note_name(midi: int) -> str:
        """Convert integer MIDI note number to note name with octave (e.g., 60 -> C4)."""
        octave = (midi // 12) - 1
        note = NOTE_NAMES[midi % 12]
        return f"{note}{octave}"

    def snap_pitch(self, hz: float, prev_hz: Optional[float], next_hz: Optional[float]) -> Tuple[int, str, float]:
        """
        Snaps an arbitrary frequency to the nearest standard musical note.
        Handles ambiguous frequencies (in between two notes):
        1. If trending higher (preceding/following notes go up) -> snap to higher note.
        2. If trending lower (preceding/following notes go down) -> snap to lower note.
        3. If it is the first note and exactly in the middle -> snap to nearest lower note.
        4. Prioritizes notes present in the user's selected scale.
        """
        midi_float = self.hz_to_midi(hz)
        lower_midi = math.floor(midi_float)
        upper_midi = math.ceil(midi_float)
        fraction = midi_float - lower_midi

        # Check if right in the middle (ambiguous tie, within 0.15 of half-step)
        is_ambiguous = abs(fraction - 0.5) <= 0.15

        if is_ambiguous:
            # Check scale priority first
            lower_in_scale = (lower_midi % 12) in self.scale_pitch_classes
            upper_in_scale = (upper_midi % 12) in self.scale_pitch_classes

            if lower_in_scale and not upper_in_scale:
                snapped_midi = lower_midi
            elif upper_in_scale and not lower_in_scale:
                snapped_midi = upper_midi
            else:
                # Use contour/trend rule
                trend = 0.0
                if next_hz is not None:
                    trend += (next_hz - hz)
                if prev_hz is not None:
                    trend += (hz - prev_hz)

                if trend > 0:
                    snapped_midi = upper_midi  # Trending higher -> higher note
                elif trend < 0:
                    snapped_midi = lower_midi  # Trending lower -> lower note
                else:
                    # Starting note or static context -> lower note
                    snapped_midi = lower_midi
        else:
            snapped_midi = round(midi_float)

        note_name = self.midi_to_note_name(snapped_midi)
        standard_hz = self.midi_to_hz(snapped_midi)
        return snapped_midi, note_name, standard_hz

    def duration_to_prosody(self, duration_ms: float, is_rest: bool = False) -> Tuple[int, str, int]:
        """
        Converts duration in milliseconds to integer atomic units and prosody symbol
        using User's dynamic Laghu / Guru Pluta representation.
        """
        ratio = duration_ms / self.t_laghu_ms
        matras, symbol, base_type, tildes = self.get_prosodic_representation(ratio)

        if is_rest:
            return matras, f"REST({matras})", 0
        return matras, symbol, matras

    def quantize_melody(self, events: List[NoteEvent]) -> List[QuantizedNote]:
        """
        Takes a sequence of raw NoteEvents (pitch and duration) and produces
        a sequence of QuantizedNotes with snapped pitches and Laghu/Guru/Pluta labels.
        """
        quantized = []
        n = len(events)

        for i, ev in enumerate(events):
            is_rest = (ev.pitch_hz is None or ev.pitch_hz <= 0)
            units, symbol, mātras = self.duration_to_prosody(ev.duration_ms, is_rest=is_rest)

            if is_rest:
                quantized.append(QuantizedNote(
                    note_name="REST",
                    midi_note=None,
                    frequency_hz=None,
                    duration_ms=ev.duration_ms,
                    units=units,
                    prosody_symbol=symbol,
                    mātras=0
                ))
            else:
                prev_hz = events[i - 1].pitch_hz if i > 0 else None
                next_hz = events[i + 1].pitch_hz if i < n - 1 else None
                midi_note, note_name, standard_hz = self.snap_pitch(ev.pitch_hz, prev_hz, next_hz)

                quantized.append(QuantizedNote(
                    note_name=note_name,
                    midi_note=midi_note,
                    frequency_hz=round(standard_hz, 2),
                    duration_ms=ev.duration_ms,
                    units=units,
                    prosody_symbol=symbol,
                    mātras=mātras
                ))

        return quantized
