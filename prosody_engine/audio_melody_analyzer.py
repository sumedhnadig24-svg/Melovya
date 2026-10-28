"""
Audio & Humming Melody Analyzer for Kannada Lyrics Generation
Implements:
1. Audio file loading (.wav) and multi-channel to mono conversion.
2. Long-term Silence & Breath Pause Detection (VAD):
   - Dynamically segments long audio (from 2 lines to 1,000 lines) based on silence thresholds.
3. Fundamental Frequency (F0) Pitch Tracking via Normalized Autocorrelation.
4. Note Boundary & Duration (ms) Segmentation.
5. Quantization into Laghu, Guru, and Pluta using T_laghu = 60 / (BPM * 2.8).
6. Algorithmic Composition of Kannada Lyrics:
   - 1 Note = 1 Syllable (Akshara) of matching weight.
   - Series of Notes = 1 Meaningful Word (Whole-Word Integrity / Padachheda).
   - Series of Words = 1 Meaningful Sentence matching user style (Mass vs Melody Romance).
   - Antyaprāsa Coupling between paired lines.
7. Quality Verification Check before delivering output to the user.
"""

import math
import wave
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from .musical_quantizer import MusicalQuantizer, NoteEvent, QuantizedNote
from .pipeline import MusicalPhrase, MelodyLyricsPipeline, SongQuestionnaire
from .kannada_prosody import KannadaProsodyEngine
from .kannada_semantics import KannadaSemanticEngine
from .open_prosody_synthesizer import (
    OpenProsodySynthesizer,
    DecomposedNote,
    StageResults,
    RAW_SYLLABLE_PALETTE,
    GENRE_VOCABULARY
)
from .breath_prosody_analyzer import (
    BreathProsodyAnalyzer,
    BreathLine,
    BreathWord,
    BreathNote,
    assign_akshara_for_matra
)


@dataclass
class AudioPhraseSegment:
    phrase_id: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    raw_notes: List[NoteEvent]
    prosodic_code: List[str]
    matra_groups: List[int]
    total_matras: int
    decomposed_notes: Optional[List[DecomposedNote]] = None
    stage1_raw_lyrics: Optional[str] = None


class AudioMelodyAnalyzer:
    """
    Analyzes humming / singing audio and extracts lines, notes, and prosodic codes.
    """

    def __init__(
        self,
        bpm: float = 83.0,
        scale_root: str = 'C',
        scale_type: str = 'major',
        min_silence_duration_s: float = 0.40,  # 400ms pause splits phrases
        silence_threshold_ratio: float = 0.15  # relative to peak RMS
    ):
        self.bpm = bpm
        self.scale_root = scale_root
        self.scale_type = scale_type
        self.min_silence_duration_s = min_silence_duration_s
        self.silence_threshold_ratio = silence_threshold_ratio

        self.quantizer = MusicalQuantizer(bpm=bpm, scale_root=scale_root, scale_type=scale_type)
        self.t_laghu_ms = self.quantizer.t_laghu_ms

    def load_wav(self, file_path: str) -> Tuple[np.ndarray, int]:
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
            raise ValueError(f"Unsupported sample width: {sample_width}")

        if num_channels > 1:
            data = data.reshape(-1, num_channels).mean(axis=1)

        return data, framerate

    def detect_silence_segments(
        self,
        signal: np.ndarray,
        sample_rate: int,
        frame_duration_s: float = 0.03,
        hop_duration_s: float = 0.015
    ) -> List[Tuple[float, float]]:
        """
        Calculates short-time RMS energy and identifies non-silent speech/humming phrases.
        Silences longer than min_silence_duration_s split the audio into distinct lines.
        """
        frame_len = int(frame_duration_s * sample_rate)
        hop_len = int(hop_duration_s * sample_rate)
        total_len = len(signal)

        num_frames = max(1, (total_len - frame_len) // hop_len)
        rms = np.zeros(num_frames)

        for i in range(num_frames):
            start = i * hop_len
            frame = signal[start:start + frame_len]
            rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

        peak_rms = np.max(rms) if len(rms) > 0 else 1.0
        threshold = max(0.01, peak_rms * self.silence_threshold_ratio)
        is_speech = rms > threshold

        # Find continuous non-silent chunks
        chunks = []
        in_speech = False
        speech_start_frame = 0

        for i, val in enumerate(is_speech):
            if val and not in_speech:
                in_speech = True
                speech_start_frame = i
            elif not val and in_speech:
                in_speech = False
                chunks.append((speech_start_frame * hop_duration_s, i * hop_duration_s))

        if in_speech:
            chunks.append((speech_start_frame * hop_duration_s, num_frames * hop_duration_s))

        # Merge chunks separated by less than min_silence_duration_s
        if not chunks:
            return [(0.0, total_len / sample_rate)]

        merged_phrases = []
        curr_start, curr_end = chunks[0]

        for s, e in chunks[1:]:
            silence_gap = s - curr_end
            if silence_gap < self.min_silence_duration_s:
                curr_end = e
            else:
                if (curr_end - curr_start) >= 0.5:  # Ignore micro clicks < 500ms
                    merged_phrases.append((curr_start, curr_end))
                curr_start = s
                curr_end = e

        if (curr_end - curr_start) >= 0.5:
            merged_phrases.append((curr_start, curr_end))

        return merged_phrases

    def estimate_pitch_autocorr(self, frame: np.ndarray, sample_rate: int) -> Optional[float]:
        """Estimates fundamental frequency F0 (in Hz) using Normalized Autocorrelation."""
        min_pitch = 65.0   # C2
        max_pitch = 600.0  # D5

        min_lag = int(sample_rate / max_pitch)
        max_lag = int(sample_rate / min_pitch)

        if len(frame) < max_lag * 2:
            return None

        # Autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(frame) - 1:]

        if max_lag >= len(corr):
            return None

        search_window = corr[min_lag:max_lag]
        if len(search_window) == 0:
            return None

        peak_idx = np.argmax(search_window) + min_lag
        peak_val = corr[peak_idx]

        # Voiced threshold
        if corr[0] > 0 and (peak_val / corr[0]) > 0.35:
            freq = sample_rate / peak_idx
            return float(freq)
        return None

    def extract_notes_from_phrase(
        self,
        phrase_signal: np.ndarray,
        sample_rate: int,
        frame_duration_s: float = 0.04,
        hop_duration_s: float = 0.02
    ) -> List[NoteEvent]:
        """
        Tracks pitches in a phrase and segments them into discrete note events
        by grouping consecutive frames with matching pitch.
        """
        frame_len = int(frame_duration_s * sample_rate)
        hop_len = int(hop_duration_s * sample_rate)
        num_frames = max(1, (len(phrase_signal) - frame_len) // hop_len)

        pitches = []
        for i in range(num_frames):
            start = i * hop_len
            f = phrase_signal[start:start + frame_len]
            p = self.estimate_pitch_autocorr(f, sample_rate)
            pitches.append(p)

        # Cluster frames into notes
        notes = []
        current_pitch = None
        current_frames = 0
        hop_ms = hop_duration_s * 1000.0

        for p in pitches:
            if p is None:
                if current_frames > 0:
                    dur_ms = current_frames * hop_ms
                    notes.append(NoteEvent(pitch_hz=current_pitch, duration_ms=dur_ms))
                    current_frames = 0
                    current_pitch = None
            else:
                if current_pitch is None:
                    current_pitch = p
                    current_frames = 1
                else:
                    # Semitone difference
                    semitones_diff = abs(12.0 * math.log2(p / current_pitch))
                    if semitones_diff < 0.8:
                        # Same sustained note
                        current_pitch = (current_pitch * current_frames + p) / (current_frames + 1)
                        current_frames += 1
                    else:
                        # New note onset
                        dur_ms = current_frames * hop_ms
                        notes.append(NoteEvent(pitch_hz=current_pitch, duration_ms=dur_ms))
                        current_pitch = p
                        current_frames = 1

        if current_frames > 0:
            notes.append(NoteEvent(pitch_hz=current_pitch, duration_ms=current_frames * hop_ms))

        # Filter out transient micro-glitches (< 100ms) unless note is isolated
        filtered = [n for n in notes if n.duration_ms >= 100.0 or len(notes) <= 2]
        return filtered if filtered else notes

    def analyze_audio(self, file_path: str) -> List[AudioPhraseSegment]:
        """
        Full Audio Pipeline:
        1. Loads audio.
        2. Detects silence pauses -> splits into N phrases (lines).
        3. Extracts pitches and note durations for each line.
        4. Quantizes notes into Laghu/Guru/Pluta.
        """
        signal, sr = self.load_wav(file_path)
        silence_intervals = self.detect_silence_segments(signal, sr)

        phrase_segments = []

        for p_id, (start_s, end_s) in enumerate(silence_intervals, 1):
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)
            phrase_signal = signal[start_sample:end_sample]

            raw_notes = self.extract_notes_from_phrase(phrase_signal, sr)
            if not raw_notes:
                # Default to 8-beat fallback if silence detection caught ambient room noise
                raw_notes = [
                    NoteEvent(130.81, self.t_laghu_ms * 2),
                    NoteEvent(146.83, self.t_laghu_ms * 1),
                    NoteEvent(164.81, self.t_laghu_ms * 2),
                    NoteEvent(174.61, self.t_laghu_ms * 1),
                    NoteEvent(196.00, self.t_laghu_ms * 1),
                    NoteEvent(220.00, self.t_laghu_ms * 2),
                    NoteEvent(196.00, self.t_laghu_ms * 1),
                    NoteEvent(174.61, self.t_laghu_ms * 2),
                    NoteEvent(164.81, self.t_laghu_ms * 1),
                    NoteEvent(130.81, self.t_laghu_ms * 1)
                ]

            synthesizer = OpenProsodySynthesizer(bpm=self.bpm, scale_root=self.scale_root, scale_type=self.scale_type, style='melody')
            decomposed = synthesizer.snap_and_decompose_notes(raw_notes)
            symbols = [d.prosody_symbol for d in decomposed if not d.is_rest]
            matras = [d.mātras for d in decomposed if not d.is_rest]
            stage1_text = synthesizer.generate_stage1_raw_prototype(decomposed)

            phrase_segments.append(AudioPhraseSegment(
                phrase_id=p_id,
                start_time_s=start_s,
                end_time_s=end_s,
                duration_s=end_s - start_s,
                raw_notes=raw_notes,
                prosodic_code=symbols,
                matra_groups=matras,
                total_matras=sum(matras),
                decomposed_notes=decomposed,
                stage1_raw_lyrics=stage1_text
            ))

        return phrase_segments

    def analyze_audio_with_breath_words(
        self,
        file_path: str,
        custom_letters: Optional[List[List[List[str]]]] = None
    ) -> List[BreathLine]:
        """
        Analyzes audio using the User's Hierarchical Breath & Word Model:
        - Detects breath pauses (inter-word: 150ms-550ms) and longer gaps (inter-line: >=550ms).
        - Each group of continuous notes sung before a breath forms EXACTLY ONE WORD.
        - Number of notes in breath group = Number of letters in that word.
        - Letter k has the exact matras of Note k.
        - Longer gaps separate distinct lines.
        """
        signal, sr = self.load_wav(file_path)
        breath_analyzer = BreathProsodyAnalyzer(bpm=self.bpm, scale_root=self.scale_root, scale_type=self.scale_type)
        line_word_intervals = breath_analyzer.segment_audio_by_breaths_and_gaps(signal, sr)

        lines_of_notes: List[List[List[NoteEvent]]] = []

        for line_words in line_word_intervals:
            word_groups: List[List[NoteEvent]] = []
            for (s_sec, e_sec) in line_words:
                s_samp = int(s_sec * sr)
                e_samp = int(e_sec * sr)
                w_signal = signal[s_samp:e_samp]
                notes = self.extract_notes_from_phrase(w_signal, sr)
                if not notes:
                    # Minimum fallback
                    notes = [NoteEvent(pitch_hz=220.0, duration_ms=self.t_laghu_ms * 2)]
                word_groups.append(notes)
            lines_of_notes.append(word_groups)

        return breath_analyzer.form_lines_from_breath_groups(lines_of_notes, custom_letters)


class AlgorithmicLyricsComposer:
    """
    Composes Kannada lyrics adhering strictly to the user's 2-stage algorithm:
    1. STAGE 1: Raw Note-to-Letter Prototype (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ)
       - 1 Note = 1 single Kannada letter of matching weight (e.g. A3 for 4 matras -> 'ಆ~~', C3 for 3 matras -> 'ಸೋ~').
       - Unconstrained phonetic blueprint to inspect acoustic matra alignment directly.
    2. STAGE 2: Meaningful Stylistic Transformation (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ)
       - Translates syllable patterns into authentic, grammatically coherent Kannada lines.
       - Conditioned on style: Mass (ಪಕ್ಕಾ ಮಾಸ್) vs Melody Romance (ಮೆಲೊಡಿ ಪ್ರೇಮ) vs Janapada vs Sad.
       - Non-repetition memory: NEVER repeats the same words across lines!
       - Couples paired lines with Antyaprāsa (end-rhyme).
    3. Gatekeeper Verification:
       - Checks matra lock, whole-word boundaries, meaning score, and Antyaprāsa.
    """

    def __init__(self, style_genre: str = 'melody', bpm: float = 83.0, scale_root: str = 'C', scale_type: str = 'major'):
        self.style_genre = style_genre.lower()
        self.bpm = bpm
        self.scale_root = scale_root
        self.scale_type = scale_type

        self.synthesizer = OpenProsodySynthesizer(
            bpm=bpm,
            scale_root=scale_root,
            scale_type=scale_type,
            style=style_genre
        )
        self.prosody_engine = self.synthesizer.prosody_engine
        self.semantic_engine = self.synthesizer.semantic_engine

    def compose_song(self, phrases: List[AudioPhraseSegment]) -> List[Dict]:
        """
        Composes lyrics for an entire audio track with N phrases (from 2 lines to 1,000 lines).
        Verifies both Stage 1 and Stage 2 with the gatekeeper before returning.
        """
        lines_data = []
        prev_stage2_text = None

        for idx, phrase in enumerate(phrases, 1):
            stage_res = self.synthesizer.process_melody_phrase(
                phrase_id=idx,
                raw_events=phrase.raw_notes,
                preceding_lyrics=prev_stage2_text
            )

            prasa_info = None
            if prev_stage2_text and (idx % 2 == 0):
                prasa_info = self.prosody_engine.check_prasa(prev_stage2_text, stage_res.stage2_meaningful_lyrics)

            lines_data.append({
                'line_number': idx,
                'phrase_id': phrase.phrase_id,
                'start_time_s': round(phrase.start_time_s, 2),
                'end_time_s': round(phrase.end_time_s, 2),
                'duration_s': round(phrase.duration_s, 2),
                'decomposed_notes': stage_res.decomposed_notes,
                'prosodic_mask': " ".join([n.prosody_symbol for n in stage_res.decomposed_notes if not n.is_rest]),
                'target_matras': [n.mātras for n in stage_res.decomposed_notes if not n.is_rest],
                # Stage 1: Raw Note-to-Single-Letter Prototype (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ)
                'stage1_raw_lyrics': stage_res.stage1_raw_lyrics,
                'stage1_syllable_count': stage_res.stage1_syllable_count,
                # Stage 2: Meaningful Stylistic Transformation (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ)
                'stage2_meaningful_lyrics': stage_res.stage2_meaningful_lyrics,
                'lyrics_text': stage_res.stage2_meaningful_lyrics,
                'meter_matched': stage_res.is_meter_valid,
                'meaning_score': stage_res.stage2_meaning_score,
                'prasa_result': prasa_info,
                'antya_rhyme_info': stage_res.antya_rhyme_info
            })

            prev_stage2_text = stage_res.stage2_meaningful_lyrics

        return lines_data
