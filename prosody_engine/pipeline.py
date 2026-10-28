"""
Lyrics Pipeline Module
Coordinates:
1. User Questionnaire (BPM, Scale, Style, Mood)
2. Audio Note Quantization into Laghu/Guru/Pluta
3. Musical Phrase Segmentation & Prosodic Mask Generation
4. Verification of candidate lyrics with:
   - Dedicated Semantic Meaning & Vocabulary Validity (25 pts)
   - Syllable-to-Note Meter Alignment (35 pts)
   - Whole-Word Integrity / Padachheda (15 pts)
   - Varga & Swara Prāsa Rhyme Scoring (25 pts)
   - Strict Meaning Gatekeeper (Nonsense / Fabricated words are strictly capped and rejected)
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from .musical_quantizer import MusicalQuantizer, NoteEvent, QuantizedNote
from .kannada_prosody import KannadaProsodyEngine
from .kannada_semantics import KannadaSemanticEngine


@dataclass
class SongQuestionnaire:
    tempo_bpm: float
    scale_root: str
    scale_type: str
    lyrics_style: str  # e.g., 'Modern Sandalwood', 'Classical', 'Folk', 'Mass'
    mood: str          # e.g., 'Romantic', 'Celebration', 'Sad'
    theme: str         # e.g., 'Love', 'Friendship', 'Philosophy'


@dataclass
class MusicalPhrase:
    phrase_id: int
    notes: List[QuantizedNote]
    prosodic_symbols: List[str]
    mātra_groups: List[int]       # Target mātras for each beat/group in this phrase
    total_mātras: int


def classify_lyric_quality(score: float, meaning_score: float = 25.0, has_fabricated_words: bool = False) -> str:
    """
    Classifies overall lyrics into graded quality tiers per user criteria:
    - If words are fabricated / meaningless: Strictly Very Bad / Bad
    - >= 85%: Good Lyrics (ಉತ್ತಮ ಸಾಹಿತ್ಯ)
    - 80% - 84%: Good / Acceptable Lyrics (ಉತ್ತಮ ಸಾಹಿತ್ಯ)
    - 50% - 79%: Moderate Lyrics (ಮಧ್ಯಮ ಸಾಹಿತ್ಯ)
    - 11% - 49%: Bad Lyrics (ಕಳಪೆ ಸಾಹಿತ್ಯ)
    - 0% - 10%: Very Bad Lyrics (ತೀರಾ ಕಳಪೆ ಸಾಹಿತ್ಯ)
    """
    if has_fabricated_words or meaning_score == 0.0:
        return 'Very Bad Lyrics - Meaningless / Fabricated Words (ತೀರಾ ಕಳಪೆ - ಅರ್ಥವಿಲ್ಲದ ಸಾಹಿತ್ಯ) 🚫'
    if meaning_score < 12.5:
        return 'Bad Lyrics - Weak / Incoherent Meaning (ಕಳಪೆ - ಅಸ್ಪಷ್ಟ ಅರ್ಥ) ❌'

    if score >= 85.0:
        return 'Good Lyrics (ಉತ್ತಮ ಸಾಹಿತ್ಯ) ✅'
    elif score >= 80.0:
        return 'Good / Acceptable Lyrics (ಉತ್ತಮ ಸಾಹಿತ್ಯ) ✅'
    elif score >= 50.0:
        return 'Moderate Lyrics (ಮಧ್ಯಮ ಸಾಹಿತ್ಯ) ⚠️'
    elif score > 10.0:
        return 'Bad Lyrics (ಕಳಪೆ ಸಾಹಿತ್ಯ) ❌'
    else:
        return 'Very Bad Lyrics (ತೀರಾ ಕಳಪೆ ಸಾಹಿತ್ಯ) 🚫'


class MelodyLyricsPipeline:
    def __init__(self, questionnaire: SongQuestionnaire):
        self.questionnaire = questionnaire
        self.quantizer = MusicalQuantizer(
            bpm=questionnaire.tempo_bpm,
            scale_root=questionnaire.scale_root,
            scale_type=questionnaire.scale_type
        )
        self.prosody_engine = KannadaProsodyEngine()
        self.semantic_engine = KannadaSemanticEngine()

    def process_melody(self, raw_events: List[NoteEvent]) -> List[MusicalPhrase]:
        """
        Processes raw audio/MIDI note events:
        1. Snaps frequencies with trend logic & scale awareness.
        2. Applies the universal formula T_laghu = 60 / (BPM * 2.8).
        3. Segments into musical phrases based on RESTs (breaths).
        4. Produces the exact Prosodic String Mask for each phrase.
        """
        quantized_notes = self.quantizer.quantize_melody(raw_events)

        phrases = []
        current_phrase_notes = []
        phrase_id = 1

        for note in quantized_notes:
            if note.note_name == "REST":
                # Breath / pause marks the end of a musical phrase
                if current_phrase_notes:
                    phrases.append(self._create_phrase(phrase_id, current_phrase_notes))
                    phrase_id += 1
                    current_phrase_notes = []
            else:
                current_phrase_notes.append(note)

        if current_phrase_notes:
            phrases.append(self._create_phrase(phrase_id, current_phrase_notes))

        return phrases

    def _create_phrase(self, phrase_id: int, notes: List[QuantizedNote]) -> MusicalPhrase:
        symbols = [n.prosody_symbol for n in notes]
        mātras = [n.mātras for n in notes]
        return MusicalPhrase(
            phrase_id=phrase_id,
            notes=notes,
            prosodic_symbols=symbols,
            mātra_groups=mātras,
            total_mātras=sum(mātras)
        )

    def verify_lyrics(self, lyrics_lines: List[str], phrases: List[MusicalPhrase]) -> Dict:
        """
        Verifies candidate lyrics across 4 core pillars (Total 100 points):
        1. Semantic Meaning & Vocabulary Validity (25 pts)
        2. Proportional Syllable-to-Note Meter Alignment (35 pts)
        3. Whole-Word Integrity / Padachheda (15 pts)
        4. Varga & Swara Prāsa Rhyme Scoring (25 pts)
        Strict Meaning Gatekeeper: Meaningless / fabricated words cannot be rated Good!
        """
        line_results = []
        num_checks = min(len(lyrics_lines), len(phrases))

        for i in range(num_checks):
            line_text = lyrics_lines[i]
            phrase = phrases[i]
            res = self.prosody_engine.verify_line_against_prosodic_mask(
                line_text,
                expected_mātras=phrase.mātra_groups
            )
            line_results.append({
                'phrase_id': phrase.phrase_id,
                'line_text': line_text,
                'target_prosody': phrase.prosodic_symbols,
                'verification': res
            })

        # 1. Semantic Meaning & Vocabulary Evaluation (up to 25 pts)
        l1 = lyrics_lines[0] if len(lyrics_lines) > 0 else ""
        l2 = lyrics_lines[1] if len(lyrics_lines) > 1 else ""
        semantic_eval = self.semantic_engine.evaluate_couplet_meaning(
            l1, l2, style_genre=self.questionnaire.lyrics_style
        )
        meaning_score = semantic_eval['meaning_score']

        # 2. Syllable-to-Note Meter Alignment (up to 35 pts, 17.5 per line)
        meter_score_total = round(sum(r['verification']['meter_score'] for r in line_results), 1)

        # 3. Whole-Word Integrity (up to 15 pts, 7.5 per line)
        word_score_total = round(sum(r['verification']['word_boundary_score'] for r in line_results), 1)

        # 4. Varga & Swara Prāsa (up to 25 pts: Antyaprāsa prioritized as primary modern song rhyme)
        prasa_checks = []
        prasa_score_total = 0.0
        for i in range(0, num_checks - 1, 2):
            p_res = self.prosody_engine.check_prasa(lyrics_lines[i], lyrics_lines[i + 1])
            antya_s = p_res['antyaprasa_score']
            adi_s = p_res['adiprasa_score']

            # Modern Song Rhyme Rule: Antyaprāsa is primary (up to 22.5 pts).
            # Ādiprāsa is an optional classical bonus (+2.5 pts), NOT a prerequisite!
            if antya_s >= 12:
                # Antyaprāsa matches! Can achieve up to 25 pts even if Ādiprāsa is 0
                base_antya = (antya_s / 20.0) * 22.5
                adi_bonus = (adi_s / 20.0) * 2.5
                scaled_prasa = min(25.0, round(base_antya + adi_bonus, 1))
            elif adi_s >= 12:
                # Classical verse style: Only Ādiprāsa matches
                scaled_prasa = round((adi_s / 20.0) * 15.0, 1)
            else:
                # Weak or disjoint rhymes
                scaled_prasa = round(((antya_s * 0.75 + adi_s * 0.25) / 20.0) * 8.0, 1)

            prasa_score_total += scaled_prasa
            prasa_checks.append({
                'lines': (i + 1, i + 2),
                'prasa_result': p_res,
                'scaled_prasa_score': scaled_prasa
            })

        # Calculate composite score out of 100 points
        overall_score = round(meaning_score + meter_score_total + word_score_total + prasa_score_total, 1)

        # STRICT MEANING GATEKEEPER:
        # If words are fabricated / have no meaning, meaning score is penalized
        # and lyrics CANNOT be rated Good or Moderate!
        has_fabricated = semantic_eval['has_fabricated_words']
        if has_fabricated or meaning_score < 12.5:
            overall_score = min(overall_score, 35.0)
            if meaning_score == 0.0 or has_fabricated:
                overall_score = min(overall_score, 15.0)

        overall_score = max(0.0, min(100.0, overall_score))
        quality_rating = classify_lyric_quality(
            overall_score,
            meaning_score=meaning_score,
            has_fabricated_words=has_fabricated
        )

        all_lines_valid = all(
            r['verification']['is_valid']
            for r in line_results
        )
        all_prasa_valid = all(
            p['prasa_result']['adiprasa'] and p['prasa_result']['antyaprasa']
            for p in prasa_checks
        )

        return {
            'overall_score': overall_score,
            'quality_rating': quality_rating,
            'meaning_score': meaning_score,
            'meter_score_total': meter_score_total,
            'word_score_total': word_score_total,
            'prasa_score_total': prasa_score_total,
            'semantic_eval': semantic_eval,
            'all_lines_valid': all_lines_valid,
            'all_prasa_valid': all_prasa_valid,
            'line_results': line_results,
            'prasa_checks': prasa_checks
        }
