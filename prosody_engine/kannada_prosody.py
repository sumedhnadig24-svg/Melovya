"""
Kannada Prosody & Syllable Verification Engine
Implements:
1. Kannada syllabification
2. Classical Chandassu rules (Laghu, Guru, and Pluta)
3. Word integrity / no-chopping verifier (Padachheda)
4. Varga-based Phonic Rhyme Scoring for Ādiprāsa and Antyaprāsa
   (Consonant Vargas, Acoustic Cousins, Distance Reductions, Graded 0-20 scores)
"""

import re
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# Unicode definitions for Kannada
KANNADA_SHORT_INDEPENDENT = set('ಅಇಉಋಎಒ')
KANNADA_LONG_INDEPENDENT = set('ಆಈಊೠಏಐಓಔ')

KANNADA_SHORT_MATRAS = set('\u0CBF\u0CC1\u0CC3\u0CC6\u0CCA') # ಿ, ು, ೃ, ೆ, ೊ
KANNADA_LONG_MATRAS = set('\u0CBE\u0CC0\u0CC2\u0CC4\u0CC7\u0CC8\u0CCB\u0CCC') # ಾ, ೀ, ೂ, ೄ, ೇ, ೈ, ೋ, ೌ

KANNADA_VIRAMA = '\u0CCD'    # ್ (halant)
KANNADA_ANUSVARA = '\u0C82'  # ಂ
KANNADA_VISARGA = '\u0C83'   # ಃ

KANNADA_CONSONANTS = set(
    'ಕಖಗಘಙ'
    'ಚಛಜಝಞ'
    'ಟಠಡಢಣ'
    'ತಥದಧನ'
    'ಪಫಬಭಮ'
    'ಯರಲವಶಷಸಹಳಱಳ'
)

# Kannada Consonant Vargas (ವರ್ಗೀಯ ವ್ಯಂಜನಗಳು)
KANNADA_VARGAS = {
    'ka': ['ಕ', 'ಖ', 'ಗ', 'ಘ', 'ಙ'],
    'cha': ['ಚ', 'ಛ', 'ಜ', 'ಝ', 'ಞ'],
    'ta_retro': ['ಟ', 'ಠ', 'ಡ', 'ಢ', 'ಣ'],
    'ta_dental': ['ತ', 'ಥ', 'ದ', 'ಧ', 'ನ'],
    'pa': ['ಪ', 'ಫ', 'ಬ', 'ಭ', 'ಮ'],
}

AVARGIYA_GROUPS = {
    'sibilants': ['ಶ', 'ಷ', 'ಸ'],
    'liquids': ['ಲ', 'ಳ', 'ರ', 'ಱ'],
    'approximants': ['ಯ', 'ವ', 'ಹ'],
}

# Cross-varga acoustic cousins (ಸಾದೃಶ್ಯ ವ್ಯಂಜನಗಳು) in Kannada poetry / lyric writing:
# e.g., ಣ and ನ (retroflex and dental nasal), ಡ and ದ, ಟ and ತ
ACOUSTIC_COUSINS = {
    ('ಣ', 'ನ'): 14, ('ನ', 'ಣ'): 14,
    ('ಡ', 'ದ'): 14, ('ದ', 'ಡ'): 14,
    ('ಟ', 'ತ'): 14, ('ತ', 'ಟ'): 14,
    ('ಠ', 'ಥ'): 13, ('ಥ', 'ಠ'): 13,
    ('ಢ', 'ಧ'): 13, ('ಧ', 'ಢ'): 13,
    ('ಲ', 'ಳ'): 14, ('ಳ', 'ಲ'): 14,
    ('ಶ', 'ಷ'): 16, ('ಷ', 'ಶ'): 16,
    ('ಶ', 'ಸ'): 14, ('ಸ', 'ಶ'): 14,
    ('ಷ', 'ಸ'): 14, ('ಸ', 'ಷ'): 14,
    ('ರ', 'ಱ'): 17, ('ಱ', 'ರ'): 17,
}

# Swara (vowel) acoustic similarities (ಸ್ವರ ಸಾದೃಶ್ಯ ನಿಯಮಾವಳಿ)
SWARA_ACOUSTIC_SIMILARITY = {
    # Short & Long Savarna Swaras (ಸವರ್ಣ ಸ್ವರಗಳು)
    ('\u0CC6', '\u0CC7'): 0.85, ('\u0CC7', '\u0CC6'): 0.85,  # ೆ ↔ ೇ
    ('\u0CBF', '\u0CC0'): 0.85, ('\u0CC0', '\u0CBF'): 0.85,  # ಿ ↔ ೀ
    ('\u0CC1', '\u0CC2'): 0.85, ('\u0CC2', '\u0CC1'): 0.85,  # ು ↔ ೂ
    ('\u0CCA', '\u0CCB'): 0.85, ('\u0CCB', '\u0CCA'): 0.85,  # ೊ ↔ ೋ
    ('ಅ', '\u0CBE'): 0.85, ('\u0CBE', 'ಅ'): 0.85,            # ಅ ↔ ಾ
    # Diphthong affinities (ಸಂಧ್ಯಾಕ್ಷರಗಳು)
    ('\u0CC7', '\u0CC8'): 0.70, ('\u0CC8', '\u0CC7'): 0.70,  # ೇ ↔ ೈ
    ('\u0CCB', '\u0CCC'): 0.70, ('\u0CCC', '\u0CCB'): 0.70,  # ೋ ↔ ೌ
}


def check_swara_similarity(v1: str, v2: str) -> Tuple[float, str]:
    """
    Checks if two swaras (vowels / matras) sound similar:
    - Identical swara: 1.0 (Full match)
    - Acoustic swara cousins (Short/Long pairs or Diphthongs): 0.70 - 0.85
    - Dissimilar swaras (e.g. ೇ vs ೌ): 0.0 (strictly fails rhyme!)
    """
    if v1 == v2:
        return 1.0, f'Exact swara match ({v1})'
    if (v1, v2) in SWARA_ACOUSTIC_SIMILARITY:
        return SWARA_ACOUSTIC_SIMILARITY[(v1, v2)], f'Acoustic swara cousin ({v1} ↔ {v2})'
    return 0.0, f'Dissimilar swaras ({v1} vs {v2}) - No vowel rhyme'


@dataclass
class Syllable:
    text: str                  # Kannada syllable (e.g. 'ನಿ', 'ನ್ನ', 'ಕಂ', 'ಬಾ~')
    base_vowel_type: str       # 'SHORT', 'LONG'
    has_anusvara_visarga: bool
    is_followed_by_conjunct: bool
    tildes_count: int          # Pluta extension count
    weight: str                # 'LAGHU' (⏑), 'GURU' (-), 'PLUTA' (-~)
    mātras: int


def extract_consonants(syl: str) -> List[str]:
    """Extracts all consonants present in a syllable string."""
    return [c for c in syl if c in KANNADA_CONSONANTS]


def extract_primary_consonant(syl: str) -> str:
    """Extracts the core pronounced consonant of a syllable."""
    cs = extract_consonants(syl)
    if not cs:
        return ''
    return cs[-1]


def extract_vowel(syl: str) -> str:
    """Extracts the vowel sound/matra of a syllable."""
    clean = syl.replace('~', '')
    for m in KANNADA_LONG_MATRAS:
        if m in clean:
            return m
    for m in KANNADA_SHORT_MATRAS:
        if m in clean:
            return m
    if KANNADA_ANUSVARA in clean:
        return KANNADA_ANUSVARA
    if KANNADA_VISARGA in clean:
        return KANNADA_VISARGA
    for ind in KANNADA_LONG_INDEPENDENT:
        if ind in clean:
            return ind
    for ind in KANNADA_SHORT_INDEPENDENT:
        if ind in clean:
            return ind
    return 'ಅ'  # Inherent short 'a' vowel


def get_consonant_varga(c: str) -> Tuple[Optional[str], int]:
    """Finds which Varga (ka, cha, ta_retro, ta_dental, pa) the consonant belongs to."""
    for v_name, members in KANNADA_VARGAS.items():
        if c in members:
            return v_name, members.index(c)
    return None, -1


def calculate_consonant_phonic_score(c1: str, c2: str) -> Tuple[int, str]:
    """
    Computes phonic rhyming score (0 to 20) between two consonants based on Kannada Varga rules:
    - Same consonant: 20/20 (Full score)
    - Same Varga, 1 letter aside: 17/20 (-3 marks)
    - Same Varga, 2 or 3 letters aside: 15/20 (-5 marks)
    - Same Varga, 4 letters aside: 13/20 (-7 marks)
    - Acoustic cousins (e.g. ಣ ↔ ನ, ಡ ↔ ದ, ಟ ↔ ತ): 13-14/20
    - Disjoint Vargas (e.g. ಕ vs ಪ): 0/20
    """
    if not c1 or not c2:
        return 0, 'Missing consonant'
    if c1 == c2:
        return 20, f'Exact consonant match ({c1})'

    v1, idx1 = get_consonant_varga(c1)
    v2, idx2 = get_consonant_varga(c2)

    # If both are in the same Varga
    if v1 and v2 and v1 == v2:
        dist = abs(idx1 - idx2)
        if dist == 1:
            return 17, f'Same varga ({v1}), 1 letter aside (-3 marks)'
        elif dist in (2, 3):
            return 15, f'Same varga ({v1}), {dist} letters aside (-5 marks)'
        else:
            return 13, f'Same varga ({v1}), 4 letters aside (-7 marks)'

    # Check acoustic cousin table (e.g. ಣ ↔ ನ, ಡ ↔ ದ, ಟ ↔ ತ)
    if (c1, c2) in ACOUSTIC_COUSINS:
        score = ACOUSTIC_COUSINS[(c1, c2)]
        return score, f'Acoustic cousin rhyme ({c1} ↔ {c2})'

    # Check Avargiya groups (sibilants, liquids)
    for grp_name, members in AVARGIYA_GROUPS.items():
        if c1 in members and c2 in members:
            dist = abs(members.index(c1) - members.index(c2))
            score = 17 if dist == 1 else 14
            return score, f'Avargiya group ({grp_name}) proximity'

    # Consonants belong to disjoint, unrelated groups (e.g. ಕ vs ಪ)
    return 0, f'Disjoint vargas ({c1} vs {c2}) - No rhyme (0/20)'


class KannadaProsodyEngine:
    def __init__(self):
        pass

    def syllabify_word(self, word: str) -> List[str]:
        """
        Splits a single Kannada word into its orthographic syllables (ಅಕ್ಷರಗಳು).
        Respects conjunct consonants (ಸಂಯುಕ್ತಾಕ್ಷರಗಳು), matras, and Pluta tildes (~).
        """
        syllables = []
        i = 0
        n = len(word)

        while i < n:
            char = word[i]

            # Skip spaces or punctuation
            if char in ' \t\n\r|!?,.-':
                i += 1
                continue

            current_syl = char
            i += 1

            # Consume any attached virama + consonants (conjunct clusters)
            while i < n and word[i] == KANNADA_VIRAMA:
                current_syl += word[i]
                i += 1
                if i < n and (word[i] in KANNADA_CONSONANTS or word[i] in 'zwj\u200c\u200d'):
                    current_syl += word[i]
                    i += 1

            # Consume attached matras (vowel signs)
            while i < n and (word[i] in KANNADA_SHORT_MATRAS or word[i] in KANNADA_LONG_MATRAS):
                current_syl += word[i]
                i += 1

            # Consume anusvara or visarga
            while i < n and (word[i] == KANNADA_ANUSVARA or word[i] == KANNADA_VISARGA):
                current_syl += word[i]
                i += 1

            # Consume trailing virama if at the end of word or prefix (e.g., ಕ್, ಲ್)
            while i < n and word[i] == KANNADA_VIRAMA:
                if i + 1 >= n or word[i + 1] not in KANNADA_CONSONANTS:
                    current_syl += word[i]
                    i += 1
                else:
                    break

            # Consume Pluta tildes (~)
            while i < n and word[i] == '~':
                current_syl += word[i]
                i += 1

            syllables.append(current_syl)

        return syllables

    def scan_syllables(self, words: List[str]) -> List[Syllable]:
        """
        Scans a sequence of Kannada words across phrase boundaries.
        Applies rules of Laghu, Guru, and Pluta:
        1. Long vowel -> Guru (2)
        2. Syllable with Anusvara (ಂ) or Visarga (ಃ) -> Guru (2)
        3. Short vowel followed by a conjunct (ಸಂಯುಕ್ತಾಕ್ಷರ) -> Guru (2)
        4. Short vowel alone -> Laghu (1)
        5. Pluta tildes (~) add +1 mātrā each to the syllable
        """
        all_syllables_raw = []
        word_mapping = []

        for w_idx, w in enumerate(words):
            syls = self.syllabify_word(w)
            for s in syls:
                all_syllables_raw.append(s)
                word_mapping.append(w_idx)

        scanned = []
        num_syls = len(all_syllables_raw)

        for i, s_text in enumerate(all_syllables_raw):
            tildes = s_text.count('~')
            clean_text = s_text.replace('~', '')

            has_anusvara_visarga = (KANNADA_ANUSVARA in clean_text or KANNADA_VISARGA in clean_text)
            has_long_vowel = (
                any(c in KANNADA_LONG_INDEPENDENT for c in clean_text) or
                any(c in KANNADA_LONG_MATRAS for c in clean_text)
            )

            # Check if followed by conjunct in the next syllable
            followed_by_conjunct = False
            if i + 1 < num_syls:
                next_s = all_syllables_raw[i + 1].replace('~', '')
                if KANNADA_VIRAMA in next_s and any(c in KANNADA_CONSONANTS for c in next_s):
                    followed_by_conjunct = True

            # Determine base weight
            if has_long_vowel or has_anusvara_visarga or followed_by_conjunct:
                base_mātras = 2
                base_weight = 'GURU'
            else:
                base_mātras = 1
                base_weight = 'LAGHU'

            # Apply Pluta extensions (if any)
            total_mātras = base_mātras + tildes
            if total_mātras > 2:
                final_weight = 'PLUTA'
            elif total_mātras == 2:
                final_weight = 'GURU'
            else:
                final_weight = 'LAGHU'

            scanned.append(Syllable(
                text=s_text,
                base_vowel_type='LONG' if has_long_vowel else 'SHORT',
                has_anusvara_visarga=has_anusvara_visarga,
                is_followed_by_conjunct=followed_by_conjunct,
                tildes_count=tildes,
                weight=final_weight,
                mātras=total_mātras
            ))

        return scanned

    def check_prasa(self, line1: str, line2: str) -> Dict:
        """
        Validates Ādiprāsa and Antyaprāsa using Varga phonic proximity and graduated scoring.
        Scores each out of 20 marks:
        - 20: Perfect exact match
        - 17: Same varga, 1 letter aside (-3)
        - 15: Same varga, 2-3 letters aside (-5)
        - 13-14: Acoustic cousins (e.g. ಣ ↔ ನ, ಡ ↔ ದ, ಟ ↔ ತ as in ಚಿನ್ನ ↔ ಬೆಣ್ಣೆ)
        - 0: Disjoint vargas (e.g. ಕ vs ಪ)
        """
        words1 = [w for w in line1.strip().split() if w not in ('|', '||')]
        words2 = [w for w in line2.strip().split() if w not in ('|', '||')]

        if not words1 or not words2:
            return {
                'adiprasa': False, 'antyaprasa': False,
                'adiprasa_score': 0, 'antyaprasa_score': 0,
                'total_prasa_score': 0,
                'adiprasa_reason': 'Missing words', 'antyaprasa_reason': 'Missing words',
                'adiprasa_pair': ('', ''), 'antyaprasa_pair': ('', '')
            }

        first_word1 = words1[0].replace('~', '')
        first_word2 = words2[0].replace('~', '')
        last_word1 = words1[-1].replace('~', '')
        last_word2 = words2[-1].replace('~', '')

        # --- 1. ĀDIPRĀSA (Initial Rhyme) ---
        # Prioritize 2nd syllable (Classical Dviteeyakshara Prāsa)
        syls1 = self.syllabify_word(first_word1)
        syls2 = self.syllabify_word(first_word2)

        adiprasa_score = 0
        adiprasa_reason = 'No match'

        if len(syls1) >= 2 and len(syls2) >= 2:
            c2_1 = extract_primary_consonant(syls1[1])
            c2_2 = extract_primary_consonant(syls2[1])
            d_score, d_reason = calculate_consonant_phonic_score(c2_1, c2_2)

            # Check preceding vowel length (Laghu vs Guru)
            v1_1 = extract_vowel(syls1[0])
            v1_2 = extract_vowel(syls2[0])
            len1 = 2 if (v1_1 in KANNADA_LONG_MATRAS or v1_1 in KANNADA_LONG_INDEPENDENT) else 1
            len2 = 2 if (v1_2 in KANNADA_LONG_MATRAS or v1_2 in KANNADA_LONG_INDEPENDENT) else 1
            if len1 != len2 and d_score > 0:
                d_score = max(0, d_score - 2)
                d_reason += ' (preceding vowel length variance -2)'

            adiprasa_score = d_score
            adiprasa_reason = f'Dviteeyakshara: {d_reason} ({c2_1} ↔ {c2_2})'
        else:
            # Fallback to 1st syllable consonant
            c1_1 = extract_primary_consonant(syls1[0]) if syls1 else ''
            c1_2 = extract_primary_consonant(syls2[0]) if syls2 else ''
            a_score, a_reason = calculate_consonant_phonic_score(c1_1, c1_2)
            adiprasa_score = a_score
            adiprasa_reason = f'Adyakshara: {a_reason} ({c1_1} ↔ {c1_2})'

        # Also check if 1st consonant alliteration has higher score
        if len(syls1) >= 1 and len(syls2) >= 1:
            c1_1 = extract_primary_consonant(syls1[0])
            c1_2 = extract_primary_consonant(syls2[0])
            a_score, a_reason = calculate_consonant_phonic_score(c1_1, c1_2)
            if a_score > adiprasa_score:
                adiprasa_score = a_score
                adiprasa_reason = f'Adyakshara alliteration: {a_reason} ({c1_1} ↔ {c1_2})'

        # --- 2. ANTYAPRĀSA (End Rhyme) ---
        end_syls1 = self.syllabify_word(last_word1)
        end_syls2 = self.syllabify_word(last_word2)

        antyaprasa_score = 0
        antyaprasa_reason = 'No match'

        if end_syls1 and end_syls2:
            last_s1 = end_syls1[-1]
            last_s2 = end_syls2[-1]
            c_end1 = extract_primary_consonant(last_s1)
            c_end2 = extract_primary_consonant(last_s2)
            v_end1 = extract_vowel(last_s1)
            v_end2 = extract_vowel(last_s2)

            c_score, c_reason = calculate_consonant_phonic_score(c_end1, c_end2)
            v_sim, v_reason = check_swara_similarity(v_end1, v_end2)

            if c_score == 0:
                antyaprasa_score = 0
                antyaprasa_reason = f'Disjoint ending consonants ({c_end1} vs {c_end2})'
            elif v_sim == 0.0:
                antyaprasa_score = 0
                antyaprasa_reason = f'{c_reason}, BUT {v_reason} (0/20)'
            elif v_sim == 1.0:
                antyaprasa_score = c_score
                antyaprasa_reason = f'{c_reason} with matching swara ({v_end1})'
            else:
                antyaprasa_score = round(c_score * v_sim)
                antyaprasa_reason = f'{c_reason} with {v_reason}'

        # Score thresholds: >= 12 is considered valid rhyming in poetry
        adiprasa_valid = (adiprasa_score >= 12)
        antyaprasa_valid = (antyaprasa_score >= 12)

        return {
            'adiprasa': adiprasa_valid,
            'antyaprasa': antyaprasa_valid,
            'adiprasa_score': adiprasa_score,
            'antyaprasa_score': antyaprasa_score,
            'total_prasa_score': adiprasa_score + antyaprasa_score,
            'adiprasa_reason': adiprasa_reason,
            'antyaprasa_reason': antyaprasa_reason,
            'adiprasa_pair': (first_word1, first_word2),
            'antyaprasa_pair': (last_word1, last_word2)
        }

    def verify_line_against_prosodic_mask(self, line_text: str, expected_mātras: List[int]) -> Dict:
        """
        Verifies whether the Kannada line matches the expected list of note mātras:
        1. Proportional syllable-to-note mātrā matching (0-17.5 points per line, 35 total).
        2. Proportional word boundary integrity (0-7.5 points per line, 15 total).
        """
        clean_line = re.sub(r'[|!?,.]', '', line_text).strip()
        words = clean_line.split()

        word_scans = []
        all_syllable_mātras = []
        syllable_to_word = []

        for w_idx, w in enumerate(words):
            syls = self.scan_syllables([w])
            total_w_mātras = sum(s.mātras for s in syls)
            word_scans.append((w, syls, total_w_mātras))
            for s in syls:
                all_syllable_mātras.append(s.mātras)
                syllable_to_word.append(w_idx)

        actual_total_mātras = sum(all_syllable_mātras)
        expected_total_mātras = sum(expected_mātras)

        # 1. Proportional Syllable-to-Note Match (17.5 points per line)
        num_notes = len(expected_mātras)
        num_syls = len(all_syllable_mātras)
        raw_matches = 0.0
        min_len = min(num_notes, num_syls)

        for j in range(min_len):
            if all_syllable_mātras[j] == expected_mātras[j]:
                raw_matches += 1.0
            elif abs(all_syllable_mātras[j] - expected_mātras[j]) == 1:
                raw_matches += 0.3

        len_penalty = abs(num_notes - num_syls) * 0.5
        raw_meter_pct = max(0.0, (raw_matches - len_penalty) / max(num_notes, 1))

        # Factor in total matras match ratio (severely penalizes wrong length gibberish)
        matra_ratio = min(actual_total_mātras, expected_total_mātras) / max(actual_total_mātras, expected_total_mātras, 1)
        meter_match_pct = raw_meter_pct * (matra_ratio ** 2)
        meter_score = round(meter_match_pct * 17.5, 1)

        exact_syllable_match = (all_syllable_mātras == expected_mātras)

        # 2. Word Boundary Check (7.5 points per line)
        valid_words_count = 0
        cum_word = 0
        cum_notes = set()
        c = 0
        for m in expected_mātras:
            c += m
            cum_notes.add(c)

        for _, _, w_m in word_scans:
            cum_word += w_m
            if cum_word in cum_notes:
                valid_words_count += 1

        num_words = len(word_scans)
        raw_word_pct = (valid_words_count / num_words) if num_words > 0 else 0.0
        word_boundary_pct = raw_word_pct * matra_ratio
        word_boundary_score = round(word_boundary_pct * 7.5, 1)

        word_boundary_ok = (valid_words_count == num_words) and (actual_total_mātras == expected_total_mātras)
        is_valid = exact_syllable_match and word_boundary_ok

        return {
            'is_valid': is_valid,
            'matches_total': (actual_total_mātras == expected_total_mātras),
            'actual_total': actual_total_mātras,
            'expected_total': expected_total_mātras,
            'exact_syllable_note_match': exact_syllable_match,
            'meter_match_pct': round(meter_match_pct * 100.0, 1),
            'meter_score': meter_score,
            'word_boundary_ok': word_boundary_ok,
            'word_boundary_pct': round(word_boundary_pct * 100.0, 1),
            'word_boundary_score': word_boundary_score,
            'syllables': [s.text for _, syls, _ in word_scans for s in syls],
            'syllable_mātras': all_syllable_mātras,
            'word_breakdown': [(w, m) for w, _, m in word_scans],
            'expected_notes': expected_mātras
        }
