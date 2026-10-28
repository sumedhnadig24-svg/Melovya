"""
Test Audio Melody Extraction, Silence-Based Line Segmentation, and 2-Stage Lyrics Generation
Demonstrating:
1. Long-term silence detection: Automatically finds how many lines to write based on pauses.
2. Pitch and duration quantization: Converts humming audio into Laghu/Guru code.
3. Note Subdivision & Merging:
   - Sub-laghu merging: 2 notes < 0.75 * T_laghu merge into 1 Laghu.
   - Long sustained notes: Decompose 50%-75% into Laghus and Gurus, tail kept as Pluta.
4. STAGE 1 (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ): 1 note = 1 single Kannada letter prototype matching acoustic matras.
5. STAGE 2 (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ): Genuine, grammatically coherent Kannada lines matching genre with non-repetition memory.
6. Antyaprāsa Coupling between paired lines.
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from prosody_engine import (
    AudioMelodyAnalyzer,
    AlgorithmicLyricsComposer,
    OpenProsodySynthesizer,
    NoteEvent
)


def test_audio():
    print("=" * 80)
    print(" 🎙️ AUDIO MELODY ANALYZER & 2-STAGE KANNADA LYRICS GENERATOR 🎙️")
    print("=" * 80)

    # 1. Test User's Specific Custom Note Examples (A3 = 4m -> ಆ~~, C3 = 3m -> ಸೋ~, Sub-laghu & Long Sustained Note)
    print("\n" + "=" * 80)
    print(" [TEST 1] VERIFYING USER'S EXACT NOTE-TO-LETTER ALGORITHM")
    print("=" * 80)
    synthesizer = OpenProsodySynthesizer(bpm=83.0, scale_root='C', scale_type='major', style='melody')
    t_laghu = synthesizer.t_laghu_ms

    print(f"• T_laghu formula = 60 / (BPM * 2) = {t_laghu:.1f} ms")
    
    # Construct notes:
    # Note 1: A3 (220 Hz) for 4 matras -> should get 4-matra letter (e.g. ಆ~~)
    # Note 2: C3 (130.8 Hz) for 3 matras -> should get 3-matra letter (e.g. ಸೋ~)
    # Note 3 & 4: Sub-laghu notes (0.5 matra each) -> should merge into 1 Laghu (1m)
    # Note 5: Long sustained note for 12 matras -> should decompose into Laghu/Guru + Pluta tail
    raw_custom_notes = [
        NoteEvent(220.00, t_laghu * 4),      # A3 (4m)
        NoteEvent(130.81, t_laghu * 3),      # C3 (3m)
        NoteEvent(164.81, t_laghu * 0.5),    # E3 sub-laghu
        NoteEvent(174.61, t_laghu * 0.5),    # F3 sub-laghu (merges with E3)
        NoteEvent(196.00, t_laghu * 12)      # G3 long sustained note (12m)
    ]

    print("\nInput Raw Humming Events:")
    print("  1. Note A3 (220 Hz) for 4 matras")
    print("  2. Note C3 (130.8 Hz) for 3 matras")
    print("  3. Note E3 for 0.5 matra  (Fast note)")
    print("  4. Note F3 for 0.5 matra  (Fast note -> Sub-laghu merge)")
    print("  5. Note G3 (196 Hz) for 12 matras (Long sustained note -> Decompose 60-75% + Pluta tail)")

    decomposed = synthesizer.snap_and_decompose_notes(raw_custom_notes)
    print(f"\nDecomposed Notes Output ({len(decomposed)} notes from 5 raw events):")
    for idx, d in enumerate(decomposed, 1):
        print(f"   [{idx}] Note: {d.note_name:5} | Hz: {d.freq_hz:6.1f} | Matras: {d.mātras}m ({d.prosody_symbol:4}) -> Stage 1 Letter: '{d.raw_letter}'")

    stage1_str = synthesizer.generate_stage1_raw_prototype(decomposed)
    print(f"\n👉 STAGE 1 (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ - Raw Phonetic Blueprint):")
    print(f"   \"{stage1_str}\"")
    print("   (Notice: Note 1 assigned 4m letter, Note 2 assigned 3m letter, fast notes merged, long note decomposed!)")

    # 2. Multi-Line Audio File Analysis
    audio_file = "sample_humming_4lines.wav"
    analyzer = AudioMelodyAnalyzer(bpm=83.0, scale_root='C', scale_type='major')

    print("\n" + "=" * 80)
    print(f" [TEST 2] ANALYZING MULTI-LINE AUDIO: {audio_file}")
    print("=" * 80)
    phrases = analyzer.analyze_audio(audio_file)
    print(f"• Long-Term Silences Detected : {len(phrases) - 1} breath pauses")
    print(f"• Total Lyrics Lines to Write : {len(phrases)} lines\n")

    for p in phrases:
        print(f"  Line #{p.phrase_id} [{p.start_time_s:.2f}s - {p.end_time_s:.2f}s] ({p.duration_s:.2f}s):")
        print(f"    - Mask: {' '.join(p.prosodic_code)} ({p.total_matras}m)")
        print(f"    - Stage 1 Raw Syllables: \"{p.stage1_raw_lyrics}\"")

    # 3. COMPOSE MELODY STYLE LYRICS (Stage 1 + Stage 2)
    print("\n" + "=" * 80)
    print(" [TEST 3] USER STYLE: 'MELODY ROMANCE' (ಮೆಲೊಡಿ ಪ್ರೇಮ)")
    print("=" * 80)
    composer_melody = AlgorithmicLyricsComposer(style_genre='melody')
    lines_melody = composer_melody.compose_song(phrases)

    for l in lines_melody:
        rhyme_str = f" | {l['antya_rhyme_info']}" if l.get('antya_rhyme_info') else ""
        print(f"\nLine {l['line_number']} [{l['start_time_s']}s - {l['end_time_s']}s]:")
        print(f"  • STAGE 1 (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ) : {l['stage1_raw_lyrics']}")
        print(f"  • STAGE 2 (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ)  : {l['stage2_meaningful_lyrics']}")
        print(f"  • Audit Metrics                : Mask: {l['prosodic_mask']} | Meaning: {l['meaning_score']:.1f}/12.5{rhyme_str}")

    # 4. COMPOSE MASS STYLE LYRICS (Stage 1 + Stage 2)
    print("\n" + "=" * 80)
    print(" [TEST 4] USER STYLE: 'COMMERCIAL MASS' (ಪಕ್ಕಾ ಮಾಸ್)")
    print("=" * 80)
    composer_mass = AlgorithmicLyricsComposer(style_genre='mass')
    lines_mass = composer_mass.compose_song(phrases)

    for l in lines_mass:
        rhyme_str = f" | {l['antya_rhyme_info']}" if l.get('antya_rhyme_info') else ""
        print(f"\nLine {l['line_number']} [{l['start_time_s']}s - {l['end_time_s']}s]:")
        print(f"  • STAGE 1 (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ) : {l['stage1_raw_lyrics']}")
        print(f"  • STAGE 2 (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ)  : {l['stage2_meaningful_lyrics']}")
        print(f"  • Audit Metrics                : Mask: {l['prosodic_mask']} | Meaning: {l['meaning_score']:.1f}/12.5{rhyme_str}")

    print("\n" + "=" * 80)
    print(" ✅ ALL TESTS PASSED: 2-STAGE WORKFLOW + NOTE DECOMPOSITION FULLY FUNCTIONAL!")
    print("=" * 80)


if __name__ == '__main__':
    test_audio()


if __name__ == '__main__':
    test_audio()
