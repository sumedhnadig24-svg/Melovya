
"""
Interactive Terminal App for Lyrics Writing Engine (v4.0)
Updated with:
1. Universal Formula: T_laghu = 60 / (BPM * 2.8)
2. Dedicated Semantic Meaning & Vocabulary Score (25 pts)
3. Strict Meaning Gatekeeper: Meaningless / fabricated words are rejected (Capped to Very Bad)
4. Swara (Vowel) Acoustic Similarity & Varga Phonic Prāsa (25 pts)
5. Syllable Meter (35 pts) & Whole-Word Integrity (15 pts)
"""

import sys
import os
import webbrowser
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from prosody_engine import (
    MelodyLyricsPipeline,
    SongQuestionnaire,
    NoteEvent
)
from rlhf_demo import run_rlhf_demo
from demo import run_demo
from test_audio_lyrics import test_audio

PRESETS_DATA = {
    "1": {
        "style": "Modern Sandalwood (Melody)",
        "mood": "Romantic Love (ಪ್ರೇಮ)",
        "theme": "Love & Admiration",
        "l1": "ನಿನ್ನ ಕಣ್ಣಲಿ ಕಂಡೆ ಲವ್ವನು",
        "l2": "ನನ್ನ ಜೀವಕೆ ತಂದೆ ಮಿಂಚನು",
        "meaning1": "In your eyes, I saw love",
        "meaning2": "To my soul, you brought a spark!"
    },
    "2": {
        "style": "Commercial Mass (ಪಕ್ಕಾ ಮಾಸ್)",
        "mood": "High Energy & Swagger (ಉತ್ಸಾಹ)",
        "theme": "Celebration & Style",
        "l1": "ನೋಡು ಸ್ಟೈಲಲಿ ರಂಗು ಚೆಲ್ಲಿದೆ",
        "l2": "ಹಾಡು ಜೋರಲಿ ಕಿಕ್ಕು ಹತ್ತಿದೆ",
        "meaning1": "Look, in style, colors are sprinkled!",
        "meaning2": "Sing aloud, the kick has fully set in!"
    },
    "3": {
        "style": "Janapada / Folk (ಜಾನಪದ)",
        "mood": "Earthy Village Rhythm (ಮಣ್ಣಿನ ಸಡಗರ)",
        "theme": "Village Life & Nature",
        "l1": "ಚಿನ್ನ ಚೆಲ್ಲಿದೆ ಬಣ್ಣ ಮಣ್ಣಲಿ",
        "l2": "ಬೆಣ್ಣೆ ಕೆನ್ನೆಯ ಚೆಂದ ಮಣ್ಣಲಿ",
        "meaning1": "Gold has sprinkled colors across the earth",
        "meaning2": "Butter-soft cheeks look beautiful on this soil"
    },
    "4": {
        "style": "Heartbreak / Melancholy (ವಿರಹ)",
        "mood": "Sorrow & Longing (ನೋವು / ಕಣ್ಣೀರು)",
        "theme": "Pain of Separation",
        "l1": "ಕಣ್ಣ ಮೌನಕೆ ನೋವು ತುಂಬಿದೆ",
        "l2": "ಬೆನ್ನ ಭಾರಕೆ ಜೀವ ಬೆಂದಿದೆ",
        "meaning1": "To the eye's silence, sorrow has filled",
        "meaning2": "Under the heavy burden, the soul is scorched"
    },
    "5": {
        "style": "Slow Romantic (Pluta Melisma)",
        "mood": "Deep Romance (ದೀರ್ಘ ಸ್ವರ / ಮೆಲಿಸಾ)",
        "theme": "Romantic Longing",
        "l1": "ಕಣ್ಣಲಿ ಬಾ~ ಒಲವೇ~~",
        "l2": "ಬಾಳಲಿ ನಾ~ ಚೆಲುವೇ~~",
        "meaning1": "In my eyes, come oh love~~",
        "meaning2": "In this life, I stay, oh beautiful one~~"
    }
}


def interactive_session():
    print("=" * 72)
    print(" 🎵 KANNADA AI LYRICS WRITING APP - INTERACTIVE STUDIO v4.0 🎵")
    print("=" * 72)

    # 1. Questionnaire
    print("\n--- [STEP 1] SONG QUESTIONNAIRE ---")
    bpm_input = input("Enter Tempo (BPM) [Default 83]: ").strip()
    bpm = float(bpm_input) if bpm_input else 83.0

    scale_root = input("Enter Scale Root (C, D, E, F, G, A, B) [Default C]: ").strip().upper() or 'C'
    scale_type = input("Enter Scale Type (major, minor, mohana) [Default major]: ").strip().lower() or 'major'

    print("\nSelect Lyrics Style & Mood:")
    print("  1. Modern Sandalwood (Melody Romance) [ಪ್ರೇಮ]")
    print("  2. Commercial Mass (High Energy Swagger) [ಪಕ್ಕಾ ಮಾಸ್]")
    print("  3. Janapada / Folk (Earthy Village Rhythm) [ಜಾನಪದ]")
    print("  4. Heartbreak / Melancholy (Pain & Separation) [ವಿರಹ]")
    print("  5. Slow Romantic (Pluta Melisma) [ದೀರ್ಘ ಸ್ವರ]")
    style_choice = input("Enter choice (1/2/3/4/5) [Default 1]: ").strip() or "1"

    selected_preset = PRESETS_DATA.get(style_choice, PRESETS_DATA["1"])

    questionnaire = SongQuestionnaire(
        tempo_bpm=bpm,
        scale_root=scale_root,
        scale_type=scale_type,
        lyrics_style=selected_preset["style"],
        mood=selected_preset["mood"],
        theme=selected_preset["theme"]
    )

    pipeline = MelodyLyricsPipeline(questionnaire)
    t_laghu = pipeline.quantizer.t_laghu_ms

    print("\n" + "-" * 62)
    print(f"✅ Configuration Saved!")
    print(f"   • Style & Mood : {selected_preset['style']} ({selected_preset['mood']})")
    print(f"   • Formula      : T_laghu = 60 / (BPM * 2.8)")
    print(f"   • Laghu Time (⏑): {t_laghu:.2f} ms")
    print(f"   • Guru Time (-):  {t_laghu * 2:.2f} ms (1 Beat)")
    print(f"   • Pluta 3 (-~):   {t_laghu * 3:.2f} ms (1.5 Beats)")
    print(f"   • Pluta 4 (-~~):  {t_laghu * 4:.2f} ms (2 Beats)")
    print("-" * 62)

    # 2. Melody Processing
    if style_choice == "5":
        # 7-note Pluta melody [2, 1, 1, 3, 1, 1, 4] = 13 mātras
        events = [
            NoteEvent(130.81, t_laghu * 2), NoteEvent(146.83, t_laghu * 1), NoteEvent(164.81, t_laghu * 1),
            NoteEvent(196.00, t_laghu * 3), NoteEvent(220.00, t_laghu * 1), NoteEvent(246.94, t_laghu * 1),
            NoteEvent(261.63, t_laghu * 4), NoteEvent(None,   t_laghu * 4),
            NoteEvent(130.81, t_laghu * 2), NoteEvent(146.83, t_laghu * 1), NoteEvent(164.81, t_laghu * 1),
            NoteEvent(196.00, t_laghu * 3), NoteEvent(220.00, t_laghu * 1), NoteEvent(246.94, t_laghu * 1),
            NoteEvent(261.63, t_laghu * 4)
        ]
    else:
        # Standard 10-note phrase: [2, 1, 2, 1, 1, 2, 1, 2, 1, 1] = 14 mātras
        events = [
            NoteEvent(130.81, t_laghu * 2), NoteEvent(146.83, t_laghu * 1),
            NoteEvent(164.81, t_laghu * 2), NoteEvent(174.61, t_laghu * 1),
            NoteEvent(196.00, t_laghu * 1), NoteEvent(220.00, t_laghu * 2),
            NoteEvent(196.00, t_laghu * 1), NoteEvent(174.61, t_laghu * 2),
            NoteEvent(164.81, t_laghu * 1), NoteEvent(130.81, t_laghu * 1),
            NoteEvent(None,   t_laghu * 4),
            NoteEvent(130.81, t_laghu * 2), NoteEvent(146.83, t_laghu * 1),
            NoteEvent(164.81, t_laghu * 2), NoteEvent(174.61, t_laghu * 1),
            NoteEvent(196.00, t_laghu * 1), NoteEvent(220.00, t_laghu * 2),
            NoteEvent(196.00, t_laghu * 1), NoteEvent(174.61, t_laghu * 2),
            NoteEvent(164.81, t_laghu * 1), NoteEvent(130.81, t_laghu * 1)
        ]

    phrases = pipeline.process_melody(events)

    print("\n--- [STEP 2] PROSODIC STRING MASK ---")
    for p in phrases:
        sym_str = " ".join([f"{n.prosody_symbol}({n.mātras}m)" for n in p.notes])
        print(f"Phrase #{p.phrase_id} [Total: {p.total_mātras}m, {len(p.notes)} notes]: {sym_str}")
        print(f"             Target Note Mātras: {p.mātra_groups}")

    # 3. Lyrics Verification
    default_l1 = selected_preset["l1"]
    default_l2 = selected_preset["l2"]

    print("\n--- [STEP 3] ENTER KANNADA LYRICS ---")
    print(f"Genre Preset Lyrics (Connected & Grammatically Meaningful):")
    print(f"  Line 1: '{default_l1}' -> {selected_preset['meaning1']}")
    print(f"  Line 2: '{default_l2}' -> {selected_preset['meaning2']}")
    print("\nPress ENTER to test preset, or type your own Kannada lines:")
    custom_l1 = input("Line 1 (or Enter for preset): ").strip() or default_l1
    custom_l2 = input("Line 2 (or Enter for preset): ").strip() or default_l2

    lyrics = [custom_l1, custom_l2]

    print("\n--- [STEP 4] RUNNING SEMANTIC MEANING & PROSODY VERIFICATION ---")
    result = pipeline.verify_lyrics(lyrics, phrases)

    score = result['overall_score']
    rating = result['quality_rating']
    sem = result['semantic_eval']

    print(f"\n========================================================")
    print(f" FINAL SCORE: {score} / 100 -> {rating}")
    print(f"========================================================")
    print(f"1. Semantic Meaning & Vocabulary: {result['meaning_score']} / 25")
    print(f"   • Summary: {sem['summary']}")
    if sem['invalid_words']:
        print(f"   • ❌ Fabricated Words Detected: {', '.join(sem['invalid_words'])}")
    print(f"2. Syllable-to-Note Meter Score : {result['meter_score_total']} / 35")
    print(f"3. Whole Word Integrity Score   : {result['word_score_total']} / 15")
    print(f"4. Varga & Swara Prāsa Score    : {result['prasa_score_total']} / 25")

    for lr in result['line_results']:
        v = lr['verification']
        status = "MATCH ✅" if v['is_valid'] else f"PARTIAL ({v['meter_score']}/17.5m, {v['word_boundary_score']}/7.5w)"
        print(f"\nLine: '{lr['line_text']}' -> {status}")
        print(f"  Expected Note Mātras: {v['expected_notes']}")
        print(f"  Actual Syllable Māt : {v['syllable_mātras']}")
        print(f"  Syllables           : {'  |  '.join(v['syllables'])}")
        print(f"  Words (mātras)      : {'  |  '.join([f'{w} ({m}m)' for w, m in v['word_breakdown']])}")

    for pr in result['prasa_checks']:
        p = pr['prasa_result']
        print(f"\nRhyme Analysis (Lines {pr['lines']}):")
        print(f"  • Ādiprāsa   (Start) : {p['adiprasa_score']}/20 -> {p['adiprasa_reason']}")
        print(f"  • Antyaprāsa (End)   : {p['antyaprasa_score']}/20 -> {p['antyaprasa_reason']}")

    print("\n" + "=" * 72)


def launch_web_studio():
    from http.server import HTTPServer, SimpleHTTPRequestHandler
    html_file = "interactive_lyrics_studio.html"
    server = None
    port = 8000

    for test_port in [8000, 8080, 5000, 5500]:
        try:
            server = HTTPServer(('127.0.0.1', test_port), SimpleHTTPRequestHandler)
            port = test_port
            break
        except Exception:
            continue

    if server:
        url = f"http://127.0.0.1:{port}/{html_file}"
        print(f"\n🌐 Local Server Started: {url}")
        print("🎤 Real-time Microphone Permissions are active on http://127.0.0.1!")
        print("Opening your browser now... Press Ctrl+C to stop server.\n")
        webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
    else:
        abs_path = os.path.abspath(html_file)
        print(f"\n🌐 Opening Visual Web Studio in your browser: file://{abs_path}")
        webbrowser.open(f"file://{abs_path}")


def main():
    print("=" * 74)
    print(" 🎵 KANNADA AI LYRICS WRITING APP - MASTER LAUNCHER v5.0 🎵")
    print("=" * 74)
    print("Select what you would like to run:")
    print("  [1] Interactive Lyrics Studio & Questionnaire (Test custom lyrics live)")
    print("  [2] Audio Melody Analyzer & Silence Line Segmenter (Humming -> Multi-line)")
    print("  [3] Run Neural AI Foundation Agent & RLHF DPO Weight Nudge Demo")
    print("  [4] Run Automated 6-Couplet Prosody Verification Suite (demo.py)")
    print("  [5] Open Visual Web Studio in Browser (with Live Mic Visualizer)")
    print("=" * 74)

    choice = input("Enter choice (1 / 2 / 3 / 4 / 5) [Default 5]: ").strip() or "5"

    if choice == "2":
        test_audio()
    elif choice == "3":
        run_rlhf_demo()
    elif choice == "4":
        run_demo()
    elif choice == "1":
        interactive_session()
    else:
        launch_web_studio()


if __name__ == '__main__':
    main()
