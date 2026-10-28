# AI Lyrics Writing App - Prototype Engine v4.0

A specialized music-prosody engine for Kannada lyrics generation, bridging the gap between musical audio (BPM, frequencies, durations) and classical/modern Indian poetic meters (**Laghu**, **Guru**, and **Pluta**).

---

## 📐 Core Mathematical Equations

### 1. Universal Atomic Time Formula
Given tempo in BPM:
$$\mathbf{T_{\text{Laghu}}} = \frac{60}{\text{BPM} \times 2} \quad (\text{seconds}) = \frac{60 \times 1000}{\text{BPM} \times 2} \quad (\text{ms})$$

* **Laghu (`⏑`)**: $1 \times T_{\text{Laghu}}$
* **Guru (`-`)**: $2 \times T_{\text{Laghu}} = \frac{60}{\text{BPM}}$ (1 full quarter-note beat)
* **Pluta (`-~`, `-~~`, `-~~~`)**: $N \times T_{\text{Laghu}} \quad (N \ge 3)$

### 2. Frequency to MIDI & Melodic Contour Snapping
$$\text{MIDI} = 69 + 12 \log_2\left(\frac{f}{440}\right)$$
* Snaps ambiguous frequencies (between semitones) based on **melodic contour (trend)** and **user-selected scale priority**.

---

## 🏛 Four Evaluation Pillars (100 Marks Total)

$$\mathbf{\text{Total Score}} = \text{Meaning \& Vocab (25)} + \text{Syllable Meter (35)} + \text{Whole Words (15)} + \text{Varga Prāsa (25)}$$

### 1. Semantic Meaning & Vocabulary Score (25 Marks)
* **Word Legitimacy**: Every word is verified against authentic Kannada roots and grammatical vibhakti/sandhi inflections.
* **Fabricated Word Detection**: Non-existent / invented words (like `ಗಾವೇ`, `ಕಾಕೌ`) are detected and penalized.
* **Sentence Coherence**: Ensures words combine into a natural, grammatically meaningful thought.
* **Theme & Style Harmony**: Vocabulary aligns with the chosen mood (Romantic, Mass, Janapada, Melancholy).
* **Strict Meaning Gatekeeper**: If fabricated words are present or Meaning Score is low, the lyrics are **strictly capped ($\le 35\%$) and flagged as "Very Bad Lyrics / Meaningless 🚫"**.

### 2. Syllable-to-Note Meter Alignment (35 Marks)
* Proportional matching of actual syllable weights ($1\text{m}, 2\text{m}, 3\text{m}, 4\text{m}$) against melody note durations.
* Scaled by total mātra ratio to prevent length discrepancies.

### 3. Whole-Word Integrity (Padachheda) (15 Marks)
* Ensures every word boundary lands on a musical beat boundary (zero word-slicing).

### 4. Varga & Swara Prāsa Rhyme Scoring (25 Marks)
* Consonant Vargas: ಕ, ಚ, ಟ, ತ, ಪ
* Acoustic Cousins: `ಣ` ↔ `ನ` (14/20), `ಡ` ↔ `ದ` (14/20), `ಟ` ↔ `ತ` (14/20), `ಲ` ↔ `ಳ` (14/20)
* Swara (Vowel) Acoustic Affinities:
  * Identical: Full score
  * Short/Long Savarna pairs (`ೆ` ↔ `ೇ`, `ಿ` ↔ `ೀ`, `ು` ↔ `ೂ`): Acoustic cousins
  * Dissimilar swaras (`ೇ` vs `ೌ` as in `ಗಾವೇ` vs `ಕಾಕೌ`): **0 marks (No vowel rhyme)**

---

## 🧠 Neural AI Foundation Agent & RLHF Architecture (v5.0)

How the system scales to an autonomous foundation model (similar to ChatGPT, Claude, and Suno AI):

### 1. The Neuro-Symbolic Dual-Engine Pipeline
```
[User Input: Melody Audio / Humming + BPM + Scale + Style + Mood]
                          │
                          ▼
[Deterministic / Algorithmic Signal Processing Frontend]
  • Pitch Snapping & Semitone Mapping
  • Time Quantization: T_laghu = 60 / (BPM * 2)
  • Prosodic String Mask Compilation: M = [G, L, G, L, L, G, L, G, L, L]
                          │
                          ▼
[Neural Foundation Language Model (Weights θ)]
  • Autoregressive Transformer with learned parameter weights W_e, W_attn, W_head
  • Pretrained on millions of song lyrics to learn open-ended poetic semantics
    (e.g., 'ತೇಲುತ ಮೈಯ ಮರೆತೆನು' without static lookup dictionaries)
  • Prosody-Constrained Logit Decoding:
      P(w_t | w_{<t}, M) = softmax( (W_head * h_t + ProsodyMask(v, M)) / T )
                          │
                          ▼
[Human Feedback Loop (RLHF / DPO)]
  • Lyricist selects preferred phrasing: y_w ≻ y_l
  • Modern Song Reward R(x, y): Antyaprāsa (45%), Semantics (35%), Prosody (15%), Ādiprāsa (5%)
  • Weight Nudging via DPO Loss Backpropagation:
      θ ← θ - η * ∇_θ L_DPO(θ)
```

### 2. Antyaprāsa-First Songwriting Rule
* **Antyaprāsa (End-Rhyme)** is the 95% standard in modern Kannada film and popular music.
* If Antyaprāsa matches (e.g. `ಲವ್ವನು` ↔ `ಮರೆತೆನು`), it achieves full Prāsa marks (**up to 25/25**), even if Ādiprāsa is 0!
* **Ādiprāsa (Initial alliteration)** is treated as an optional classical bonus, **never a prerequisite or penalty**.

---

## 🎙️ Real-Time Audio Humming & 2-Stage Lyrics Generator (v5.2)

Processes audio from live microphone or files (`.wav`, `.mp3`, `.m4a`, `.ogg`):
1. **Long-Term Silence & Breath Pause Detection (VAD)**:
   - Evaluates short-time RMS energy.
   - Automatically segments audio into phrases/lines whenever natural breath silences ($\ge 400\text{ms}$) occur.
   - Dynamically scales to **2 lines, 4 lines, 10 lines, 50 lines, 100 lines, or 1,000 lines**.
2. **Pitch Tracking ($F_0$) & Trend Snapping**:
   - Uses normalized autocorrelation to track humming pitch and identify sustained note boundaries.
   - Snaps ambiguous notes using melodic contour trend (ascending $\rightarrow$ higher note, descending $\rightarrow$ lower note, initial ambiguous $\rightarrow$ lower note).
3. **Note Subdivision & Merging**:
   - **Sub-Laghu Merging**: 2 fast notes each $< 0.75 \times T_{\text{Laghu}}$ merge into 1 Laghu ($1\text{m}$).
   - **Note $\le 4$ Matras**: Kept intact as Laghu ($1\text{m}$), Guru ($2\text{m}$), Pluta 3 ($-~$, $3\text{m}$ e.g. `ಸೋ~`), or Pluta 4 ($-~~$, $4\text{m}$ e.g. `ಆ~~`).
   - **Long Sustained Notes ($5$ to $30$ Matras)**: Converted $50\% - 75\%$ randomly into Laghus ($1\text{m}$) and Gurus ($2\text{m}$), keeping the remaining tail as Pluta sustain ($-~$ or $-~~$).
   - **Musical Gaps**: Melodic silences and breath pauses are preserved as spaces between words/lines.
4. **STAGE 1: Raw Note-to-Single-Letter Prototype (ಅಕ್ಷರ ಪ್ರತ್ಯಕ್ಷ ರೂಪ)**:
   - For every single note, assigns exactly **1 single Kannada letter/akshara** matching the acoustic matra weight.
   - Unconstrained by dictionary words (e.g., A3 for $4\text{m}$ $\rightarrow$ `ಆ~~`, C3 for $3\text{m}$ $\rightarrow$ `ಸೋ~`).
   - Displays the unconstrained phonetic acoustic blueprint (`ಹಾ~~ ತೋ~ ಜ ರಾ ಕಂ ...`) directly to the user to inspect matra alignment.
5. **STAGE 2: Meaningful Stylistic Transformation (ಅರ್ಥಪೂರ್ಣ ಸಾಹಿತ್ಯ)**:
   - Translates the acoustic prosodic pattern into authentic, grammatically coherent Kannada lines.
   - Conditioned on style: **Melody Romance (ಮೆಲೊಡಿ ಪ್ರೇಮ)**, **Commercial Mass (ಪಕ್ಕಾ ಮಾಸ್)**, **Janapada / Folk (ಜಾನಪದ)**, or **Heartbreak / Sad (ವಿರಹ ಭಾವ)**.
   - **Non-Repetition Memory**: Tracks words across lines so lines in the same song never repeat the same words.
   - **Antyaprāsa Coupling**: Pairs consecutive couplet lines with matching end-rhymes (consonant varga + swara similarity).
6. **Quality Gatekeeper Verification**:
   - Automatically audits meter lock, word boundaries, semantic meaning score, and Antyaprāsa before delivering output.

---

## 🏆 Graduated Quality Tiers

* **$\ge 85\%$**: **Good Lyrics (ಉತ್ತಮ ಸಾಹಿತ್ಯ) ✅**
* **$80\% - 84\%$**: **Good / Acceptable Lyrics (ಉತ್ತಮ ಸಾಹಿತ್ಯ) ✅**
* **$50\% - 79\%$**: **Moderate Lyrics (ಮಧ್ಯಮ ಸಾಹಿತ್ಯ) ⚠️**
* **$11\% - 49\%$**: **Bad Lyrics (ಕಳಪೆ ಸಾಹಿತ್ಯ) ❌**
* **$0\% - 10\%$**: **Very Bad Lyrics (ತೀರಾ ಕಳಪೆ ಸಾಹಿತ್ಯ) 🚫**

---

## 🚀 Running the Application

```bash
# 1. Master Console Launcher (Interactive menu for all tools)
python app.py

# 2. Audio Melody Analyzer & Multi-Line Generator CLI
python test_audio_lyrics.py

# 3. Neural Foundation Agent & RLHF DPO Weight Nudging Demo
python rlhf_demo.py

# 4. Visual Web Studio (with Live Mic Recording & Audio File Upload)
# Open interactive_lyrics_studio.html directly in any web browser
```
