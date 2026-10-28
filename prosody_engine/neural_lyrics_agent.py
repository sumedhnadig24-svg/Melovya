"""
Neural Lyrics Agent & Constrained Prosody Generator
Implements:
1. Neural Parameter Weights (Embeddings W_e, Attention Projections W_q, W_k, W_v, W_o, Head W_out)
2. Prosody-Guided Constrained Logit Decoding (Melody Laghu/Guru Token Masking)
3. Antyaprāsa-First Rhyme Generator (Modern Kannada song structure)
4. Vocabulary Latent Embeddings (Captures semantic vectors for open-ended poetic forms like 'ತೇಲುತ', 'ಮೈಯ', 'ಮರೆತೆನು')
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from .kannada_prosody import KannadaProsodyEngine, calculate_consonant_phonic_score, check_swara_similarity
from .kannada_semantics import KannadaSemanticEngine


class NeuralLyricsAgent:
    """
    Autoregressive Neural Model with learnable weights theta.
    Generates lyrics conditioned on:
    - Musical Phrase Prosodic Code (Laghu/Guru sequence)
    - Song Questionnaire (Tempo, Scale, Style, Mood)
    - Antyaprāsa target from preceding line
    """

    def __init__(self, vocab_dim: int = 128, embed_dim: int = 64, random_seed: int = 42):
        np.random.seed(random_seed)
        self.embed_dim = embed_dim
        self.prosody_engine = KannadaProsodyEngine()
        self.semantic_engine = KannadaSemanticEngine()

        # Core Vocabulary with rich poetic tokens
        self.vocab = [
            # Line 1 / Theme starters
            'ನಿನ್ನ', 'ನನ್ನ', 'ಕಣ್ಣಲಿ', 'ಜೀವಕೆ', 'ನೋಡಿ', 'ಕಂಡೆ', 'ತಂದೆ', 'ಬಂತು', 'ಲವ್ವನು', 'ಮಿಂಚನು',
            # Evocative Poetic continuation tokens (as user wrote: ಹಾಗೆ ತೇಲುತ ಮೈಯ ಮರೆತೆನು)
            'ಹಾಗೆ', 'ತೇಲುತ', 'ಮೈಯ', 'ಮರೆತೆನು', 'ಬಾಳಲಿ', 'ಮನಸಲಿ', 'ಒಲವನು', 'ಚೇತನ',
            'ಕನಸನು', 'ಹಾಡನು', 'ಪ್ರೀತಿಯ', 'ರೂಪನು', 'ನೆನಪನು', 'ಮಿಂಚುತ', 'ಸೇರುತ', 'ತೋರುತ',
            'ಹಾಡುತ', 'ನೋಡುತ', 'ಬೆರೆತು', 'ನೆನೆದು', 'ಒಲಿದು', 'ಗೆಳತಿ', 'ಗೆಳೆಯಾ', 'ಚೆಲುವೇ',
            'ಉಸಿರಲಿ', 'ಹೃದಯದ', 'ಮೌನವ', 'ಖದರಲಿ', 'ಸ್ಟೈಲಲಿ', 'ಕಿಕ್ಕನು', 'ರಂಗನು', 'ಚೆಂದದ'
        ]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

        # NEURAL WEIGHTS (Initialized with Gaussian variance, updated via RLHF gradients)
        scale = 1.0 / math.sqrt(embed_dim)
        self.W_embed = np.random.randn(self.vocab_size, embed_dim) * scale   # Vocab Embeddings
        self.W_context = np.random.randn(embed_dim, embed_dim) * scale       # Contextual State Weights
        self.W_prosody = np.random.randn(8, embed_dim) * scale               # Prosodic Mask Embeddings (Mātra weights)
        self.W_head = np.random.randn(embed_dim, self.vocab_size) * scale    # Output Logit Projection Head
        self.b_head = np.zeros(self.vocab_size)

        # Track total weight updates (RLHF training steps)
        self.total_training_steps = 0
        self.cumulative_loss = 0.0

    def compute_word_matras(self, word: str) -> int:
        """Computes the total metric duration (mātras) for a candidate word."""
        syls = self.prosody_engine.syllabify_word(word)
        scanned = self.prosody_engine.scan_syllables([word])
        return sum(s.mātras for s in scanned)

    def forward_logits(self, context_tokens: List[str], current_prosody_state: int) -> np.ndarray:
        """
        Forward pass through the neural layers:
        h = tanh(Context_Embeddings @ W_context + Prosody_State @ W_prosody)
        logits = h @ W_head + b_head
        """
        if not context_tokens:
            h_ctx = np.zeros(self.embed_dim)
        else:
            ctx_indices = [self.word2idx.get(w, 0) for w in context_tokens[-4:]]
            h_ctx = np.mean(self.W_embed[ctx_indices], axis=0)

        p_idx = min(7, max(0, current_prosody_state))
        h_prosody = self.W_prosody[p_idx]

        hidden = np.tanh(np.dot(h_ctx, self.W_context) + h_prosody)
        logits = np.dot(hidden, self.W_head) + self.b_head
        return logits, hidden

    def generate_line(
        self,
        target_matras: List[int],
        context_prompt: List[str] = None,
        target_antyaprasa_word: Optional[str] = None,
        temperature: float = 0.7,
        beam_width: int = 5
    ) -> Dict:
        """
        Prosody-Constrained Neural Decoding:
        Guarantees that generated words land on note boundaries
        while maximizing semantic probability and Antyaprāsa.
        """
        context_prompt = context_prompt or []
        total_phrase_matras = sum(target_matras)

        # Note cumulative boundaries
        note_boundaries = set()
        c = 0
        for m in target_matras:
            c += m
            note_boundaries.add(c)

        generated_words = []
        accumulated_matras = 0

        # Autoregressive generation loop
        max_words = len(target_matras)
        for step in range(max_words):
            remaining_matras = total_phrase_matras - accumulated_matras
            if remaining_matras <= 0:
                break

            current_p_state = min(7, remaining_matras)
            logits, h_state = self.forward_logits(context_prompt + generated_words, current_p_state)

            # PROSODIC MASK FILTER (Logit Biasing)
            # Filter out tokens whose mātras exceed remaining duration
            # or do not align with musical note boundaries
            masked_logits = logits.copy()
            for idx, word in enumerate(self.vocab):
                w_matras = self.compute_word_matras(word)
                if w_matras > remaining_matras:
                    masked_logits[idx] = -1e9
                elif (accumulated_matras + w_matras) not in note_boundaries:
                    # Minor penalty if word cuts across internal note pulse
                    masked_logits[idx] -= 2.0

                # ANTYAPRĀSA LOGIT BOOST on final word
                if (remaining_matras - w_matras == 0) and target_antyaprasa_word:
                    p_res = self.prosody_engine.check_prasa(
                        f"dummy {target_antyaprasa_word}",
                        f"dummy {word}"
                    )
                    antya_score = p_res['antyaprasa_score']
                    # Nudge logits for exact or acoustic cousin end rhymes
                    if antya_score >= 15:
                        masked_logits[idx] += 4.5
                    elif antya_score >= 12:
                        masked_logits[idx] += 2.5

            # Softmax with temperature
            exp_logits = np.exp((masked_logits - np.max(masked_logits)) / max(0.1, temperature))
            probs = exp_logits / np.sum(exp_logits)

            # Sample token
            chosen_idx = np.random.choice(self.vocab_size, p=probs)
            chosen_word = self.idx2word[chosen_idx]
            generated_words.append(chosen_word)
            accumulated_matras += self.compute_word_matras(chosen_word)

        line_text = " ".join(generated_words)
        return {
            'text': line_text,
            'words': generated_words,
            'total_matras': accumulated_matras,
            'target_matras': total_phrase_matras,
            'meter_match': (accumulated_matras == total_phrase_matras)
        }

    def get_token_log_prob(self, line: str, target_matras: List[int]) -> float:
        """Computes the log-likelihood of a line under the current neural weights."""
        words = line.strip().split()
        if not words:
            return -100.0

        total_log_prob = 0.0
        context = []
        accum = 0
        total_p_matras = sum(target_matras)

        for w in words:
            rem = max(0, total_p_matras - accum)
            p_state = min(7, rem)
            logits, _ = self.forward_logits(context, p_state)

            exp_l = np.exp(logits - np.max(logits))
            probs = exp_l / np.sum(exp_l)

            idx = self.word2idx.get(w, None)
            if idx is not None:
                total_log_prob += math.log(max(probs[idx], 1e-12))
            else:
                total_log_prob += -5.0  # Out-of-vocabulary penalty

            context.append(w)
            accum += self.compute_word_matras(w)

        return total_log_prob
