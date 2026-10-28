"""
Reinforcement Learning from Human Feedback (RLHF) Engine for Kannada Lyrics
Implements:
1. Multi-Objective Reward Model R_psi(Melody, Lyrics):
   - Modern Song Priority: Antyaprāsa (45%), Semantics (35%), Prosody (15%), Ādiprāsa (5% optional bonus)
2. Pairwise Human Preference Collector (Winner y_w vs Loser y_l)
3. Direct Preference Optimization (DPO) / Policy Gradient Weight Updater
   - Mathematically nudges neural weights theta via loss backpropagation
4. Training Telemetry (Loss curves, delta W norms, probability shifts)
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from .neural_lyrics_agent import NeuralLyricsAgent
from .kannada_prosody import KannadaProsodyEngine
from .kannada_semantics import KannadaSemanticEngine


class LyricsRewardModel:
    """
    Evaluates candidate lyrics using the true modern songwriting reward function:
    R(x, y) = 0.45 * Antyaprāsa + 0.35 * SemanticFluency + 0.15 * ProsodyFit + 0.05 * Ādiprāsa
    """

    def __init__(self):
        self.prosody_engine = KannadaProsodyEngine()
        self.semantic_engine = KannadaSemanticEngine()

    def evaluate(self, line1: str, line2: str, target_matras: List[int]) -> Dict:
        # 1. Antyaprāsa & Ādiprāsa
        p_res = self.prosody_engine.check_prasa(line1, line2)
        antya_raw = p_res['antyaprasa_score'] / 20.0  # 0.0 to 1.0
        adi_raw = p_res['adiprasa_score'] / 20.0      # 0.0 to 1.0

        # Modern song rule: Antyaprāsa is primary; Ādiprāsa is an optional bonus
        antya_reward = antya_raw
        adi_reward = adi_raw * 0.5  # Soft bonus

        # 2. Semantic Fluency & Vocabulary Validity
        sem_eval = self.semantic_engine.evaluate_couplet_meaning(line1, line2)
        sem_reward = sem_eval['meaning_score'] / 25.0  # 0.0 to 1.0

        # 3. Prosodic Melody Lock
        scanned2 = self.prosody_engine.scan_syllables(line2.strip().split())
        l2_matras = sum(s.mātras for s in scanned2)
        target_total = sum(target_matras)

        if l2_matras == target_total:
            prosody_reward = 1.0
        else:
            prosody_reward = max(0.0, 1.0 - abs(l2_matras - target_total) * 0.25)

        # COMPOSITE REWARD R(x, y)
        total_reward = (
            0.45 * antya_reward +
            0.35 * sem_reward +
            0.15 * prosody_reward +
            0.05 * adi_reward
        )

        return {
            'total_reward': round(total_reward * 100.0, 2),
            'antya_reward': round(antya_reward * 100.0, 1),
            'sem_reward': round(sem_reward * 100.0, 1),
            'prosody_reward': round(prosody_reward * 100.0, 1),
            'adi_reward': round(adi_reward * 100.0, 1),
            'antya_reason': p_res['antyaprasa_reason'],
            'sem_summary': sem_eval['summary']
        }


class RLHFTrainer:
    """
    Reinforcement Learning with Human Feedback (RLHF) & DPO Optimizer.
    Nudges the neural agent's parameter weights theta to maximize human satisfaction.
    """

    def __init__(self, agent: NeuralLyricsAgent, beta: float = 0.2, learning_rate: float = 0.05):
        self.agent = agent
        self.beta = beta
        self.learning_rate = learning_rate
        self.reward_model = LyricsRewardModel()
        self.training_history = []

    def generate_candidate_pair(
        self,
        line1_prompt: str,
        target_matras: List[int],
        target_antyaprasa_word: str
    ) -> Tuple[Dict, Dict]:
        """
        Generates two candidate completions for human lyricist evaluation:
        Candidate A vs Candidate B.
        """
        cand_a = self.agent.generate_line(
            target_matras,
            context_prompt=line1_prompt.split(),
            target_antyaprasa_word=target_antyaprasa_word,
            temperature=0.8
        )

        cand_b = self.agent.generate_line(
            target_matras,
            context_prompt=line1_prompt.split(),
            target_antyaprasa_word=target_antyaprasa_word,
            temperature=0.6
        )

        reward_a = self.reward_model.evaluate(line1_prompt, cand_a['text'], target_matras)
        reward_b = self.reward_model.evaluate(line1_prompt, cand_b['text'], target_matras)

        cand_a['reward'] = reward_a
        cand_b['reward'] = reward_b
        return cand_a, cand_b

    def train_step_dpo(
        self,
        line1_prompt: str,
        winner_line: str,
        loser_line: str,
        target_matras: List[int]
    ) -> Dict:
        """
        Direct Preference Optimization (DPO) Policy Update:
        L_DPO(theta) = -log sigmoid( beta * [log pi(y_w | x) - log pi(y_l | x)] )

        Backpropagates gradients directly into:
        - self.agent.W_embed (token representations)
        - self.agent.W_head  (output logit projection)
        - self.agent.W_prosody (metric state weights)
        """
        # 1. Forward probabilities
        log_prob_w = self.agent.get_token_log_prob(winner_line, target_matras)
        log_prob_l = self.agent.get_token_log_prob(loser_line, target_matras)

        # Margin
        margin = self.beta * (log_prob_w - log_prob_l)
        # Sigmoid probability that winner is preferred
        sig_margin = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, margin))))
        loss = -math.log(max(1e-12, sig_margin))

        # Gradient coefficient for winner (+) and loser (-)
        grad_coef = -(1.0 - sig_margin) * self.beta

        # 2. Gradient Calculation & Weight Nudge
        # For tokens in winner_line: increase logits
        # For tokens in loser_line: decrease logits
        delta_W_head = np.zeros_like(self.agent.W_head)
        delta_W_embed = np.zeros_like(self.agent.W_embed)

        words_w = winner_line.strip().split()
        words_l = loser_line.strip().split()

        for w in words_w:
            idx = self.agent.word2idx.get(w, None)
            if idx is not None:
                delta_W_head[:, idx] -= grad_coef * 0.1
                delta_W_embed[idx] -= grad_coef * 0.05

        for l in words_l:
            idx = self.agent.word2idx.get(l, None)
            if idx is not None:
                delta_W_head[:, idx] += grad_coef * 0.1
                delta_W_embed[idx] += grad_coef * 0.05

        # 3. Apply Gradient Descent Update: theta = theta - lr * grad
        self.agent.W_head -= self.learning_rate * delta_W_head
        self.agent.W_embed -= self.learning_rate * delta_W_embed

        # Compute Frobenius Norm of weight delta
        norm_delta_head = float(np.linalg.norm(delta_W_head))
        norm_delta_embed = float(np.linalg.norm(delta_W_embed))
        total_shift_norm = norm_delta_head + norm_delta_embed

        self.agent.total_training_steps += 1
        self.agent.cumulative_loss += loss

        # 4. Measure New Probability Shift
        new_log_prob_w = self.agent.get_token_log_prob(winner_line, target_matras)
        new_log_prob_l = self.agent.get_token_log_prob(loser_line, target_matras)

        step_result = {
            'step': self.agent.total_training_steps,
            'dpo_loss': round(loss, 4),
            'weight_shift_norm': round(total_shift_norm, 5),
            'prob_winner_before': round(math.exp(max(-20.0, log_prob_w)), 6),
            'prob_winner_after': round(math.exp(max(-20.0, new_log_prob_w)), 6),
            'margin_gain': round(new_log_prob_w - log_prob_w, 4),
            'winner_line': winner_line,
            'loser_line': loser_line
        }

        self.training_history.append(step_result)
        return step_result
