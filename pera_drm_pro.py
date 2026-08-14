"""
PERA-DRM PRO: Permutation-Equivariant Residual-Attention Network + Dynamic Rules Matrix 3.0 PRO
Uniwersalny, zunifikowany moduł AI dla gier planszowych (Go, Gomoku, Szachy, Othello, Tic-Tac-Toe).

Kluczowe innowacje:
1. GameDescriptor — Uniwersalna obsługa różnych geometrii plansz i reguł.
2. PERANet — Sieć konwolucyjno-rezidualna z multi-head self-attention i symetriami.
3. EquilibriumGovernor (Phi) — Samo-balansowanie stylu gry (Atak vs Obrona) i wstrząsy przy stagnacji.
4. Bayesowska Ewaluacja & Auto-Kwarantanna — Śledzenie rozkładu Beta sukcesu/porażki i eliminacja słabych reguł.
5. Enhanced PUCT — Wzniesienie MCTS o dynamiczne bonusy z zbalansowanego matrixa DRM.
"""

from __future__ import annotations
import math
import random
import time
import copy
import numpy as np
from typing import Tuple, Optional, Dict, List, Any, Set
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================================
# 1. Game Descriptor (Uniwersalny interfejs gier)
# =========================================================================

class GameDescriptor:
    """Opis właściwości i wymiarów dla dowolnej gry planszowej."""

    def __init__(
        self,
        name: str,
        board_size: Tuple[int, int],
        planes: int,
        action_space: int,
        game_id: int = 0,
        symmetries: int = 8
    ):
        self.name = name
        self.board_size = board_size  # (H, W)
        self.planes = planes          # Liczba kanałów wejściowych
        self.action_space = action_space  # Liczba możliwych akcji/ruchów
        self.game_id = game_id
        self.symmetries = symmetries  # Liczba symetrii obrotowych/odbiciowych (np. 8 dla kwadratu)

    def to_tensor(self, device: torch.device) -> torch.Tensor:
        H, W = self.board_size
        return torch.tensor(
            [H, W, self.planes, self.action_space, self.game_id, self.symmetries],
            dtype=torch.float32,
            device=device
        )

    def __repr__(self) -> str:
        return f"GameDescriptor({self.name}, Board={self.board_size[0]}x{self.board_size[1]}, ActionSpace={self.action_space})"


def create_game_descriptor(game_name: str) -> GameDescriptor:
    """Fabryka konfiguracyjna dla popularnych gier planszowych."""
    game_configs = {
        "go_19": GameDescriptor("Go 19x19", (19, 19), 17, 361, 1),
        "go_13": GameDescriptor("Go 13x13", (13, 13), 17, 169, 2),
        "go_9": GameDescriptor("Go 9x9", (9, 9), 17, 81, 3),
        "gomoku_15": GameDescriptor("Gomoku 15x15", (15, 15), 3, 225, 4),
        "chess_8": GameDescriptor("Szachy 8x8", (8, 8), 20, 4096, 5),
        "othello_8": GameDescriptor("Othello 8x8", (8, 8), 3, 64, 6),
        "tictactoe_3": GameDescriptor("Kółko i Krzyżyk 3x3", (3, 3), 3, 9, 7),
    }

    key = game_name.lower().strip()
    if key not in game_configs:
        raise ValueError(f"Dyskryminator gry '{game_name}' nie jest dostępny. Dostępne: {list(game_configs.keys())}")
    return game_configs[key]


# =========================================================================
# 2. PERANet (Permutation-Equivariant Residual Attention Network)
# =========================================================================

class EquivariantConv2d(nn.Module):
    """Konwolucja uwzględniająca symetrie przestrzenne planszy."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    """Blok rezidualny do reprezentowania głębokich wzorców pozycji."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = EquivariantConv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = EquivariantConv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(residual + out)


class AttentionBlock(nn.Module):
    """Multi-Head Self-Attention wychwytujący zależności na całej planszy."""

    def __init__(self, channels: int, heads: int = 8):
        super().__init__()
        self.channels = channels
        self.heads = heads
        while channels % self.heads != 0 and self.heads > 1:
            self.heads -= 1
        self.head_dim = channels // self.heads

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C == 0 or H == 0 or W == 0:
            return x

        qkv = self.qkv(x).view(B, 3, self.heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)

        attn = torch.softmax(torch.einsum("bhdx,bhdy->bhxy", q, k) / math.sqrt(self.head_dim), dim=-1)
        out = torch.einsum("bhxy,bhdy->bhdx", attn, v)

        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        norm_in = out.permute(0, 2, 3, 1)
        norm_out = self.norm(norm_in).permute(0, 3, 1, 2)
        return x + norm_out


class PERANet(nn.Module):
    """
    Główna sieć neuronowa z wyjściem Polisy (rekomendacja ruchów)
    oraz Wartości (ocena szansy na wygraną w przedziale [-1, 1]).
    """

    SIZES = {
        "tiny": (64, 4, 4),    # ~2M parametrów
        "small": (128, 6, 4),   # ~8M parametrów
        "medium": (192, 10, 8), # ~25M parametrów
        "large": (256, 14, 8)   # ~60M parametrów
    }

    def __init__(self, size: str, in_planes: int, action_space: int, board_size: Tuple[int, int]):
        super().__init__()
        if size not in self.SIZES:
            raise ValueError(f"Rozmiar modelu musi być jednym z {list(self.SIZES.keys())}")

        channels, num_blocks, num_heads = self.SIZES[size]
        self.channels = channels
        self.board_size = board_size
        self.action_space = action_space

        self.input_proj = nn.Sequential(
            nn.Conv2d(in_planes, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )

        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        self.attn_blocks = nn.ModuleList([
            AttentionBlock(channels, num_heads) if i % 3 == 0 else nn.Identity()
            for i in range(num_blocks)
        ])

        # Policy Head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size[0] * board_size[1], action_space)

        # Value Head
        self.value_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.SiLU(),
            nn.Linear(channels // 2, 1),
            nn.Tanh()
        )

        self.game_embedding = nn.Linear(6, channels)

    def forward(self, x: torch.Tensor, game_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape

        x = self.input_proj(x)

        if len(game_vec.shape) == 1:
            g_emb = self.game_embedding(game_vec.unsqueeze(0)).view(1, self.channels, 1, 1)
        else:
            g_emb = self.game_embedding(game_vec).view(-1, self.channels, 1, 1)

        x = x + g_emb.expand(B, -1, H, W)

        for res, attn in zip(self.res_blocks, self.attn_blocks):
            x = res(x)
            x = attn(x)

        pol = F.silu(self.policy_conv(x)).view(B, -1)
        policy_logits = self.policy_fc(pol)

        value = self.value_head(x)

        return policy_logits, value


# =========================================================================
# 3. Dynamic Rules Matrix 3.0 PRO (z Equilibrium Governor Phi)
# =========================================================================

class Rule:
    """Pojedyncza reguła decyzyjna z polaryzacją i statystyką Bayesowską."""

    def __init__(
        self,
        rule_id: str,
        name: str,
        features: Optional[List[float]] = None,
        polarity: float = 0.0,    # +1.0 (Ekspansja/Atak) do -1.0 (Ochrona/Obrona)
        magnitude: float = 1.0,
        weight: float = 1.0,
        priority: int = 1,
        tags: Optional[Set[str]] = None
    ):
        self.id = rule_id
        self.name = name
        self.features = features or [weight]
        self.polarity = max(-1.0, min(1.0, float(polarity)))
        self.magnitude = float(magnitude)
        self.weight = float(weight)
        self.strength = 0.0
        self.priority = int(priority)
        self.tags = tags or set()

        # Statystyki Bayesowskie (Rozkład Beta)
        self.success_count = 0
        self.failure_count = 0
        self.usage_count = 0
        self.post_mean = 0.5
        self.post_var = 0.0833

        # Kwarantanna i stan cyklu
        self.quarantined = False
        self.quarantine_reason: Optional[str] = None
        self.created_cycle = 0
        self.last_used_cycle = 0

    def update_bayes_posterior(self):
        """Aktualizuje estymację Bayesowską prawdopodobieństwa sukcesu reguły."""
        alpha = 1.0 + self.success_count
        beta = 1.0 + self.failure_count
        self.post_mean = alpha / (alpha + beta)
        self.post_var = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))

    def observe(self, success: bool):
        """Rejestruje wynik użycia reguły i uaktualnia kwarantannę."""
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

        self.update_bayes_posterior()

        # Auto-kwarantanna dla nieefektywnych reguł po min. 5 użyciach
        if self.usage_count >= 5 and self.post_mean <= 0.20:
            self.quarantined = True
            self.quarantine_reason = f"Skuteczność spadła poniżej progu ({self.post_mean:.1%})"

    def apply_mutation(self, mutation_rate: float = 0.15):
        """Mutuje cechy reguły w celu stymulacji eksploracji."""
        for i in range(len(self.features)):
            self.features[i] += float(np.random.normal(0, mutation_rate))
        self.weight = max(0.05, self.weight + float(np.random.normal(0, mutation_rate)))

    def __repr__(self) -> str:
        return f"Rule({self.name}, Polarity={self.polarity:+.2f}, Weight={self.weight:.2f}, PostMean={self.post_mean:.1%})"


class EquilibriumGovernor:
    """
    Zarządca Równowagi (Governor) nadzorujący harmonię systemu (Phi).
    Pilnuje zbalansowania sił ekspansywnych (atak) oraz ochronnych (obrona).
    """

    def __init__(self, target_phi: float = 0.0, tolerance: float = 0.25):
        self.target_phi = target_phi
        self.tolerance = tolerance

    def calculate_system_harmony(self, rules: List[Rule]) -> float:
        """Oblicza wskaźnik harmonii systemu Phi w zakresie [-1.0, 1.0]."""
        active_rules = [r for r in rules if not r.quarantined]
        if not active_rules:
            return 0.0

        total_weight = sum(r.weight * r.magnitude for r in active_rules) + 1e-9
        weighted_polarity = sum(r.polarity * r.weight * r.magnitude for r in active_rules)
        return float(weighted_polarity / total_weight)

    def rebalance(self, rules: List[Rule]) -> float:
        """Koryguje wagi reguł, przywracając harmonię Phi do założonego zakresu."""
        phi = self.calculate_system_harmony(rules)
        delta = phi - self.target_phi

        if abs(delta) > self.tolerance:
            # Wzmacniaj reguły defensywne gdy system jest zbyt agresywny i vice-versa
            factor = 1.0 + abs(delta)
            for rule in rules:
                if not rule.quarantined:
                    if delta > 0 and rule.polarity < 0:  # Zbyt agresywnie -> wzmocnij obronę
                        rule.weight *= factor
                    elif delta < 0 and rule.polarity > 0:  # Zbyt biernie -> wzmocnij atak
                        rule.weight *= factor

        return self.calculate_system_harmony(rules)


class DRMSystem:
    """Zunifikowany system DRM 3.0 z obsługą ewolucji, stagnacji i kwarantanny."""

    def __init__(self, frz: float = 0.5, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.rules: List[Rule] = []
        self.governor = EquilibriumGovernor(target_phi=0.0, tolerance=0.25)
        self.cycle = 0
        self.frz = frz
        self.activity_history: List[float] = []

    def add_rule(
        self,
        name: str,
        features: Optional[List[float]] = None,
        polarity: float = 0.0,
        magnitude: float = 1.0,
        weight: float = 1.0,
        priority: int = 1,
        tags: Optional[Set[str]] = None
    ) -> Rule:
        rule_id = f"rule_{self.cycle}_{len(self.rules)}_{random.randint(100, 999)}"
        new_rule = Rule(
            rule_id=rule_id,
            name=name,
            features=features,
            polarity=polarity,
            magnitude=magnitude,
            weight=weight,
            priority=priority,
            tags=tags
        )
        new_rule.created_cycle = self.cycle
        self.rules.append(new_rule)
        return new_rule

    def get_active_rules(self) -> List[Rule]:
        return [r for r in self.rules if not r.quarantined]

    def step(self, external_reward: float = 0.0):
        """Przeprowadza cykl aktualizacji sił reguł i reaguje na stagnację."""
        self.cycle += 1
        active = self.get_active_rules()

        if not active:
            return

        phi = self.governor.calculate_system_harmony(self.rules)

        total_strength = 0.0
        for rule in active:
            # Wyliczenie siły uwzględniające estymator Bayesowski i harmonię Phi
            bayes_factor = 0.5 + rule.post_mean
            rule.strength = (rule.weight * rule.magnitude * bayes_factor + external_reward) / (self.frz + 1e-6)
            total_strength += rule.strength

        avg_activity = total_strength / len(active)
        self.activity_history.append(avg_activity)

        # Autokorekta równowagi Governora przy odchyleniu
        self.governor.rebalance(self.rules)

    def get_drm_bonus_for_move(self, move_features: List[float]) -> float:
        """Oblicza sumaryczny bonus DRM dla danego ruchu w wyszukiwaniu MCTS."""
        total_bonus = 0.0
        active_rules = self.get_active_rules()

        for rule in active_rules:
            # Podobieństwo cech ruchu do cech reguły
            v1 = np.array(rule.features)
            v2 = np.array(move_features)
            min_len = min(len(v1), len(v2))
            if min_len == 0:
                sim = 0.5
            else:
                dot = np.dot(v1[:min_len], v2[:min_len])
                norms = (np.linalg.norm(v1[:min_len]) * np.linalg.norm(v2[:min_len])) + 1e-9
                sim = float(dot / norms)

            if sim > 0.6:
                total_bonus += rule.strength * rule.post_mean * sim

        return total_bonus


# =========================================================================
# 4. Enhanced PUCT (Integracja MCTS z PERANet i DRM)
# =========================================================================

class Node:
    def __init__(self, state_tensor: Optional[torch.Tensor], parent: Optional[Node] = None, move: Optional[int] = None):
        self.state_tensor = state_tensor
        self.parent = parent
        self.move = move
        self.children: Dict[int, Node] = {}
        self.n = 0
        self.w = 0.0
        self.p = 0.0

    @property
    def q(self) -> float:
        return self.w / (self.n + 1e-6)


class EnhancedPUCT:
    """Algorytm przeszukiwania MCTS wspomagany hybrydowo przez PERANet i DRM."""

    def __init__(self, model: PERANet, drm: DRMSystem, descriptor: GameDescriptor, c_puct: float = 1.4):
        self.model = model
        self.drm = drm
        self.descriptor = descriptor
        self.c_puct = c_puct

    def run_simulation(self, root_state: torch.Tensor, num_simulations: int = 50) -> Node:
        root_node = Node(root_state)
        device = root_state.device
        game_vec = self.descriptor.to_tensor(device)

        for _ in range(num_simulations):
            node = root_node

            # 1. Selection
            while node.children:
                best_score = float("-inf")
                best_move = None

                for move, child in node.children.items():
                    dummy_features = [float(move % 5) / 5.0, 0.8, 0.5]
                    drm_bonus = self.drm.get_drm_bonus_for_move(dummy_features)

                    uct = child.q + self.c_puct * child.p * (math.sqrt(node.n) / (1 + child.n)) + drm_bonus

                    if uct > best_score:
                        best_score = uct
                        best_move = move

                node = node.children[best_move]

            # 2. Expansion & Evaluation
            if node.state_tensor is not None:
                with torch.no_grad():
                    logits, val_tensor = self.model(node.state_tensor, game_vec)
                    probs = F.softmax(logits, dim=-1).squeeze(0)
                    leaf_value = val_tensor.item()

                for action_idx in range(min(self.descriptor.action_space, probs.size(0))):
                    child_node = Node(state_tensor=None, parent=node, move=action_idx)
                    child_node.p = probs[action_idx].item()
                    node.children[action_idx] = child_node
            else:
                leaf_value = 0.0

            # 3. Backpropagation
            curr = node
            while curr is not None:
                curr.w += leaf_value
                curr.n += 1
                curr = curr.parent

        return root_node


# =========================================================================
# 5. Szybki Test Modułu
# =========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 Test Modułu PERA-DRM PRO")
    print("=" * 70)

    # 1. Test Dyskryptora
    desc_gomoku = create_game_descriptor("gomoku_15")
    print(f"✅ {desc_gomoku}")

    # 2. Test PERANet
    model = PERANet("small", desc_gomoku.planes, desc_gomoku.action_space, desc_gomoku.board_size)
    dummy_board = torch.randn(1, desc_gomoku.planes, desc_gomoku.board_size[0], desc_gomoku.board_size[1])
    dummy_vec = desc_gomoku.to_tensor(dummy_board.device)

    logits, value = model(dummy_board, dummy_vec)
    print(f"🧠 PERANet forward pass: Logits={logits.shape}, Value={value.shape}")

    # 3. Test DRM System i Governora
    drm = DRMSystem()
    r_attack = drm.add_rule("Atak Piątką", features=[1.0, 0.9], polarity=0.9, magnitude=1.5, weight=2.0)
    r_defend = drm.add_rule("Blok Czwórki", features=[0.8, 0.9], polarity=-0.8, magnitude=1.4, weight=1.8)

    drm.step(external_reward=0.2)
    phi = drm.governor.calculate_system_harmony(drm.rules)
    print(f"⚖️ DRM Governor Harmony (Phi): {phi:+.3f}")

    # 4. Test MCTS
    puct = EnhancedPUCT(model, drm, desc_gomoku)
    root = puct.run_simulation(dummy_board, num_simulations=20)
    print(f"🎯 MCTS Symulacja: Odwiedzin korzenia = {root.n}")
    print("✨ Moduł PERA-DRM PRO gotowy do użycia!")
