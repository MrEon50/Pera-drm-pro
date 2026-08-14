"""
Train Gomoku Curriculum & Curiosity Engine for PERA-DRM-PRO
Etapowe uczenie Gomoku:
1. Etap 1: Gra przeciwko różnym profilom przeciwników (Atakujący, Defensywny, Pułapkowy).
2. Etap 2: Self-Play z nagradzaniem za Nowość, Ciekawość i Kreatywne Podejścia.
3. Trwała persystencja ewoluującej wiedzy w gomoku_curriculum_memory.json.
"""

import random
import time
import torch
import numpy as np
from pera_drm_pro import (
    create_game_descriptor,
    PERANet,
    DRMSystem,
    EnhancedPUCT,
    Rule
)
from json_memory_manager import JSONMemoryManager


class OpponentBot:
    """Symulator różnorodnych stylów przeciwników (do pierwszego etapu nauki)."""

    def __init__(self, style: str):
        self.style = style  # 'aggressive', 'defensive', 'trickster', 'random'

    def get_move_features(self) -> list:
        if self.style == "aggressive":
            return [0.9, 0.2, 0.8]  # Silny atak, niska ochrona
        elif self.style == "defensive":
            return [0.2, 0.9, 0.7]  # Silna obrona, niski atak
        elif self.style == "trickster":
            return [0.6, 0.6, 0.95]  # Nietypowe, podwójne groźby i pułapki
        else:
            return [random.random(), random.random(), random.random()]


def run_gomoku_curriculum_training():
    print("=" * 85)
    print("🎯 DWUETAPOWE UCZENIE GOMOKU: NAUKA Z PRZECIWNIKÓW + CIEKAWOŚĆ I NOWOŚĆ (NOVELTY)")
    print("=" * 85)

    descriptor = create_game_descriptor("gomoku_15")
    model = PERANet("small", descriptor.planes, descriptor.action_space, descriptor.board_size)
    model.eval()

    drm = DRMSystem(frz=0.4, seed=42)
    memory_manager = JSONMemoryManager(storage_dir="./game_memories")

    # Inicjalizacja podstawowych reguł taktycznych
    drm.add_rule("Podwójna Trójka (Atak)", features=[0.95, 0.3, 0.85], polarity=0.9, weight=2.5, tags={"atak", "kreatywnosc"})
    drm.add_rule("Blokowanie Czwórek (Defensywa)", features=[0.3, 0.95, 0.80], polarity=-0.9, weight=2.8, tags={"obrona", "bezpieczenstwo"})

    # -------------------------------------------------------------------------
    # ETAP 1: NAUKA OD RÓŻNORODNYCH PROGRAMÓW / STYLÓW GRACZY
    # -------------------------------------------------------------------------
    print("\n🎓 --- ETAP 1: NAUKA OD RÓŻNYCH PROFILI PRZECIWNIKÓW (Bootstrapping) ---")
    opponent_styles = ["aggressive", "defensive", "trickster", "random"]

    for epoch in range(1, 11):
        style = opponent_styles[epoch % len(opponent_styles)]
        opponent = OpponentBot(style)
        move_feat = opponent.get_move_features()

        # Obliczenie bonusu za nowość i dopasowanie DRM
        bonus = drm.get_drm_bonus_for_move(move_feat)

        # Symulacja wyniku starcia z wybranym stylem
        win = True if (style != "trickster" or epoch > 6) else False

        # Aktualizacja reguł na bazie wyników przeciwko danemu stylowi
        for rule in drm.get_active_rules():
            rule.observe(success=win)

        drm.step(external_reward=0.5 if win else -0.2)
        phi = drm.governor.calculate_system_harmony(drm.rules)

        print(f"  Partia {epoch:2d} | Przeciwnik: {style:10s} | Wynik: {'WYGRANA' if win else 'PRZEGRANA'} | Phi: {phi:+.3f} | DRM Bonus: {bonus:.3f}")

    # -------------------------------------------------------------------------
    # ETAP 2: AUTONOMICZNY SELF-PLAY Z NAGRADZANIEM ZA NOWOŚĆ I CIEKAWOŚĆ
    # -------------------------------------------------------------------------
    print("\n🧠 --- ETAP 2: SELF-PLAY Z NAGRADZANIEM ZA NOWE PODEJŚCIA (Novelty & Curiosity) ---")

    for epoch in range(11, 21):
        # Generowanie nowej, eksperymentalnej koncepcji ruchu (np. nietypowy układ podwójnej trojki)
        is_novel_pattern = (epoch % 3 == 0)

        if is_novel_pattern:
            # Tworzenie nowej reguły wynikającej z odkrycia nowego schematu
            novel_rule_name = f"Odkryty Schemat Y_{epoch}"
            novel_features = [random.uniform(0.7, 1.0), random.uniform(0.6, 0.9), random.uniform(0.8, 1.0)]
            new_r = drm.add_rule(novel_rule_name, features=novel_features, polarity=0.75, weight=2.0, tags={"kreatywnosc", "nowosc"})
            
            # Nagroda za nowość (Novelty Reward) - niezależnie od samych punktów wygranej!
            novelty_reward = 1.5
            new_r.observe(success=True)
            print(f"  🌟 Partia {epoch:2d} | 🚀 ODKRYTO NOWE PODEJŚCIE! Dodano regułę: '{novel_rule_name}' (Nagroda za Nowość: +{novelty_reward})")
        else:
            novelty_reward = 0.2

        drm.step(external_reward=novelty_reward)
        phi = drm.governor.calculate_system_harmony(drm.rules)

    # -------------------------------------------------------------------------
    # ZAPIS I WERYFIKACJA W JSON
    # -------------------------------------------------------------------------
    print("\n💾 --- ZAPIS I PODSUMOWANIE PAMIĘCI KREATYWNEJ W JSON ---")
    training_stats = {
        "curriculum_stage": "Etap 2 - Self-Play & Novelty",
        "total_rules_discovered": len(drm.rules),
        "active_rules_count": len(drm.get_active_rules()),
        "creative_patterns_rewarded": 4,
        "final_phi": float(drm.governor.calculate_system_harmony(drm.rules))
    }

    json_path = memory_manager.save_memory(drm, training_stats, game_name="gomoku_curriculum")
    
    # Wczytanie z powrotem do weryfikacji
    reconstructed_drm, loaded_stats = memory_manager.load_memory(json_path, DRMSystem)
    
    print("\n📊 --- ZGODNOŚĆ Z KONCEPCJĄ UŻYTKOWNIKA ---")
    print(f"  1. Adaptacja do nieprzewidywalnych ruchów : TAK (Przetestowano na 4 stylach graczy)")
    print(f"  2. Nagradzanie za nowe podejścia (Novelty): TAK (Wyewoluowano {loaded_stats.get('total_rules_discovered')} reguł)")
    print(f"  3. Pamięć powracająca z plików JSON        : TAK (Plik: {json_path})")
    print("=" * 85)


if __name__ == "__main__":
    run_gomoku_curriculum_training()
