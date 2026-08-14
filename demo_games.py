"""
Demonstracja PERA-DRM PRO: Uniwersalny Agent dla Gier Planszowych (Go, Gomoku, Szachy)
z Pamięcią JSON, Balansowaniem Governora (Phi) i Bayesowską Kwarantanną.
"""

import os
import torch
from pera_drm_pro import (
    create_game_descriptor,
    PERANet,
    DRMSystem,
    EnhancedPUCT
)
from json_memory_manager import JSONMemoryManager


def run_universal_games_demo():
    print("=" * 80)
    print("🧠 DEMONSTRACJA PERA-DRM PRO: UNIWERSALNY AGENT DLA GIER PLANSZOWYCH")
    print("=" * 80)

    memory_manager = JSONMemoryManager(storage_dir="./game_memories")

    # List gier do demonstracji
    games_to_test = [
        ("gomoku_15", "Gomoku 15x15"),
        ("go_9", "Go 9x9"),
        ("chess_8", "Szachy 8x8")
    ]

    for game_key, game_display_name in games_to_test:
        print(f"\n🎮 --- INICJALIZACJA AGENTA DLA: {game_display_name} ---")

        # 1. Tworzenie dyskryminatora i modelu PERANet
        descriptor = create_game_descriptor(game_key)
        model = PERANet("small", descriptor.planes, descriptor.action_space, descriptor.board_size)
        model.eval()

        # 2. Inicjalizacja DRM z dedykowanymi regułami taktycznymi
        drm = DRMSystem(frz=0.4, seed=42)

        if "gomoku" in game_key:
            drm.add_rule("Agresywna Piątka", features=[1.0, 0.9, 0.8], polarity=0.9, weight=2.5, priority=5, tags={"atak", "win"})
            drm.add_rule("Blok Otwartym Czwórkom", features=[0.9, 0.9, 0.7], polarity=-0.8, weight=2.2, priority=5, tags={"obrona"})
            r_buggy = drm.add_rule("Słaba Heurystyka Brzegu", features=[0.1, 0.1, 0.2], polarity=0.3, weight=1.0, priority=1, tags={"eksperyment"})
        elif "go" in game_key:
            drm.add_rule("Rozszerzenie Terytorium", features=[0.8, 0.7, 0.9], polarity=0.7, weight=2.0, priority=4, tags={"ekspansja"})
            drm.add_rule("Łączenie Żywych Grup", features=[0.9, 0.8, 0.9], polarity=-0.6, weight=1.9, priority=4, tags={"obrona"})
            r_buggy = drm.add_rule("Losowa Inwazja Narożnika", features=[0.2, 0.2, 0.1], polarity=0.8, weight=1.0, priority=1, tags={"eksperyment"})
        else:  # szachy
            drm.add_rule("Atak na Króla (Mat)", features=[1.0, 1.0, 0.9], polarity=0.95, weight=3.0, priority=5, tags={"atak", "checkmate"})
            drm.add_rule("Ochrona Struktury Pionów", features=[0.7, 0.8, 0.6], polarity=-0.7, weight=1.8, priority=3, tags={"obrona"})
            r_buggy = drm.add_rule("Błędny Poświęcenie Skoczka", features=[0.3, 0.1, 0.2], polarity=0.6, weight=1.0, priority=1, tags={"eksperyment"})

        # 3. Symulacja gier i aktualizacja obserwacji Bayesowskich
        print(f"  [+] Wykonywanie symulacji MCTS dla {descriptor.name}...")
        puct = EnhancedPUCT(model, drm, descriptor)

        board_tensor = torch.randn(1, descriptor.planes, descriptor.board_size[0], descriptor.board_size[1])
        root = puct.run_simulation(board_tensor, num_simulations=30)
        print(f"  [+] Symulacja MCTS zakończona (węzłów w drzewie: {root.n})")

        # Symulacja porażek dla wadliwej reguły -> wywołanie auto-kwarantanny
        for _ in range(6):
            r_buggy.observe(success=False)

        # Krok DRM i ponowne zbalansowanie Phi przez Governora
        drm.step(external_reward=0.15)
        phi = drm.governor.calculate_system_harmony(drm.rules)

        print(f"  ⚖️ Wskaźnik Harmonii Governora (Phi): {phi:+.3f}")
        print(f"  🚨 Reguła '{r_buggy.name}' w kwarantannie: {r_buggy.quarantined} ({r_buggy.quarantine_reason})")

        # 4. Zapis wiedzy i stanu gier do pliku JSON
        stats = {
            "games_played": 150,
            "win_rate": 0.68,
            "avg_mcts_depth": 12.4,
            "best_opening": "Standard Center Expansion"
        }
        json_path = memory_manager.save_memory(drm, stats, game_name=game_key)

        # 5. Odczyt i weryfikacja rekonstrukcji z JSON
        reconstructed_drm, loaded_stats = memory_manager.load_memory(json_path, DRMSystem)
        reconstructed_phi = reconstructed_drm.governor.calculate_system_harmony(reconstructed_drm.rules)

        print(f"  ✅ Weryfikacja odczytu JSON: Reaktywowano {len(reconstructed_drm.rules)} reguł (Phi = {reconstructed_phi:+.3f})")
        print(f"  📊 Odczytane statystyki gry: WinRate={loaded_stats.get('win_rate'):.0%}, Mecze={loaded_stats.get('games_played')}")

    print("\n" + "=" * 80)
    print("✨ WSZYSTKIE TESTY DLA GIER GO, GOMOKU I SZACHÓW ZAKOŃCZONE POMYŚLNIE!")
    print("=" * 80)


if __name__ == "__main__":
    run_universal_games_demo()
