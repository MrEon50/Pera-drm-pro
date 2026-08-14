"""
Automatyczny Pipeline Treningowy dla Bota Gomoku w PERA-DRM-PRO
Obsługuje konfigurowalne poziomy trudności (Początkujący, Średniozaawansowany, Zaawansowany, Własny)
oraz szczegółowy raport z przebiegu każdej partii.
"""

import os
import time
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from pera_drm_pro import (
    create_game_descriptor,
    PERANet,
    DRMSystem,
    EnhancedPUCT
)
from gomoku_tactical_analyzer import GomokuTacticalAnalyzer, PatternType
from json_memory_manager import JSONMemoryManager


def pretrain_gomoku_bot(
    num_games: int = 20,
    num_simulations_per_move: int = 20,
    difficulty_name: str = "Średniozaawansowany"
):
    print("=" * 85)
    print(f"🏋️ TRENING AI GOMOKU | Poziom: {difficulty_name.upper()} | Parti: {num_games} | MCTS Symulacje: {num_simulations_per_move}")
    print("=" * 85)

    descriptor = create_game_descriptor("gomoku_15")
    memory_manager = JSONMemoryManager(storage_dir="./game_memories")
    memory_file = memory_manager.get_game_filepath("gomoku_interactive")
    weights_file = "./game_memories/gomoku_peranet_weights.pt"

    # 1. Inicjalizacja Modelu i Optymalizatora
    model = PERANet("small", descriptor.planes, descriptor.action_space, descriptor.board_size)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if os.path.exists(weights_file):
        try:
            model.load_state_dict(torch.load(weights_file, weights_only=True))
            print(f"🧠 [PyTorch] Wczytano istniejące wagi sieci z: {weights_file}")
        except Exception as e:
            print(f"⚠️ Nie udało się wczytać wag: {e}")

    # 2. Inicjalizacja DRM
    try:
        drm, stats = memory_manager.load_memory(memory_file, DRMSystem)
        print(f"📖 [DRM] Wczytano istniejącą bazę reguł z pliku JSON ({len(drm.rules)} reguł)")
    except Exception:
        drm = DRMSystem(frz=0.4, seed=42)
        drm.add_rule("Podwójna Trójka (Atak)", features=[0.9, 0.3, 0.8], polarity=0.9, weight=2.5, tags={"atak"})
        drm.add_rule("Blok Otwartym Czwórkom", features=[0.3, 0.9, 0.8], polarity=-0.8, weight=2.8, tags={"obrona"})
        drm.add_rule("Przekątny Ciąg 4", features=[0.8, 0.8, 0.7], polarity=0.7, weight=2.0, tags={"atak"})
        stats = {"games_played": 0, "human_wins": 0, "bot_wins": 0}

    tactical_guard = GomokuTacticalAnalyzer(board_size=15)
    puct = EnhancedPUCT(model, drm, descriptor)

    start_time = time.time()
    bot1_wins = 0
    bot2_wins = 0
    draws = 0

    print("-" * 85)

    # 3. Pętla Treningowa
    for game_idx in range(1, num_games + 1):
        game_start = time.time()
        board_grid = np.zeros((15, 15), dtype=int)
        game_history = []
        current_player = 1  # 1: Czarny (Bot 1), 2: Biały (Bot 2)
        moves_count = 0
        winner = 0

        while moves_count < 80:  # Limit ruchów
            valid_moves = [(r, c) for r in range(15) for c in range(15) if board_grid[r, c] == 0]
            if not valid_moves:
                break

            # KROK A: Sprawdź Strażnika Taktyki (General Gomoku)
            tactical_override = tactical_guard.get_tactical_override(board_grid, current_player)

            if tactical_override is not None:
                (r, c), _ = tactical_override
            else:
                # KROK B: Symulacja MCTS + PERANet + DRM
                plane_p1 = (board_grid == current_player).astype(np.float32)
                plane_p2 = (board_grid == (3 - current_player)).astype(np.float32)
                plane_empty = (board_grid == 0).astype(np.float32)

                tensor_3d = np.stack([plane_p1, plane_p2, plane_empty], axis=0)
                board_tensor = torch.tensor(tensor_3d, dtype=torch.float32).unsqueeze(0)

                root_node = puct.run_simulation(board_tensor, num_simulations=num_simulations_per_move)
                best_idx = max(root_node.children, key=lambda m: root_node.children[m].n)
                r, c = best_idx // 15, best_idx % 15

                if board_grid[r, c] != 0:
                    r, c = valid_moves[random.randint(0, len(valid_moves) - 1)]

            board_grid[r, c] = current_player
            game_history.append((board_grid.copy(), current_player, r, c))
            moves_count += 1

            # Sprawdzenie wygranej po ruchu
            patterns = tactical_guard.analyze_move(board_grid, r, c, current_player)
            if PatternType.WIN_5 in patterns:
                winner = current_player
                break

            current_player = 3 - current_player

        if winner == 1:
            bot1_wins += 1
            winner_str = "Bot 1 (Czarny)"
        elif winner == 2:
            bot2_wins += 1
            winner_str = "Bot 2 (Biały)"
        else:
            draws += 1
            winner_str = "Remis / Limit"

        # 4. Aktualizacja Wag Sieci Neuronowej (Backpropagation)
        model.train()
        total_loss_accum = 0.0
        sample_count = min(10, len(game_history))

        for grid_state, player, r, c in game_history[-sample_count:]:
            plane_p1 = (grid_state == player).astype(np.float32)
            plane_p2 = (grid_state == (3 - player)).astype(np.float32)
            plane_empty = (grid_state == 0).astype(np.float32)

            tensor_3d = np.stack([plane_p1, plane_p2, plane_empty], axis=0)
            inp_tensor = torch.tensor(tensor_3d, dtype=torch.float32).unsqueeze(0)
            game_vec = descriptor.to_tensor(inp_tensor.device)

            target_action = torch.tensor([r * 15 + c], dtype=torch.long)
            target_value = torch.tensor([[1.0 if player == winner else (-1.0 if winner != 0 else 0.0)]], dtype=torch.float32)

            optimizer.zero_grad()
            pol_logits, val_pred = model(inp_tensor, game_vec)

            loss_policy = F.cross_entropy(pol_logits, target_action)
            loss_value = F.mse_loss(val_pred, target_value)
            total_loss = loss_policy + loss_value

            total_loss.backward()
            optimizer.step()
            total_loss_accum += total_loss.item()

        model.eval()
        avg_loss = total_loss_accum / max(1, sample_count)

        # 5. Aktualizacja DRM i Governora
        for rule in drm.get_active_rules():
            rule.observe(success=(winner != 0))

        drm.step(external_reward=0.2 if winner != 0 else -0.1)
        phi = drm.governor.calculate_system_harmony(drm.rules)
        stats["games_played"] += 1

        game_elapsed = time.time() - game_start

        # RAPORT DLA KAŻDEJ PARTII
        active_count = len(drm.get_active_rules())
        quarantined_count = len(drm.rules) - active_count
        print(f"🎮 Partia {game_idx:2d}/{num_games} | Wynik: {winner_str:15s} | Ruchów: {moves_count:2d} | Loss: {avg_loss:.4f} | Phi: {phi:+.3f} | DRM Reguły: {active_count} akt / {quarantined_count} kwar | Czas: {game_elapsed:.2f}s")

    # 6. Zapisywanie Wykształconego Modelu i Bazy JSON
    print("-" * 85)
    os.makedirs("./game_memories", exist_ok=True)
    torch.save(model.state_dict(), weights_file)
    print(f"💾 [PyTorch] Zapisano wykształcone wagi sieci do: {weights_file}")

    memory_manager.save_memory(drm, stats, filepath=memory_file, game_name="gomoku_interactive")
    print(f"💾 [DRM JSON] Zapisano zaktualizowaną bazę wiedzy DRM do: {memory_file}")

    elapsed = time.time() - start_time
    print(f"\n📊 PODSUMOWANIE TRENINGU ({difficulty_name.upper()}):")
    print(f"  • Razem rozegranych gier: {num_games}")
    print(f"  • Wygrane Bot 1 (Czarny): {bot1_wins} ({bot1_wins/num_games:.0%})")
    print(f"  • Wygrane Bot 2 (Biały) : {bot2_wins} ({bot2_wins/num_games:.0%})")
    print(f"  • Remisy                : {draws}")
    print(f"  • Całkowity czas treningu: {elapsed:.1f} sekund")
    print("=" * 85)


if __name__ == "__main__":
    pretrain_gomoku_bot(num_games=10, num_simulations_per_move=10, difficulty_name="Szybki Test")
