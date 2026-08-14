"""
Gra Interaktywna Gomoku 15x15 z AI PERA-DRM-PRO
Interaktywne Menu Główne, Komendy w Grze (m/q/r/save/help),
Konfigurowalny Trening AI (Początkujący -> Zaawansowany) oraz Pełne Logowanie Ruchów.
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
from pera_drm_pro import (
    create_game_descriptor,
    PERANet,
    DRMSystem,
    EnhancedPUCT
)
from json_memory_manager import JSONMemoryManager
from gomoku_tactical_analyzer import GomokuTacticalAnalyzer, PatternType
from train_bot import pretrain_gomoku_bot


class MoveRecord:
    """Rekord pojedynczego ruchu w logu śledzenia."""

    def __init__(
        self,
        move_num: int,
        player_name: str,
        row: int,
        col: int,
        patterns: List[str],
        decision_type: str,
        eval_v: Optional[float] = None,
        phi: Optional[float] = None
    ):
        self.move_num = move_num
        self.player_name = player_name
        self.row = row
        self.col = col
        self.patterns = patterns
        self.decision_type = decision_type
        self.eval_v = eval_v
        self.phi = phi
        self.timestamp = time.strftime("%H:%M:%S")

    def to_log_string(self) -> str:
        p_str = ", ".join(self.patterns) if self.patterns else "Zwykłe pozycjonowanie"
        v_str = f" | V: {self.eval_v:+.2f}" if self.eval_v is not None else ""
        phi_str = f" | Phi: {self.phi:+.3f}" if self.phi is not None else ""
        return (
            f"[{self.timestamp}] Ruch #{self.move_num:02d} | {self.player_name:13s} "
            f"-> Pole ({self.row:2d}, {self.col:2d}) | Typ: {self.decision_type:22s} | Wzorce: [{p_str}]{v_str}{phi_str}"
        )


class GomokuBoard:
    """Klasa obsługująca zasady i stan planszy Gomoku (15x15)."""

    def __init__(self, size: int = 15):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)  # 0: puste, 1: Gracz (X), 2: Bot (O)
        self.history: List[Tuple[int, int]] = []

    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.history.clear()

    def is_valid_move(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size and self.grid[row, col] == 0

    def make_move(self, row: int, col: int, player: int) -> bool:
        if self.is_valid_move(row, col):
            self.grid[row, col] = player
            self.history.append((row, col))
            return True
        return False

    def check_win(self, player: int) -> bool:
        """Sprawdza czy gracz ułożył ciąg 5 kamieni w poziomie, pionie lub po przekątnych."""
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.grid[r, c] != player:
                    continue
                for dr, dc in directions:
                    count = 1
                    nr, nc = r + dr, c + dc
                    while 0 <= nr < self.size and 0 <= nc < self.size and self.grid[nr, nc] == player:
                        count += 1
                        if count >= 5:
                            return True
                        nr += dr
                        nc += dc
        return False

    def is_full(self) -> bool:
        return np.all(self.grid != 0)

    def to_tensor(self) -> torch.Tensor:
        """Konwertuje planszę na 3-kanałowy tensor PyTorch (Planes: Gracz, Bot, Puste)."""
        plane_human = (self.grid == 1).astype(np.float32)
        plane_bot = (self.grid == 2).astype(np.float32)
        plane_empty = (self.grid == 0).astype(np.float32)

        tensor_3d = np.stack([plane_human, plane_bot, plane_empty], axis=0)
        return torch.tensor(tensor_3d, dtype=torch.float32).unsqueeze(0)

    def display(self):
        """Wyświetla czytelną planszę w konsoli."""
        cols_header = "    " + " ".join([f"{c:2d}" for c in range(self.size)])
        print(cols_header)
        print("   ┌" + "─" * (self.size * 3) + "┐")

        symbols = {0: " . ", 1: " X ", 2: " O "}
        for r in range(self.size):
            row_str = f"{r:2d} │"
            for c in range(self.size):
                row_str += symbols[self.grid[r, c]]
            row_str += "│"
            print(row_str)
        print("   └" + "─" * (self.size * 3) + "┘")


class InteractiveGomokuGame:
    """Główny kontroler gry z Menu Głównym, komendami w grze i opcjami treningowymi."""

    def __init__(self, board_size: int = 15):
        self.board = GomokuBoard(board_size)
        self.descriptor = create_game_descriptor("gomoku_15" if board_size == 15 else "go_9")
        self.memory_manager = JSONMemoryManager(storage_dir="./game_memories")
        self.memory_file = self.memory_manager.get_game_filepath("gomoku_interactive")
        self.weights_file = "./game_memories/gomoku_peranet_weights.pt"
        self.tactical_analyzer = GomokuTacticalAnalyzer(board_size=board_size)
        self.move_records: List[MoveRecord] = []

        self._reload_model_and_memory()

    def _reload_model_and_memory(self):
        """Wczytuje aktualne wagi sieci neuronowej oraz wiedzę DRM z plików."""
        self.model = PERANet("small", self.descriptor.planes, self.descriptor.action_space, self.descriptor.board_size)

        if os.path.exists(self.weights_file):
            try:
                self.model.load_state_dict(torch.load(self.weights_file, weights_only=True))
                print(f"🧠 [AI PERANet] Wczytano wagi sieci z: {self.weights_file}")
            except Exception as e:
                print(f"⚠️ Błąd ładowania wag: {e}")
        else:
            print("ℹ️ [AI PERANet] Brak wag sieci! Możesz użyć opcji Treningu w Menu.")

        self.model.eval()

        try:
            self.drm, self.stats = self.memory_manager.load_memory(self.memory_file, DRMSystem)
            print(f"📖 [AI DRM] Wczytano wiedzę bota z pliku JSON ({len(self.drm.rules)} reguł)!")
        except Exception:
            self.drm = DRMSystem(frz=0.4, seed=42)
            self.drm.add_rule("Podwójna Trójka (Atak)", features=[0.9, 0.3, 0.8], polarity=0.9, weight=2.5, tags={"atak"})
            self.drm.add_rule("Blok Otwartym Czwórkom", features=[0.3, 0.9, 0.8], polarity=-0.8, weight=2.8, tags={"obrona"})
            self.drm.add_rule("Przekątny Ciąg 4", features=[0.8, 0.8, 0.7], polarity=0.7, weight=2.0, tags={"atak"})
            self.stats = {"games_played": 0, "human_wins": 0, "bot_wins": 0}

        self.puct = EnhancedPUCT(self.model, self.drm, self.descriptor)

    def bot_think_and_move(self) -> Tuple[int, int, str, Optional[float], Optional[float]]:
        """Wylicza ruch bota z użyciem Strażnika Taktycznego lub MCTS+PERANet+DRM."""
        # 1. WARSTWA STRAŻNIKA TAKTYCZNEGO (General Gomoku)
        tactical_override = self.tactical_analyzer.get_tactical_override(self.board.grid, current_player=2)
        if tactical_override is not None:
            (tr, tc), reason = tactical_override
            phi = self.drm.governor.calculate_system_harmony(self.drm.rules)
            return tr, tc, f"🛡️ TAKTYCZNY ({reason})", 0.99, phi

        # 2. MCTS + PERANET + DRM
        board_tensor = self.board.to_tensor()
        game_vec = self.descriptor.to_tensor(board_tensor.device)

        with torch.no_grad():
            logits, val_tensor = self.model(board_tensor, game_vec)
            bot_val_eval = val_tensor.item()

        root_node = self.puct.run_simulation(board_tensor, num_simulations=40)

        best_move_idx = max(root_node.children, key=lambda m: root_node.children[m].n)
        row = best_move_idx // self.board.size
        col = best_move_idx % self.board.size

        if not self.board.is_valid_move(row, col):
            empty_cells = [(r, c) for r in range(self.board.size) for c in range(self.board.size) if self.board.is_valid_move(r, c)]
            row, col = empty_cells[0]

        phi = self.drm.governor.calculate_system_harmony(self.drm.rules)
        return row, col, "🧠 STRATEGICZNY (MCTS+DRM)", bot_val_eval, phi

    def print_help_commands(self):
        print("\n" + "=" * 70)
        print("💡 DOSTĘPNE KOMENDY I FORMAT WPROWADZANIA RUCHÓW:")
        print("=" * 70)
        print("  wiersz kolumna - Wykonaj ruch podając Wiersz (oś Y: 0-14) i Kolumnę (oś X: 0-14)")
        print("                   Przykład: '5 5' (środek-góra) lub '7 7' (sam środek planszy)")
        print("  m  / menu      - Wyjście do Menu Głównego")
        print("  r  / restart   - Rozpocznij aktualną partię od nowa")
        print("  s  / save      - Zapisz logi ruchów aktualnego meczu do pliku")
        print("  h  / help      - Pokaż ten spis komend")
        print("  q  / quit      - Zamknij aplikację")
        print("=" * 70 + "\n")

    def play_match(self):
        """Prowadzi pojedynczą rozgrywkę meczową."""
        self.board.reset()
        self.move_records.clear()

        print("\n" + "=" * 80)
        print("🎮 ROZPOCZĘTO MECZ GOMOKU 15x15 (Człowiek [X] vs PERA-DRM PRO AI [O])")
        print("=" * 80)
        print("📍 Format podawania ruchów: Wiersz(Y) Kolumna(X), np. '5 5' lub '7 7'")
        print("💡 Wpisz 'h' lub 'help' aby zobaczyć pełny spis komend.\n")

        self.board.display()

        while True:
            # -----------------------------------------------------------------
            # RUCH CZŁOWIEKA (X)
            # -----------------------------------------------------------------
            cmd = input("\n👉 Twój ruch [Wiersz(Y) Kolumna(X) np. 5 5 lub 7 7] (Gracz X): ").strip().lower()

            if cmd in ["m", "menu"]:
                print("↩️ Powrót do Menu Głównego...")
                break
            elif cmd in ["q", "quit"]:
                print("🚪 Zamykanie aplikacji...")
                sys.exit(0)
            elif cmd in ["r", "restart"]:
                print("🔄 Rozpoczynanie partii od nowa...")
                self.board.reset()
                self.move_records.clear()
                self.board.display()
                continue
            elif cmd in ["h", "help"]:
                self.print_help_commands()
                continue
            elif cmd in ["s", "save"]:
                self._export_game_logs_custom()
                continue

            parts = cmd.split()
            if len(parts) != 2:
                print("⚠️ Nieprawidłowy format! Podaj dwie liczby oznaczające współrzędne na osi Y (Wiersz 0-14) i osi X (Kolumna 0-14), np. '5 5' lub '7 7'!")
                continue

            try:
                r, c = int(parts[0]), int(parts[1])
            except ValueError:
                print("⚠️ Wprowadź liczby jako współrzędne, np. '5 5' lub '7 7'!")
                continue

            if not self.board.is_valid_move(r, c):
                print(f"❌ Nieprawidłowy ruch na polu ({r}, {c})! Upewnij się, że wiersz (0-14) i kolumna (0-14) są wolne, np. '5 5'.")
                continue

            # Analiza wzorców taktycznych człowieka
            human_patterns_enum = self.tactical_analyzer.analyze_move(self.board.grid, r, c, player=1)
            human_patterns_names = [p.value for p in human_patterns_enum]

            self.board.make_move(r, c, player=1)
            move_idx = len(self.move_records) + 1

            rec_human = MoveRecord(
                move_num=move_idx,
                player_name="Człowiek [X]",
                row=r,
                col=c,
                patterns=human_patterns_names,
                decision_type="Ruch Gracza"
            )
            self.move_records.append(rec_human)
            print(f"📝 {rec_human.to_log_string()}")

            self.board.display()

            if self.board.check_win(player=1):
                print("\n🎉 GRATULACJE! WYGRAŁEŚ Z AI PERA-DRM PRO!")
                self.stats["human_wins"] += 1
                self._update_and_save_learning(bot_won=False)
                self._display_full_move_log_summary()
                break

            if self.board.is_full():
                print("\n🤝 REMIS! Plansza została całkowicie zapełniona.")
                self._update_and_save_learning(bot_won=False)
                self._display_full_move_log_summary()
                break

            # -----------------------------------------------------------------
            # RUCH BOTA AI (O)
            # -----------------------------------------------------------------
            print("\n🤖 AI PERA-DRM PRO myśli...")
            br, bc, dec_type, bot_v, bot_phi = self.bot_think_and_move()

            bot_patterns_enum = self.tactical_analyzer.analyze_move(self.board.grid, br, bc, player=2)
            bot_patterns_names = [p.value for p in bot_patterns_enum]

            self.board.make_move(br, bc, player=2)
            move_idx = len(self.move_records) + 1

            rec_bot = MoveRecord(
                move_num=move_idx,
                player_name="AI Bot [O]",
                row=br,
                col=bc,
                patterns=bot_patterns_names,
                decision_type=dec_type,
                eval_v=bot_v,
                phi=bot_phi
            )
            self.move_records.append(rec_bot)
            print(f"📝 {rec_bot.to_log_string()}")

            self.board.display()

            if self.board.check_win(player=2):
                print("\n💀 AI PERA-DRM PRO WYGRAŁO MECZ!")
                self.stats["bot_wins"] += 1
                self._update_and_save_learning(bot_won=True)
                self._display_full_move_log_summary()
                break

            if self.board.is_full():
                print("\n🤝 REMIS! Plansza została całkowicie zapełniona.")
                self._update_and_save_learning(bot_won=False)
                self._display_full_move_log_summary()
                break

    def _export_game_logs_custom(self):
        """Zapisuje logi ruchów do podanego przez użytkownika pliku."""
        if not self.move_records:
            print("⚠️ Brak ruchów w aktualnej partii do zapisania!")
            return

        filename = input("💾 Podaj nazwę pliku dla logów (np. 'mój_mecz.log'): ").strip()
        if not filename:
            filename = f"game_log_{time.strftime('%Y%m%d_%H%M%S')}.log"
        if not filename.endswith(".log") and not filename.endswith(".txt"):
            filename += ".log"

        filepath = os.path.join("./game_memories", filename)
        os.makedirs("./game_memories", exist_ok=True)

        lines = [rec.to_log_string() for rec in self.move_records]
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"✅ Logi meczu zapisano pomyślnie w: {filepath}")

    def _display_full_move_log_summary(self):
        """Wyświetla i zapisuje pełny log śledzenia wszystkich ruchów w meczu."""
        print("\n" + "=" * 85)
        print("📜 PODSUMOWANIE RUCHÓW W MECZU (GAME MOVES TRANSCRIPT)")
        print("=" * 85)

        log_lines = []
        for rec in self.move_records:
            line_str = rec.to_log_string()
            print(line_str)
            log_lines.append(line_str)

        trace_file = "./game_memories/last_game_trace.log"
        os.makedirs("./game_memories", exist_ok=True)
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        print("=" * 85)
        print(f"💾 Ostatni mecz automatycznie zapisano w: {trace_file}")
        print("=" * 85)

    def _update_and_save_learning(self, bot_won: bool):
        """Aktualizuje statystyki Bayesowskie reguł i zapisuje zmienioną wiedzę w pliku JSON."""
        self.stats["games_played"] += 1

        for rule in self.drm.get_active_rules():
            rule.observe(success=bot_won)

        self.drm.step(external_reward=1.0 if bot_won else -0.5)

        saved_path = self.memory_manager.save_memory(self.drm, self.stats, filepath=self.memory_file, game_name="gomoku_interactive")
        print(f"💾 [Pamięć JSON] Wiedza bota została zaktualizowana i zapisana w: {saved_path}")

    def menu_training_options(self):
        """Podmenu wyboru poziomu trudności i parametrów treningu AI."""
        print("\n" + "=" * 70)
        print("🏋️ MENEDŻER TRENINGU AI (DOSTOSOWANIE POZIOMU TRUDNOŚCI)")
        print("=" * 70)
        print("1. 🟢 Początkujący      (10 gier, szybki MCTS = 10 symulacji)")
        print("2. 🟡 Średniozaawansowany (30 gier, zbalansowany MCTS = 25 symulacji)")
        print("3. 🔴 Zaawansowany / Ekspert (100 gier, intensywny MCTS = 50 symulacji)")
        print("4. ⚙️ Własne Ustawienia (Określ własną liczbę gier i symulacji MCTS)")
        print("0. ↩️ Powrót do Menu Głównego")
        print("=" * 70)

        choice = input("👉 Wybierz opcję (0-4): ").strip()

        if choice == "1":
            pretrain_gomoku_bot(num_games=10, num_simulations_per_move=10, difficulty_name="Początkujący")
            self._reload_model_and_memory()
        elif choice == "2":
            pretrain_gomoku_bot(num_games=30, num_simulations_per_move=25, difficulty_name="Średniozaawansowany")
            self._reload_model_and_memory()
        elif choice == "3":
            pretrain_gomoku_bot(num_games=100, num_simulations_per_move=50, difficulty_name="Zaawansowany / Ekspert")
            self._reload_model_and_memory()
        elif choice == "4":
            try:
                g = int(input("Wpisz liczbę gier do rozegrania (np. 50): "))
                s = int(input("Wpisz liczbę symulacji MCTS na ruch (np. 30): "))
                pretrain_gomoku_bot(num_games=max(1, g), num_simulations_per_move=max(5, s), difficulty_name="Własny Poziom")
                self._reload_model_and_memory()
            except ValueError:
                print("⚠️ Nieprawidłowe liczby! Trening anulowany.")
        elif choice == "0":
            return
        else:
            print("⚠️ Nieznana opcja!")

    def show_ai_stats(self):
        """Wyświetla aktualne statystyki bota, poziom harmonii i aktywne reguły DRM."""
        print("\n" + "=" * 70)
        print("📊 STATYSTYKI BOTA AI & BAZA WIEDZY DRM (ZAPISANA W JSON)")
        print("=" * 70)
        print(f"  • Rozegranych gier : {self.stats.get('games_played', 0)}")
        print(f"  • Wygrane Człowieka : {self.stats.get('human_wins', 0)}")
        print(f"  • Wygrane Bota AI  : {self.stats.get('bot_wins', 0)}")
        phi = self.drm.governor.calculate_system_harmony(self.drm.rules)
        print(f"  • Governor Phi    : {phi:+.3f} (Atak vs Obrona)")

        active_rules = self.drm.get_active_rules()
        print(f"\n  📜 Aktywne Reguły DRM ({len(active_rules)}):")
        for r in active_rules:
            print(f"    - {r.name:30s} | Polaryzacja: {r.polarity:+.1f} | Waga: {r.weight:.2f} | Estymacja Bayes: {r.post_mean:.1%}")

        quarantined = [r for r in self.drm.rules if r.quarantined]
        if quarantined:
            print(f"\n  🚨 Reguły w Kwarantannie ({len(quarantined)}):")
            for r in quarantined:
                print(f"    - {r.name:30s} | Powód: {r.quarantine_reason}")
        print("=" * 70)

    def main_menu(self):
        """Główna pętla Menu Aplikacji."""
        while True:
            print("\n" + "=" * 70)
            print("🎮 PERA-DRM PRO: GOMOKU AI - MENU GŁÓWNE")
            print("=" * 70)
            print("1. 🕹️ Zagraj z AI (Gomoku 15x15)")
            print("2. 🏋️ Trenuj AI (Wybor Poziomu: Początkujący -> Zaawansowany)")
            print("3. 📊 Wyświetl Statystyki Bota & Reguły DRM z JSON")
            print("4. 💾 Zapisz / Eksportuj Log Ostatniego Meczu")
            print("0. 🚪 Wyjście z gry")
            print("=" * 70)

            choice = input("👉 Wybierz opcję (0-4): ").strip()

            if choice == "1":
                self.play_match()
            elif choice == "2":
                self.menu_training_options()
            elif choice == "3":
                self.show_ai_stats()
            elif choice == "4":
                self._export_game_logs_custom()
            elif choice == "0":
                print("🚪 Dziękujemy za grę z PERA-DRM PRO! Do zobaczenia!")
                sys.exit(0)
            else:
                print("⚠️ Nieprawidłowa opcja, spróbuj ponownie.")


if __name__ == "__main__":
    game = InteractiveGomokuGame(board_size=15)
    game.main_menu()
