"""
Gomoku Tactical Pattern Analyzer (Integrator z General Gomoku)
Eksplicytna detekcja i priorytetyzacja wzorców taktycznych w Gomoku:
- WIN_5: Wygrana 5 w rzędzie
- THREAT_4_OPEN: Otwarty 4 (wymóg natychmiastowego zagrania/blokady)
- DOUBLE_THREAT_4: Podwójna czwórka
- FORK_4_3: Widelec 4+3
- THREAT_4_HALF: Półotwarty 4
- FORK_3_3: Widelec 3+3
- THREAT_3_OPEN: Otwarty 3
- BROKEN_3: Przerwany 3 (z dziurą)
"""

import numpy as np
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set


class PatternType(Enum):
    WIN_5 = "win_5"
    THREAT_4_OPEN = "threat_4_open"
    DOUBLE_THREAT_4 = "double_threat_4"
    FORK_4_3 = "fork_4_3"
    THREAT_4_HALF = "threat_4_half"
    FORK_3_3 = "fork_3_3"
    THREAT_3_OPEN = "threat_3_open"
    BROKEN_3 = "broken_3"
    EXTENSION_2 = "extension_2"


class GomokuTacticalAnalyzer:
    """
    Krytyczna warstwa taktyczna zapobiegająca ignorowaniu zasad i groźb w Gomoku.
    Gwarantuje, że AI nigdy nie przegapi natychmiastowej wygranej lub wymaganej blokady!
    """

    PATTERN_PRIORITIES = {
        PatternType.WIN_5: 1000000,
        PatternType.THREAT_4_OPEN: 200000,
        PatternType.DOUBLE_THREAT_4: 100000,
        PatternType.FORK_4_3: 50000,
        PatternType.THREAT_4_HALF: 20000,
        PatternType.FORK_3_3: 10000,
        PatternType.THREAT_3_OPEN: 5000,
        PatternType.BROKEN_3: 1000,
        PatternType.EXTENSION_2: 100
    }

    def __init__(self, board_size: int = 15):
        self.board_size = board_size

    def analyze_move(self, board_grid: np.ndarray, row: int, col: int, player: int) -> Set[PatternType]:
        """
        Analizuje jakie wzorce tworzy postawienie kamienia `player` (1: Gracz, 2: Bot) na polu (row, col).
        """
        if board_grid[row, col] != 0:
            return set()

        detected = set()
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # Zastąp pole na chwilę symulowanym kamieniem
        temp_grid = board_grid.copy()
        temp_grid[row, col] = player

        count_4_open = 0
        count_4_half = 0
        count_3_open = 0

        for dr, dc in directions:
            line_len, open_ends = self._check_line(temp_grid, row, col, dr, dc, player)

            if line_len >= 5:
                detected.add(PatternType.WIN_5)
            elif line_len == 4:
                if open_ends >= 2:
                    detected.add(PatternType.THREAT_4_OPEN)
                    count_4_open += 1
                elif open_ends == 1:
                    detected.add(PatternType.THREAT_4_HALF)
                    count_4_half += 1
            elif line_len == 3:
                if open_ends >= 2:
                    detected.add(PatternType.THREAT_3_OPEN)
                    count_3_open += 1
            elif line_len == 2:
                if open_ends >= 1:
                    detected.add(PatternType.EXTENSION_2)

        # Wykryj widelce i groźby podwójne
        if count_4_open >= 2 or (count_4_open >= 1 and count_4_half >= 1):
            detected.add(PatternType.DOUBLE_THREAT_4)
        if count_4_open >= 1 and count_3_open >= 1:
            detected.add(PatternType.FORK_4_3)
        if count_3_open >= 2:
            detected.add(PatternType.FORK_3_3)

        return detected

    def _check_line(self, grid: np.ndarray, row: int, col: int, dr: int, dc: int, player: int) -> Tuple[int, int]:
        """Liczy długość ciągłej linii oraz liczbę otwartych końców."""
        count = 1
        open_ends = 0

        # Sprawdź w stronę dodatnią (dr, dc)
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and grid[r, c] == player:
            count += 1
            r += dr
            c += dc
        if 0 <= r < self.board_size and 0 <= c < self.board_size and grid[r, c] == 0:
            open_ends += 1

        # Sprawdź w stronę ujemną (-dr, -dc)
        r, c = row - dr, col - dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and grid[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        if 0 <= r < self.board_size and 0 <= c < self.board_size and grid[r, c] == 0:
            open_ends += 1

        return count, open_ends

    def get_tactical_override(self, board_grid: np.ndarray, current_player: int) -> Optional[Tuple[Tuple[int, int], str]]:
        """
        Główny Strażnik Taktyki:
        1. Jeśli istnieje ruch dający 5-w-rzędzie dla current_player -> ZAGRAJ GO (Wygrana!)
        2. Jeśli przeciwnik ma ruch dający 5-w-rzędzie (czyli ma otwartą 4 lub 4-półotwartą) -> ZABLOKUJ GO NATYCHMIAST!
        3. Jeśli istnieje ruch tworzący podwójną 4 lub widelec 4+3 -> ZAGRAJ GO!
        4. Jeśli przeciwnik może utworzyć podwójną 4 lub otwartą 4 -> ZABLOKUJ GO!
        """
        opponent = 1 if current_player == 2 else 2
        valid_moves = [(r, c) for r in range(self.board_size) for c in range(self.board_size) if board_grid[r, c] == 0]

        if not valid_moves:
            return None

        # 0. STRATEGICZNE OTWARCIE (Centrum i Sąsiedztwo w pierwszych 2 ruchach)
        total_stones = np.count_nonzero(board_grid)
        if total_stones == 0:
            return (self.board_size // 2, self.board_size // 2), "🎯 Strategiczne Otwarcie Centrum (Tengen)"
        elif total_stones == 1:
            # Postaw w bezpośrednim sąsiedztwie pierwszego kamienia (lub w centrum)
            occupied = np.argwhere(board_grid != 0)
            if len(occupied) > 0:
                fr, fc = occupied[0]
                neighbors = [
                    (fr + dr, fc + dc)
                    for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                    if (dr != 0 or dc != 0) and 0 <= fr + dr < self.board_size and 0 <= fc + dc < self.board_size and board_grid[fr + dr, fc + dc] == 0
                ]
                if neighbors:
                    # Wybierz sąsiada najbliżej centrum planszy (7, 7)
                    center = self.board_size // 2
                    best_n = min(neighbors, key=lambda p: abs(p[0] - center) + abs(p[1] - center))
                    return best_n, "🎯 Strategiczna Odpowiedź w Sąsiedztwie Otwarcia"

        # 1. WYGRANA NATYCHMIASTOWA (NASZA 5)
        for r, c in valid_moves:
            patterns = self.analyze_move(board_grid, r, c, current_player)
            if PatternType.WIN_5 in patterns:
                return (r, c), "🏆 Natychmiastowa Wygrana (5-w-rzędzie)"

        # 2. BEZWZGLĘDNA BLOKADA WYGRANEJ PRZECIWNIKA (PRZECIWNIK MA 4)
        for r, c in valid_moves:
            patterns_opp = self.analyze_move(board_grid, r, c, opponent)
            if PatternType.WIN_5 in patterns_opp:
                return (r, c), "🚨 BEZWZGLĘDNA BLOKADA Wygranej Przeciwnika (Blokada 5)"

        # 3. NASZA POTĘŻNA GROŹBA (DOUBLE 4 lub FORK 4+3 lub THREAT 4 OPEN)
        for r, c in valid_moves:
            patterns = self.analyze_move(board_grid, r, c, current_player)
            if PatternType.DOUBLE_THREAT_4 in patterns or PatternType.THREAT_4_OPEN in patterns or PatternType.FORK_4_3 in patterns:
                return (r, c), "⚡ Atak Groźbą Czwórki / Widelcem 4+3"

        # 4. BLOKADA POTĘŻNEJ GROŹBY PRZECIWNIKA (BLOKADA OTWARTYCH 4 / WIDELCA)
        for r, c in valid_moves:
            patterns_opp = self.analyze_move(board_grid, r, c, opponent)
            if PatternType.DOUBLE_THREAT_4 in patterns_opp or PatternType.THREAT_4_OPEN in patterns_opp or PatternType.FORK_4_3 in patterns_opp:
                return (r, c), "🛡️ Blokada Otwartej Czwórki / Widelca Przeciwnika"

        # 5. BLOKADA OTWARTYCH TRÓJEK PRZECIWNIKA (Kiedy przeciwnik ma otwartą 3)
        for r, c in valid_moves:
            patterns_opp = self.analyze_move(board_grid, r, c, opponent)
            if PatternType.THREAT_3_OPEN in patterns_opp or PatternType.FORK_3_3 in patterns_opp:
                return (r, c), "🛡️ Blokada Otwartej Trójki Przeciwnika"

        return None
