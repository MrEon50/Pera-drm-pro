# 🧠 PERA-DRM PRO: Neuro-Symbolic AI Engine for Board Games

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Neuro--Symbolic_Hybrid-blueviolet?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Permutation-Equivariant Residual-Attention Network + Dynamic Rules Matrix 3.0 PRO + General Gomoku Tactical Guard**

*Zunifikowany, modułowy silnik sztucznej inteligencji dedykowany dla gier planszowych (Gomoku, Go, Szachy, Othello, Kółko i Krzyżyk) z trwałą pamięcią JSON, ewolucją Bayesowską, samoregulacją stylu gry ($\Phi$) oraz eksplicytnym strażnikiem taktyki.*

</div>

---

## 📖 Spis Treści

1. [Geneza i Kluczowe Innowacje](#-geneza-i-kluczowe-innowacje)
2. [Architektura Systemu (Neuro-Symbolic)](#-architektura-systemu-neuro-symbolic)
3. [Główne Moduły i Komponenty](#-główne-moduły-i-komponenty)
4. [Szybki Start i Uruchomienie](#-szybki-start-i-uruchomienie)
5. [Gra Interaktywna (`play_gomoku.py`)](#-gra-interaktywna-play_gomokupy)
6. [Zarządzanie Treningiem i Poziomami Trudności](#-zarządzanie-treningiem-i-poziomami-trudności)
7. [System Pamięci JSON i Bayesowska Ewolucja](#-system-pamięci-json-i-bayesowska-ewolucja)
8. [Struktura Projektu](#-struktura-projektu)
9. [Obsługiwane Gry Planszowe](#-obsługiwane-gry-planszowe)
10. [Licencja](#-licencja)

---

## 🌟 Geneza i Kluczowe Innowacje

Tradycyjne podejścia oparte na czystych sieciach neuronowych (typu *AlphaZero*) borykają się z poważnym problemem: **traktują planszę wyłącznie statystycznie**, przez co potrafią przeoczyć elementarne groźby taktyczne (np. natychmiastową wygraną przeciwnika przez otwartą czwórkę), dopóki nie rozegrają milionów partii.

**PERA-DRM PRO** rozwiązuje ten problem poprzez **Hybrydową Architekturę Neuro-Symboliczną**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               STAN PLANSZY / RUCH GRACZA               │
                       └───────────────────────────┬────────────────────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
 ┌────────────────────────────────────┐                         ┌────────────────────────────────────┐
 │     WARSTWA TAKTYCZNA (General)    │                         │       WARSTWA NEURONOWA (PERA)     │
 │  Eksplicytny Strażnik Zasad:       │                         │  PERANet (ResNet + Self-Attention) │
 │  • Natychmiastowa Wygrana (5)      │                         │  • Estymacja Szansy Wygranej (V)   │
 │  • Bezwzględna Blokada 5 i 4       │                         │  • Rozkład Prawdopodobieństwa (P)  │
 │  • Wykrywanie Widelców 4+3 i 3+3   │                         │  • Uczenie Reprezentacji Pozycji   │
 └─────────────────┬──────────────────┘                         └─────────────────┬──────────────────┘
                   │                                                              │
                   │ (Priorytet Taktyczny)                                        │ (Ocena Strategiczna)
                   └───────────────────────────────┬──────────────────────────────┘
                                                   ▼
                                ┌────────────────────────────────────┐
                                │     DRM 3.0 & ENHANCED PUCT        │
                                │  • Equilibrium Governor Phi        │
                                │  • Bayesowska Ewolucja (Beta Dist) │
                                │  • Trwała Pamięć w Plikach JSON    │
                                └──────────────────┬─────────────────┘
                                                   │
                                                   ▼
                                ┌────────────────────────────────────┐
                                │     OPTYMALNY, BEZBŁĘDNY RUCH      │
                                └────────────────────────────────────┘
```

### 🔑 4 Filary Silnika:
1. **PERANet (Deep Neural Engine)**: Sieć konwolucyjno-rezidualna z wielogłowicową uwagą (*Multi-Head Self-Attention*), uwzględniająca przestrzenne symetrie planszy (obroty, odbicia).
2. **Dynamic Rules Matrix 3.0 PRO (Symbolic Engine)**: Dynamiczna matryca reguł logicznych z polaryzacją (Atak vs Obrona) i estymacją Bayesowską (Rozkład Beta).
3. **Equilibrium Governor ($\Phi$)**: Autonomiczny zarządca równowagi stylu gry, zapobiegający popadaniu w skrajną pasywność lub bezmyślną agresję.
4. **General Gomoku Tactical Guard**: Bezwzględny moduł sprawdzający krytyczne wzorce taktyczne, eliminujący „luki logiczne” i gwarantujący 100% obrony przed natychmiastowymi groźbami.

---

## 🏛️ Architektura Systemu (Neuro-Symbolic)

### 1. Matematyka Harmonii Governora ($\Phi$)
Wskaźnik $\Phi \in [-1.0, +1.0]$ definiuje stan równowagi systemu:

$$\Phi = \frac{\sum_{i \in \text{Active}} \text{Polarity}_i \cdot \text{Weight}_i \cdot \text{Magnitude}_i}{\sum_{i \in \text{Active}} \text{Weight}_i \cdot \text{Magnitude}_i + \epsilon}$$

* $\Phi > +0.25$: Styl silnie ekspansywny / atakujący $\rightarrow$ Governor wzmacnia wagi reguł defensywnych.
* $\Phi < -0.25$: Styl pasywny / asekuracyjny $\rightarrow$ Governor stymuluje reguły ataku.

### 2. Bayesowska Ocena Skuteczności Reguł (Beta Distribution)
Każda reguła taktyczna posiada parametry sukcesów ($\alpha$) i porażek ($\beta$):

$$\text{Posterior Mean} = \frac{1 + \text{Successes}}{2 + \text{Successes} + \text{Failures}}$$

$$\text{Posterior Variance} = \frac{\alpha \cdot \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$$

* **Auto-Kwarantanna**: Jeżeli po min. 5 użyciach skuteczność reguły spadnie $\le 20\%$, zostaje ona automatycznie przeniesiona do archiwum kwarantanny, nie psując jakości decyzyjnej bota.

---

## 📦 Główne Moduły i Komponenty

| Plik | Rola i Funkcjonalność |
| :--- | :--- |
| `pera_drm_pro.py` | **Serce Silnika AI**: Klasy `GameDescriptor`, `PERANet`, `EquilibriumGovernor`, `DRMSystem`, `EnhancedPUCT`. |
| `gomoku_tactical_analyzer.py` | **Strażnik Taktyki Gomoku**: Detekcja ciągów 5, otwartych 4, widelców 4+3, podwójnych trójek oraz heurystyka otwarcia centrum (*Tengen*). |
| `json_memory_manager.py` | **Trwała Pamięć**: Serializacja i deserializacja wag, historii i baz wiedzy do formatu JSON. |
| `play_gomoku.py` | **Aplikacja Interaktywna**: Menu Główne, Komendy w grze (`m`, `r`, `s`, `h`, `q`), rejestracja logów ruchów na żywo. |
| `train_bot.py` | **Pipeline Treningowy**: Pre-trening bota (Self-Play i przeciwnicy syntetyczni) z podglądem postępów dla każdej partii. |
| `train_gomoku_curriculum.py` | **Curriculum & Curiosity**: Uczenie dwuetapowe (4 profile przeciwników + nagradzanie za innowacje *Novelty*). |
| `demo_games.py` | **Weryfikacja Wielogrowa**: Test uniwersalności dla Gomoku 15x15, Go 9x9 oraz Szachów 8x8. |

---

## 🚀 Szybki Start i Uruchomienie

### Wymagania:
* Python 3.8+ (Zalecany Python 3.10+)
* `torch`, `numpy`

```bash
# Instalacja zależności
pip install torch numpy

# Przejście do katalogu projektu
cd PERA-DRM-PRO

# Uruchomienie aplikacji interaktywnej
python play_gomoku.py
```

---

## 🎮 Gra Interaktywna (`play_gomoku.py`)

Po uruchomieniu `play_gomoku.py` wyświetla się przejrzysty panel sterowania:

```text
======================================================================
🎮 PERA-DRM PRO: GOMOKU AI - MENU GŁÓWNE
======================================================================
1. 🕹️ Zagraj z AI (Gomoku 15x15)
2. 🏋️ Trenuj AI (Wybor Poziomu: Początkujący -> Zaawansowany)
3. 📊 Wyświetl Statystyki Bota & Reguły DRM z JSON
4. 💾 Zapisz / Eksportuj Log Ostatniego Meczu
0. 🚪 Wyjście z gry
======================================================================
```

### 💡 Dostępne Komendy Podczas Gry:
* **`wiersz kolumna`** (np. `5 5` lub `7 7`): Wykonanie ruchu na planszy (Oś Y: 0–14, Oś X: 0–14).
* **`m` / `menu`**: Natychmiastowy powrót do Menu Głównego.
* **`r` / `restart`**: Rozpoczęcie aktualnej partii od nowa na czystej planszy.
* **`s` / `save`**: Zapisanie logów aktualnego meczu do wybranego pliku `.log`.
* **`h` / `help`**: Wyświetlenie spisu komend i instrukcji.
* **`q` / `quit`**: Wyjście z aplikacji.

### 📝 Live Move Tracking Log (Podgląd Ruchów w Czasie Rzeczywistym):
Każde posunięcie jest natychmiast kategoryzowane i oceniane:
```text
📝 [18:40:05] Ruch #04 | AI Bot [O] -> Pole ( 5,  4) | Typ: 🛡️ TAKTYCZNY (Blokada Otwartej Trójki) | Wzorce: [threat_3_open] | V: +0.99 | Phi: +0.193
📝 [18:40:28] Ruch #05 | Człowiek [X] -> Pole ( 4,  6) | Typ: Ruch Gracza | Wzorce: [extension_2]
```

---

## 🏋️ Zarządzanie Treningiem i Poziomami Trudności

Z poziomu Menu Głównego (Opcja 2) lub uruchamiając `python train_bot.py`, masz do dyspozycji 4 tryby treningu:

1. **🟢 Początkujący**: 10 gier self-play, szybki MCTS (10 symulacji/ruch) $\sim 20$ sekund.
2. **🟡 Średniozaawansowany**: 30 gier self-play, zbalansowany MCTS (25 symulacji/ruch) $\sim 1$ minuta.
3. **🔴 Zaawansowany / Ekspert**: 100 gier self-play, głęboki MCTS (50 symulacji/ruch) $\sim 4$ minuty.
4. **⚙️ Własny Poziom**: Dowolna konfiguracja liczby partii i głębokości symulacji.

Raport z treningu wyświetla postępy w czasie rzeczywistym:
```text
🎮 Partia  1/10 | Wynik: Wygrana Bot 1   | Ruchów: 42 | Loss: 6.4387 | Phi: +0.193 | DRM Reguły: 3 akt / 0 kwar | Czas: 4.40s
🎮 Partia  2/10 | Wynik: Wygrana Bot 2   | Ruchów: 38 | Loss: 6.4071 | Phi: +0.193 | DRM Reguły: 3 akt / 0 kwar | Czas: 4.18s
```

---

## 💾 System Pamięci JSON i Bayesowska Ewolucja

Wszystkie nabyte umiejętności bota są zapisywane w katalogu `./game_memories/`:
* `gomoku_interactive_memory.json` – Wyewoluowana baza reguł DRM, statystyki rozkładu Beta i wskaźnik $\Phi$.
* `gomoku_peranet_weights.pt` – Zoptymalizowane wagi sieci neuronowej PyTorch.
* `last_game_trace.log` – Pełny stenogram ostatnio rozegranego meczu z człowiekiem.

### Przykładowa struktura wpisu reguły w JSON:
```json
{
  "id": "rule_0_1_842",
  "name": "Blok Otwartym Czwórkom",
  "polarity": -0.8,
  "magnitude": 1.0,
  "weight": 2.8,
  "strength": 8.12,
  "post_mean": 0.88,
  "post_var": 0.015,
  "quarantined": false,
  "tags": ["obrona", "bezpieczenstwo"]
}
```

---

## 📁 Struktura Projektu

```
PERA-DRM-PRO/
├── gomoku_tactical_analyzer.py # Warstwa Taktyczna & Strażnik Wzorców (General Gomoku)
├── pera_drm_pro.py             # Silnik Główny AI (GameDescriptor, PERANet, DRM, Governor, MCTS)
├── json_memory_manager.py      # Menedżer Trwałej Pamięci JSON
├── play_gomoku.py              # Aplikacja z Menu Głównym i Komendami
├── train_bot.py                # Automatyczny Pre-Trening z Raportem
├── train_gomoku_curriculum.py  # Uczenie Dwuetapowe z Nagrodą za Nowość (Curiosity Engine)
├── demo_games.py               # Wielogrowa Demonstracja (Go, Gomoku, Szachy)
├── game_memories/              # Bazy Wiedzy JSON, Wagi PyTorch i Zapisy Meczów
│   ├── gomoku_interactive_memory.json
│   ├── gomoku_peranet_weights.pt
│   └── last_game_trace.log
└── readme.md                   # Kompletna Dokumentacja Projektu
```

---

## ♟️ Obsługiwane Gry Planszowe

Moduł `GameDescriptor` pozwala na bezproblemowe podłączenie dowolnej gry o dyskretnej geometrii:
* **Gomoku**: 15x15 (Plansza 3 kanały, 225 akcji)
* **Go (Weiqi)**: 9x9, 13x13, 19x19 (Plansza 17 kanałów, do 361 akcji)
* **Szachy**: 8x8 (Plansza 20 kanałów, 4096 akcji)
* **Othello (Reversi)**: 8x8 (Plansza 3 kanały, 64 akcje)
* **Kółko i Krzyżyk**: 3x3 (Plansza 3 kanały, 9 akcji)

---

## 📄 Licencja

Projekt udostępniony na licencji **MIT License** — swoboda modyfikacji, integracji w projektach komercyjnych i naukowych.
