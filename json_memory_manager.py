"""
JSON Memory Manager for PERA-DRM-PRO
Trwały zapis i odczyt stanu bazy wiedzy, reguł DRM, rozkładów Bayesowskich
oraz statystyk gier (Go, Gomoku, Szachy, Othello) w czytelnym formacie JSON.
"""

import json
import os
import time
from typing import Dict, List, Any, Tuple, Optional


class JSONMemoryManager:
    """
    Menedżer pamięci obsługujący serializację i deserializację bazy wiedzy PERA-DRM do/z plików JSON.
    """

    def __init__(self, storage_dir: str = "."):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir, exist_ok=True)

    def get_game_filepath(self, game_name: str) -> str:
        safe_name = game_name.lower().replace(" ", "_").replace("-", "_")
        return os.path.join(self.storage_dir, f"{safe_name}_memory.json")

    def save_memory(self, drm_system: Any, game_stats: Dict[str, Any], filepath: Optional[str] = None, game_name: str = "default") -> str:
        """
        Zapisuje stan systemu DRM, reguły, statystyki Bayesowskie oraz metadane do pliku JSON.
        """
        if filepath is None:
            filepath = self.get_game_filepath(game_name)

        active_rules_data = []
        quarantined_rules_data = []

        for rule in drm_system.rules:
            rule_dict = {
                "id": rule.id,
                "name": rule.name,
                "polarity": float(rule.polarity),
                "magnitude": float(rule.magnitude),
                "weight": float(rule.weight),
                "strength": float(rule.strength),
                "priority": int(rule.priority),
                "features": [float(f) for f in rule.features],
                "tags": list(rule.tags),
                "usage_count": int(rule.usage_count),
                "success_count": int(rule.success_count),
                "failure_count": int(rule.failure_count),
                "post_mean": float(rule.post_mean),
                "post_var": float(rule.post_var),
                "quarantined": bool(rule.quarantined),
                "quarantine_reason": rule.quarantine_reason,
                "created_cycle": int(rule.created_cycle),
                "last_used_cycle": int(rule.last_used_cycle)
            }
            if rule.quarantined:
                quarantined_rules_data.append(rule_dict)
            else:
                active_rules_data.append(rule_dict)

        memory_payload = {
            "version": "PERA-DRM-PRO-1.0",
            "timestamp": time.time(),
            "date_string": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_name": game_name,
            "system_state": {
                "cycle": drm_system.cycle,
                "harmony_phi": float(drm_system.governor.calculate_system_harmony(drm_system.rules)),
                "activity_history": [float(a) for a in drm_system.activity_history[-20:]],
                "total_rules": len(drm_system.rules),
                "active_rules_count": len(active_rules_data),
                "quarantined_rules_count": len(quarantined_rules_data)
            },
            "game_stats": game_stats,
            "rules": active_rules_data,
            "quarantine_archive": quarantined_rules_data
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(memory_payload, f, indent=2, ensure_ascii=False)

        print(f"💾 [JSONMemoryManager] Pamięć gier zapisaną pomyślnie w: {filepath}")
        return filepath

    def load_memory(self, filepath: str, drm_system_cls: Any) -> Tuple[Any, Dict[str, Any]]:
        """
        Wczytuje wiedzę z pliku JSON i rekonstruuje obiekt systemu DRM oraz statystyki gier.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Nie znaleziono pliku pamięci JSON: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)

        game_name = payload.get("game_name", "unknown")
        drm_system = drm_system_cls()
        drm_system.cycle = payload.get("system_state", {}).get("cycle", 0)

        # Rekonstrukcja reguł z bazy JSON
        all_rules_raw = payload.get("rules", []) + payload.get("quarantine_archive", [])
        
        for r_data in all_rules_raw:
            rule = drm_system.add_rule(
                name=r_data["name"],
                features=r_data.get("features", [1.0]),
                polarity=r_data.get("polarity", 0.0),
                magnitude=r_data.get("magnitude", 1.0),
                weight=r_data.get("weight", 1.0),
                priority=r_data.get("priority", 1),
                tags=set(r_data.get("tags", []))
            )
            rule.id = r_data["id"]
            rule.strength = r_data.get("strength", 0.0)
            rule.usage_count = r_data.get("usage_count", 0)
            rule.success_count = r_data.get("success_count", 0)
            rule.failure_count = r_data.get("failure_count", 0)
            rule.post_mean = r_data.get("post_mean", 0.5)
            rule.post_var = r_data.get("post_var", 0.08)
            rule.quarantined = r_data.get("quarantined", False)
            rule.quarantine_reason = r_data.get("quarantine_reason", None)
            rule.created_cycle = r_data.get("created_cycle", 0)
            rule.last_used_cycle = r_data.get("last_used_cycle", 0)

        game_stats = payload.get("game_stats", {})
        print(f"📖 [JSONMemoryManager] Wczytano {len(drm_system.rules)} reguł z pamięci: {filepath}")
        return drm_system, game_stats
