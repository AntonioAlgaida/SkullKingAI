import os
from functools import lru_cache

class PromptLoader:
    """
    Loads and caches prompt text files from src/prompts/
    """
    
    # Define the base path relative to this file
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")

    @classmethod
    @lru_cache(maxsize=None)
    def load(cls, category: str, filename: str) -> str:
        """
        Loads a specific text file. 
        Usage: PromptLoader.load("rules", "card_hierarchy")
        """
        path = os.path.join(cls.PROMPTS_DIR, category, f"{filename}.txt")
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except FileNotFoundError:
            return f"[ERROR: Prompt file not found: {category}/{filename}.txt]"

    @classmethod
    def get_bidding_bundle(cls, persona_name: str) -> str:
        """Assembles rules for the Bidding Phase."""
        parts = [
            cls.load("rules", "game_intro"),
            cls.load("rules", "card_hierarchy"),
            cls.load("rules", "scoring"),
            "--- STRATEGY GUIDE ---\n" + cls.get_persona(persona_name)
        ]
        return "\n\n".join(parts)

    @classmethod
    def get_playing_bundle(cls, persona_name: str) -> str:
        """Assembles rules for the Playing Phase."""
        parts = [
            cls.load("rules", "game_intro"),
            cls.load("rules", "card_hierarchy"),
            cls.load("rules", "trick_mechanics"),
            # We skip full scoring.txt to save space, but add a reminder
            "SCORING REMINDER: Bonuses (+20 to +40) only apply if you meet your bid exactly. Missing your bid gives a penalty.",
            "--- STRATEGY GUIDE ---\n" + cls.get_persona(persona_name)
        ]
        return "\n\n".join(parts)

    @classmethod
    def get_persona(cls, persona_name: str) -> str:
        name = persona_name.lower().replace("-", "_").replace(" ", "_")
        return cls.load("personas", name)