# scripts/seed_memory.py
import sys
import os
import logging
import hydra

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.rag_engine import StrategyMemory

# Set up the logger
log = logging.getLogger(__name__)

def seed():
    memory = StrategyMemory()
    
    # Seed a Zero Strategy
    memory.memorize_rule(
        rule_text="When holding the Skull King, do not play it early. Wait for a Mermaid to appear.",
        persona="Forced-Zero",
        metadata={"phase": "PLAYING", "topic": "skull_king"}
    )
    
    # Seed a Rational Strategy
    memory.memorize_rule(
        rule_text="If you have met your bid (Full), lead a low card to force others to win.",
        persona="Rational",
        metadata={"phase": "PLAYING", "topic": "lead_strategy"}
    )
    
    print("Memory seeded successfully.")

if __name__ == "__main__":
    seed()