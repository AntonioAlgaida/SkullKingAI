# main_test_simulation.py

import numpy as np
import time
from skull_king.skull_king_env import SkullKingEnv
from agents.heuristic import HeuristicAgent

def run_game(render=True):
    # 1. Setup
    env = SkullKingEnv(num_players=4)
    obs, _ = env.reset()
    
    bots = [HeuristicAgent(env.physics) for _ in range(4)]
    game_over = False
    
    print(f"--- STARTING NEW GAME (4 Players) ---")
    
    while not game_over:
        current_player = env.current_player_index
        phase = env.phase # 0=Bidding, 1=Playing
        round_num = env.current_round
        
        # 2. Get Action
        mask = env.get_action_mask(current_player)
        action = bots[current_player].act(obs, mask)
        
        # 3. Render Play
        if render:
            if phase == 0:
                print(f"[R{round_num}] Player {current_player} Bids: {action}")
            elif phase == 1:
                if action == 74:
                    print(f"[R{round_num}] Player {current_player} Plays: TIGRESS (as ESCAPE)")
                else:
                    card_str = str(env.physics.actions[action]['card'])
                    # Clean up the string for easier reading
                    clean_str = card_str.replace("Card(", "").replace(")", "").replace("NUMBER", "").replace("SPECIAL", "")
                    print(f"[R{round_num}] Player {current_player} Plays: {card_str}")

        obs, reward, terminated, truncated, info = env.step(action)
        
        # 4. Check for Trick End (New Logic)
        if "trick_winner" in info:
            winner = info["trick_winner"]
            bonus = info["trick_bonus"]
            destroyed = info["trick_destroyed"]
            
            if destroyed:
                print(f"\t>>> TRICK DESTROYED (Kraken) <<<")
            else:
                bonus_str = f" (+{bonus} Bonus)" if bonus > 0 else ""
                print(f"\t>>> Trick Won by Player {winner}{bonus_str} <<<")
            
            # Formatting line between tricks
            if "round_rewards" not in info:
                print("-" * 40)

        # 5. Check for Round/Game End
        if "round_rewards" in info:
            print(f"\n{'='*15} ROUND {round_num} ENDED {'='*15}")
            print(f"Bids:   {info['bids']}")
            print(f"Won:    {info['won']}")
            print(f"Scores: {info['round_rewards']}")
            print(f"Total:  {env.scores}\n")
            # time.sleep(1) 
            
        if terminated:
            print(f"=== GAME OVER ===")
            print(f"Final Scores: {info['final_scores']}")
            game_over = True

if __name__ == "__main__":
    for _ in range(10):
        run_game()

# To run with uv:
# uv run main_simulation.py