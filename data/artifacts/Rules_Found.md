# Skull King AI: Discovered Strategies

## Persona: FORCED ZERO STRATEGIES

### rule_1
- **Context:** Round 2 | Target Bid: 0
- **Strategy:** When you have only one high‑value suit card (e.g., Yellow 14) **and** an Escape in hand, and it’s your turn to play a trick where the lead suit is not that card’s suit, **always play the Escape**.

### rule_4
- **Context:** Round 6 | Target Bid: 0
- **Strategy:** If you are void in the lead suit, **never play a high‑value trump (e.g., Black 11) unless you can guarantee that an opponent will beat it with a higher trump or a special**; instead, play an Escape or your lowest non‑trump card. In Round 6, Player 0 was void in the Yellow lead but chose to play Black 11—an unopposed high trump—thereby securing the trick and making the 0‑bid impossible.

### rule_3
- **Context:** Round 1 | Target Bid: 0
- **Strategy:** In a one‑card round, a player who holds a single Pirate cannot bid 0 because the Pirate will win the trick unless an opponent also holds a higher special (Mermaid or Skull King). Since each opponent can play only one card, they cannot simultaneously beat the Pirate and avoid winning the trick. Therefore, if you are the only player with a Pirate and no other cards, you must bid at least 1; a 0 bid guarantees a trick win and thus a catastrophic score.

## Persona: RATIONAL STRATEGIES

### rule_11
- **Context:** Round 5 | Target Bid: 3
- **Strategy:** When holding any trump of value 5 or lower (Black 1‑5) without a higher trump or a Pirate/Skull King to back it up, do **not** count those low trumps toward your bid; instead subtract one from the estimated number of tricks you expect to win.

### rule_12
- **Context:** Round 6 | Target Bid: 2
- **Strategy:** If you hold a **Pirate** in a hand that also contains low‑value suit cards and the round’s bid is modest (≤ 3), avoid leading a suit card that you can beat with a higher suit or trump. Instead, lead the Pirate immediately. This forces any opponent who holds the corresponding higher‑ranked suit or a Trump to win the trick (or lose it if they also have a special), thereby preventing them from using their high cards later and giving you a better chance to hit your exact bid.

### rule_13
- **Context:** Round 6 | Target Bid: 3
- **Strategy:** When your hand contains a low‑value trump (e.g., Black 9) and at least one opponent has bid non‑zero, **do not play the low trump unless you can confirm that all higher trumps are already out**.

### rule_4
- **Context:** Round 1 | Target Bid: 1
- **Strategy:** In a one‑card round, a Mermaid is only safe if no opponent holds a Pirate or a Skull King. If Player 2 bids 1 with only a Mermaid, the instant an opponent plays a Pirate (or Skull King) the bid becomes impossible. Therefore, P2 should bid 0 in this situation, or otherwise play a Pirate to force the opponent to lose the trick. Opponents can exploit this by leading a Pirate immediately after a Mermaid to guarantee a win.

### rule_5
- **Context:** Round 1 | Target Bid: 1
- **Strategy:** When holding a Mermaid, do not count it as a guaranteed win unless you are certain that no Pirates or Skull Kings are in play; otherwise lower your bid by 1.

### rule_6
- **Context:** Round 3 | Target Bid: 2
- **Strategy:** If you have a Pirate (or any special) and you suspect an opponent may hold a White Whale, do NOT play the special on the first trick. Instead, play a low suit card (e.g., Green 3) to force the opponent to waste a high card or play the White Whale, which will nullify your special and guarantee you lose that trick. This mis‑reading of the opponent’s potential White Whale caused Player 1’s bid of 2 to become impossible, as the Pirate was rendered powerless when the opponent played the White Whale.

### rule_7
- **Context:** Round 3 | Target Bid: 4
- **Strategy:** If you hold the Skull King and you’re the only player with a trump (e.g., Black 8), never lead the Skull King on the first trick unless you know the opponent has no Pirate or Mermaid. In Trick 2 of Round 3, Player 3 misread that the opponent had no special and played the Skull King, only to be beaten by a Pirate that the opponent held. This forced Player 3 to lose three tricks and made the 4‑trick bid impossible. The correct play is to lead a high suit card (Yellow 11) first, forcing the opponent to either play a special or a lower trump, then save the Skull King for a trick where you can guarantee it wins.

### rule_8
- **Context:** Round 3 | Target Bid: 4
- **Strategy:** When you possess only one special card (e.g., Skull King) and all other cards are lower trumps or suits, count that special as a single guaranteed win and bid at most one trick for it; do not assume the remaining cards will win tricks unless they are higher than every possible opponent card in their suit or trump. [ACTION]: <ID>

### rule_9
- **Context:** Round 4 | Target Bid: 4
- **Strategy:** When holding a special card that can be beaten by a higher special (e.g., a Mermaid when Pirates or Skull King may appear), do not count it as a guaranteed win; subtract one from your bid for each such special.

### rule_10
- **Context:** Round 4 | Target Bid: 2
- **Strategy:** If you hold a mid‑to‑high trump (e.g., Black 12) and an opponent has bid 0 or 1, don’t play that trump immediately. Opponents with low bids often keep a higher trump (Black 13/14) to win a trick when you play your own. In round 4, Player 3’s Black 12 was played on the first trick while Player 2 (bid 0) still held Black 13. Player 2 used that higher trump to win the trick, turning a potential win into a loss and making the 2‑trick bid impossible.

