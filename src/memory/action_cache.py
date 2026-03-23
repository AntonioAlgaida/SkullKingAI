# src/memory/action_cache.py
#
# Semantic Action Cache (SAC).
#
# After the LLM selects a legal action, the (semantic-state-query, action_id)
# pair is stored in a ChromaDB collection.  On the next decision, if the
# current state is semantically close enough to a cached state AND the cached
# action is still legal, we return the cached action directly — bypassing the
# LLM call entirely.
#
# Thresholds (L2 distance, lower = more similar):
#   BIDDING  : 0.10  — hand toxicity + round + opponent summary rarely change
#   PLAYING  : 0.05  — trick composition is sensitive; only very close hits used
#
# Cache hit rate is logged and exposed as a convergence signal: rising hit rate
# means the agent's policy is stabilising.

import os
import logging
from typing import Optional, Tuple
import chromadb

logger = logging.getLogger(__name__)

# L2-distance thresholds below which a cached action is reused.
# Tighter for PLAYING because the exact trick context matters more.
BIDDING_THRESHOLD = 0.10
PLAYING_THRESHOLD = 0.05


class SemanticActionCache:
    """
    Stores and retrieves (semantic-state-query → action_id) pairs.

    Backed by a dedicated ChromaDB collection so it persists across runs and
    benefits from the same embedding model as the Grimoire.
    """

    def __init__(self, persistence_path: str = "data/chroma_db"):
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        abs_path = os.path.join(base_dir, persistence_path)
        os.makedirs(abs_path, exist_ok=True)

        client = chromadb.PersistentClient(path=abs_path)
        self.collection = client.get_or_create_collection("action_cache")

        self._hits   = 0
        self._misses = 0

    # ── Public API ─────────────────────────────────────────────────────────── #

    def lookup(
        self,
        query_text: str,
        legal_actions: list,
        phase: str,
    ) -> Optional[int]:
        """
        Returns a cached action_id if a sufficiently similar state exists AND
        the cached action is currently legal.  Returns None on a cache miss.
        """
        if self.collection.count() == 0:
            self._misses += 1
            return None

        threshold = BIDDING_THRESHOLD if phase == "BIDDING" else PLAYING_THRESHOLD

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=1,
                where={"phase": phase},
                include=["documents", "distances", "metadatas"],
            )
        except Exception as e:
            logger.debug(f"[ActionCache] query error: {e}")
            self._misses += 1
            return None

        if not results["distances"] or not results["distances"][0]:
            self._misses += 1
            return None

        dist      = results["distances"][0][0]
        meta      = results["metadatas"][0][0]
        action_id = int(meta["action_id"])

        if dist > threshold:
            self._misses += 1
            logger.debug(
                f"[ActionCache] MISS (dist={dist:.4f} > {threshold}) "
                f"phase={phase}"
            )
            return None

        if action_id not in legal_actions:
            # Cached action no longer available (different hand) — fall through.
            self._misses += 1
            logger.debug(
                f"[ActionCache] MISS — cached action {action_id} not legal "
                f"(legal={legal_actions})"
            )
            return None

        self._hits += 1
        logger.info(
            f"[ActionCache] HIT  (dist={dist:.4f})  action={action_id}  "
            f"hit_rate={self.hit_rate:.1%}"
        )
        return action_id

    def store(
        self,
        query_text: str,
        action_id: int,
        phase: str,
        persona: str,
    ) -> None:
        """
        Stores a (query, action) pair.  Skips if a very close entry already
        exists to avoid bloating the collection with near-duplicates.
        """
        # Skip deduplication check if collection is empty (saves a query).
        if self.collection.count() > 0:
            try:
                dup = self.collection.query(
                    query_texts=[query_text],
                    n_results=1,
                    where={"phase": phase},
                    include=["distances"],
                )
                if dup["distances"] and dup["distances"][0]:
                    if dup["distances"][0][0] < 0.02:
                        logger.debug("[ActionCache] skipping store — near-duplicate exists")
                        return
            except Exception:
                pass  # dedup is best-effort; never block a store

        entry_id = f"cache_{self.collection.count() + 1}"
        self.collection.add(
            documents=[query_text],
            metadatas=[{"action_id": action_id, "phase": phase, "persona": persona}],
            ids=[entry_id],
        )
        logger.debug(f"[ActionCache] stored {entry_id}: action={action_id} phase={phase}")

    # ── Metrics ────────────────────────────────────────────────────────────── #

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def total_lookups(self) -> int:
        return self._hits + self._misses

    def stats_str(self) -> str:
        return (
            f"ActionCache: {self._hits}/{self.total_lookups} hits "
            f"({self.hit_rate:.1%})  entries={self.collection.count()}"
        )
