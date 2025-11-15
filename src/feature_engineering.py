import numpy as np
import pandas as pd
from typing import Tuple


class PokerFeatureEngine:
    """
    Transforms raw poker card data into features that actually matter.
    
    Instead of feeding a model 10 disconnected numbers, we give it
    what a poker player would look at: flushes, straights, rank patterns.
    """
    
    HAND_LABELS = {
        0: 'Nothing',
        1: 'One pair',
        2: 'Two pairs',
        3: 'Three of a kind',
        4: 'Straight',
        5: 'Flush',
        6: 'Full house',
        7: 'Four of a kind',
        8: 'Straight flush',
        9: 'Royal flush'
    }
    
    def __init__(self):
        pass
    
    def _extract_suits_and_ranks(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Pull out suits and ranks from the flat row structure."""
        suits = row.iloc[[0, 2, 4, 6, 8]].values.astype(int)
        ranks = row.iloc[[1, 3, 5, 7, 9]].values.astype(int)
        return suits, ranks
    
    def _is_flush(self, suits: np.ndarray) -> int:
        """All same suit? That's a flush component."""
        return int(len(np.unique(suits)) == 1)
    
    def _is_straight(self, ranks: np.ndarray) -> int:
        """
        Check if ranks form a sequence.
        Tricky part: Ace can be high (10-J-Q-K-A) or low (A-2-3-4-5).
        """
        sorted_ranks = np.sort(ranks)
        
        # Normal straight: consecutive numbers
        if np.all(np.diff(sorted_ranks) == 1):
            return 1
        
        # Special case: A-2-3-4-5 (wheel)
        if np.array_equal(sorted_ranks, [1, 2, 3, 4, 5]):
            return 1
        
        # Special case: 10-J-Q-K-A
        if np.array_equal(sorted_ranks, [1, 10, 11, 12, 13]):
            return 1
        
        return 0
    
    def _is_royal_straight(self, ranks: np.ndarray) -> int:
        """
        Check if this is specifically 10-J-Q-K-A.
        This is needed to distinguish Royal Flush from regular Straight Flush.
        """
        sorted_ranks = np.sort(ranks)
        return int(np.array_equal(sorted_ranks, [1, 10, 11, 12, 13]))
    
    def _rank_patterns(self, ranks: np.ndarray) -> Tuple[int, int, int]:
        """
        The real magic. This is what separates pairs from trips from quads.
        
        Returns:
            unique_ranks: How many different ranks (5=nothing, 2=quad or full house)
            max_count: Most times a rank appears (4=quad, 3=trips, 2=pair)
            second_max_count: Second most frequent (useful for full house vs trips)
        """
        unique, counts = np.unique(ranks, return_counts=True)
        counts_sorted = np.sort(counts)[::-1]
        
        unique_ranks = len(unique)
        max_count = counts_sorted[0]
        second_max_count = counts_sorted[1] if len(counts_sorted) > 1 else 0
        
        return unique_ranks, max_count, second_max_count
    
    def transform_row(self, row: pd.Series) -> dict:
        """Convert one hand (row) into poker-meaningful features."""
        suits, ranks = self._extract_suits_and_ranks(row)
        
        unique_ranks, max_count, second_max = self._rank_patterns(ranks)
        
        return {
            'is_flush': self._is_flush(suits),
            'is_straight': self._is_straight(ranks),
            'is_royal_straight': self._is_royal_straight(ranks),  # NEW!
            'unique_ranks': unique_ranks,
            'max_rank_count': max_count,
            'second_max_rank_count': second_max
        }
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform entire dataset. Keep it vectorized where possible."""
        features = df.iloc[:, :10].apply(self.transform_row, axis=1, result_type='expand')
        return features
    
    @staticmethod
    def get_hand_name(hand_id: int) -> str:
        return PokerFeatureEngine.HAND_LABELS.get(hand_id, 'Unknown')