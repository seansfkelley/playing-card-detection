import numpy as np
from typing import ClassVar
from dataclasses import dataclass


@dataclass(frozen=True)
class CardRect:
    """
    A rectangular area on a card. Origin is in the top left. Use negative numbers to start from the bottom or right.
    """

    left: int
    right: int
    top: int
    bottom: int

    def as_nparray(self, card_width: int, card_height: int):
        left, right = sorted(
            (self.left + card_width) % card_width,
            (self.right + card_width) % card_width,
        )
        top, bottom = sorted(
            (self.top + card_height) % card_height,
            (self.bottom + card_height) % card_height,
        )
        return np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class CardGroup:
    card_names: frozenset[str]
    identifiable_rects: frozenset[CardRect]


@dataclass(frozen=True)
class Deck:
    width: int
    height: int
    cards: frozenset[CardGroup]
