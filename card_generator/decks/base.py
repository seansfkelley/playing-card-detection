import numpy as np
from typing import ClassVar
from dataclasses import dataclass

# Measurements are provided in millimeters, so scale them up to a reasonable pixel size.
ARBITRARY_ZOOM_FACTOR = 4


# unsafe hash -> frozen wouldn't let us define the constructor in this way
@dataclass(unsafe_hash=True)
class CardRect:
    """
    A rectangular area on a card. Origin is in the top left. Use negative numbers to start from the bottom or right.
    """

    left: int
    right: int
    top: int
    bottom: int

    def __init__(self, left_mm: int, right_mm: int, top_mm: int, bottom_mm: int):
        self.left = left_mm * ARBITRARY_ZOOM_FACTOR
        self.right = right_mm * ARBITRARY_ZOOM_FACTOR
        self.top = top_mm * ARBITRARY_ZOOM_FACTOR
        self.bottom = bottom_mm * ARBITRARY_ZOOM_FACTOR

    def as_nparray(self, card_width: int, card_height: int):
        left, right = sorted(
            (
                (self.left + card_width) % card_width,
                (self.right + card_width) % card_width,
            )
        )
        top, bottom = sorted(
            (
                (self.top + card_height) % card_height,
                (self.bottom + card_height) % card_height,
            )
        )
        return np.array(
            [
                [left, top],
                [right, top],
                [right, bottom],
                [left, bottom],
            ],
            dtype=np.int,
        )


@dataclass(frozen=True)
class CardGroup:
    card_names: frozenset[str]
    identifiable_rects: frozenset[CardRect]


@dataclass
class Deck:
    width: int
    height: int
    cards: frozenset[CardGroup]

    def __init__(self, width_mm: int, height_mm: int, cards: frozenset[CardGroup]):
        self.width = width_mm * ARBITRARY_ZOOM_FACTOR
        self.height = height_mm * ARBITRARY_ZOOM_FACTOR
        self.cards = cards
