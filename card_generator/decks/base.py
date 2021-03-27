import numpy as np
from typing import ClassVar
from dataclasses import dataclass

# Measurements are provided in millimeters, so scale them up to a reasonable pixel size.
ARBITRARY_SCALE_FACTOR = 4


# unsafe hash -> frozen wouldn't let us define the constructor in this way
@dataclass(unsafe_hash=True)
class IdentifiableCardRect:
    """
    A rectangular area on a card. Origin is in the top left. Use negative numbers to start from the bottom or right.
    """

    left: int
    right: int
    top: int
    bottom: int
    hull_area_range: tuple[int, int]

    def __init__(
        self,
        left_mm: int,
        right_mm: int,
        top_mm: int,
        bottom_mm: int,
        hull_area_range_mm: tuple[int, int],
    ):
        self.left = left_mm * ARBITRARY_SCALE_FACTOR
        self.right = right_mm * ARBITRARY_SCALE_FACTOR
        self.top = top_mm * ARBITRARY_SCALE_FACTOR
        self.bottom = bottom_mm * ARBITRARY_SCALE_FACTOR
        self.hull_area_range = (
            hull_area_range_mm[0] * ARBITRARY_SCALE_FACTOR * ARBITRARY_SCALE_FACTOR,
            hull_area_range_mm[1] * ARBITRARY_SCALE_FACTOR * ARBITRARY_SCALE_FACTOR,
        )

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
    identifiable_rects: frozenset[IdentifiableCardRect]


@dataclass
class Deck:
    width: int
    height: int
    cards: frozenset[CardGroup]

    def __init__(self, width_mm: int, height_mm: int, cards: frozenset[CardGroup]):
        self.width = width_mm * ARBITRARY_SCALE_FACTOR
        self.height = height_mm * ARBITRARY_SCALE_FACTOR
        self.cards = cards
