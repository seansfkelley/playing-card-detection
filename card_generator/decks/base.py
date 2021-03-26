import numpy as np
from typing import ClassVar
from dataclasses import dataclass

# I'm actually not totally sure what this is for, but without it, intermediate images are tiny!
ARBITRARY_ZOOM_FACTOR = 4


# This is never mutated except in post-init. `frozen` would still break that.
@dataclass(unsafe_hash=True)
class CardRect:
    """
    A rectangular area on a card. Origin is in the top left. Use negative numbers to start from the bottom or right.
    """

    left: int
    right: int
    top: int
    bottom: int

    def __post_init__(self):
        self.left *= ARBITRARY_ZOOM_FACTOR
        self.right *= ARBITRARY_ZOOM_FACTOR
        self.top *= ARBITRARY_ZOOM_FACTOR
        self.bottom *= ARBITRARY_ZOOM_FACTOR

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


@dataclass
class Deck:
    width: int
    height: int
    cards: frozenset[CardGroup]

    def __post_init__(self):
        self.width *= ARBITRARY_ZOOM_FACTOR
        self.height *= ARBITRARY_ZOOM_FACTOR
