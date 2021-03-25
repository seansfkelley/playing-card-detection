import numpy as np
from typing import ClassVar
from dataclasses import dataclass

@dataclass
class CardRect:
  """
  A rectangular area on a card. Origin is in the top left. Use negative numbers to start from the bottom or right.
  """

  left: int
  right: int
  top: int
  bottom: int

  def as_nparray(self, card_width: int, card_height: int):
    left, right = sorted((self.left + card_width) % card_width, (self.right + card_width) % card_width)
    top, bottom = sorted((self.top + card_height) % card_height, (self.bottom + card_height) % card_height)
    return np.array([
      [left, top],
      [right, top],
      [right, bottom],
      [left, bottom],
    ], dtype=np.float32)

@dataclass
class CardGroup:
  card_names: ClassVar[set[str]]
  identifiable_rects: ClassVar[set[CardRect]]

@dataclass
class Deck:
  width: ClassVar[int]
  height: ClassVar[int]
  cards: ClassVar[set[CardGroup]]
