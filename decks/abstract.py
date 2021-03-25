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

@dataclass
class CardGroup:
  card_names: ClassVar[set[str]]
  identifiable_rects: ClassVar[set[CardRect]]

@dataclass
class Deck:
  width: ClassVar[int]
  height: ClassVar[int]
  cards: ClassVar[set[CardGroup]]
