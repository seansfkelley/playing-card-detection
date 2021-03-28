from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..types import Image, ConvexHull
from ..decks.base import Deck
from .image_source import BackgroundImageSource, CardImageSource

CardInScene = tuple[str, tuple[ConvexHull, ...]]
Scene = tuple[Image, list[CardInScene]]


@dataclass
class SceneGenerationInputs:
    width: int
    height: int
    deck: Deck
    backgrounds: BackgroundImageSource
    cards: CardImageSource


class SceneGenerator(ABC):
    _inputs: SceneGenerationInputs

    def __init__(self, inputs: SceneGenerationInputs):
        self._inputs = inputs

    @abstractmethod
    def generate_scene(self, n: int) -> Scene:
        ...
