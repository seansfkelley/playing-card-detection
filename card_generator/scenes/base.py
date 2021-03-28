from abc import ABC, abstractmethod
from dataclasses import dataclass
import imgaug as ia
from ..types import Image, ConvexHull
from ..decks.base import Deck
from .image_source import BackgroundImageSource, CardImageSource

Scene = tuple[Image, ia.BoundingBoxesOnImage]


@dataclass
class SceneGenerator(ABC):
    width: int
    height: int
    deck: Deck
    backgrounds: BackgroundImageSource
    cards: CardImageSource

    def __post_init__(self):
        assert self.width > self.deck.width * 2
        assert self.height > self.deck.height * 2

    def hull_to_keypoints(self, hull: ConvexHull, *, dx: int = 0, dy: int = 0):
        return ia.KeypointsOnImage(
            [ia.Keypoint(x=x + dx, y=y + dy) for x, y in hull.reshape(-1, 2)],
            shape=(self.height, self.width, 3),
        )

    @abstractmethod
    def generate_scene(self, n: int) -> Scene:
        ...
