from abc import ABC, abstractmethod
from dataclasses import dataclass
from imgaug import Keypoint, KeypointsOnImage
from ..types import Image, ConvexHull
from ..decks.base import Deck
from .image_source import BackgroundImageSource, CardImageSource

CardInScene = tuple[str, tuple[ConvexHull, ...]]
Scene = tuple[Image, list[CardInScene]]


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

    def hull_to_keypoints(self, hull: ConvexHull, dx: int = 0, dy: int = 0):
        # hull is a cv2.Contour, shape : Nx1x2
        return KeypointsOnImage(
            [Keypoint(x=p[0] + dx, y=p[1] + dy) for p in hull.reshape(-1, 2)],
            shape=(self._inputs.height, self._inputs.width, 3),
        )

    @abstractmethod
    def generate_scene(self, n: int) -> Scene:
        ...
