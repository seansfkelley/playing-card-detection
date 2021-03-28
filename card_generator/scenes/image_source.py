from __future__ import annotations

import numpy as np
from dataclasses import dataclass
import os
from glob import glob
import random
import pickle
import matplotlib.image as mpl_image
from cached_property import cached_property
from ..types import Image, ConvexHull

ImageWithHulls = tuple[Image, list[ConvexHull]]

CACHE_FILENAME = "image_source_cache.pickle"


@dataclass
class CardWithMetadata:
    name: str
    image: Image
    hulls: list[ConvexHull]


@dataclass
class BackgroundImageSource:
    _backgrounds: list[Image]

    @staticmethod
    def from_disk(directory: str) -> BackgroundImageSource:
        try:
            backgrounds = BackgroundImageSource._from_cache(directory)
            print(f"loaded {len(backgrounds)} backgrounds from cache")
        except FileNotFoundError:
            backgrounds = BackgroundImageSource._from_source_images(directory)
            print(f"loaded {len(backgrounds)} backgrounds")
            BackgroundImageSource._write_cache(directory, backgrounds)
            print("wrote background cache file to cache.pickle")

        return BackgroundImageSource(_backgrounds=backgrounds)

    @staticmethod
    def _from_cache(directory: str) -> list[Image]:
        with open(os.path.join(directory, CACHE_FILENAME), "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _from_source_images(directory: str) -> list[Image]:
        backgrounds = []
        for f in glob(os.path.join(directory, "**/*.jpg")):
            backgrounds.append((mpl_image.imread(f) * 255).astype(np.uint8))
        return backgrounds

    @staticmethod
    def _write_cache(directory: str, backgrounds: list[Image]):
        with open(os.path.join(directory, CACHE_FILENAME), "wb") as f:
            pickle.dump(backgrounds, f)

    def get_random_background(self) -> Image:
        return random.choice(self._backgrounds)


@dataclass
class CardImageSource:
    _cards: dict[str, list[ImageWithHulls]]

    @staticmethod
    def from_disk(directory: str) -> CardImageSource:
        cards: dict[str, list[ImageWithHulls]] = {}
        for card_dir in glob(os.path.join(directory, "*")):
            card_name = os.path.basename(card_dir)
            cards[card_name] = []
            # all pickles should have images, but nto vice versa, so start with them
            for pickle_file in glob(os.path.join(card_dir, "*.pickle")):
                image_file = os.path.splitext(pickle_file)[0] + ".png"
                if not os.path.exists(image_file):
                    print(f"no corresponding image file for pickle {pickle_file}")
                    continue
                with open(pickle_file, "rb") as f:
                    hulls = pickle.load(f)
                cards[card_name].append(
                    ((mpl_image.imread(image_file) * 255).astype(np.uint8), hulls)
                )

        return CardImageSource(_cards=cards)

    @cached_property
    def _card_names(self) -> list[str]:
        return list(self._cards)

    def get_random_cards(self, n: int) -> list[CardWithMetadata]:
        cards = []
        for c in random.sample(self._card_names, n):
            image, hulls = random.choice(self._cards[c])
            cards.append(CardWithMetadata(name=c, image=image, hulls=hulls))
        return cards
