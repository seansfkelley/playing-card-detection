# TODO: in addition to the existing behavior, it should probably also perspective warp to mimic
# the directions that a person might hold the cards in their hard relative to the camera

import math
import numpy as np
from dataclasses import dataclass
import itertools
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from .image_source import CardWithMetadata
from .base import SceneGenerator, Scene
from ..util import show_images_in_windows
from ..types import ConvexHull, Image

MAX_FAN_ANGLE = 15
BOUNDING_BOX_BUFFER = 3


@dataclass
class CardInFan:
    name: str
    image: Image
    hulls: list[ConvexHull]
    keypoints: list[ia.KeypointsOnImage]

    def augment(self, augmentation: iaa.Augmenter):
        aug = augmentation.to_deterministic()
        self.image = aug.augment_image(self.image)
        # do this one at a time in order to ensure determinism is the same for all of them
        self.keypoints = [aug.augment_keypoints(k) for k in self.keypoints]

    def get_bounding_boxes(self, width: int, height: int) -> list[ia.BoundingBox]:
        bounding_boxes = []
        for keypoints in self.keypoints:
            keypoints_x = [k.x for k in keypoints]
            min_x = max(0, int(min(keypoints_x) - BOUNDING_BOX_BUFFER))
            max_x = min(width, int(max(keypoints_x) + BOUNDING_BOX_BUFFER))

            keypoints_y = [k.y for k in keypoints]
            min_y = max(0, int(min(keypoints_y) - BOUNDING_BOX_BUFFER))
            max_y = min(height, int(max(keypoints_y) + BOUNDING_BOX_BUFFER))

            bounding_boxes.append(
                ia.BoundingBox(x1=min_x, y1=min_y, x2=max_x, y2=max_y, label=self.name)
            )
        return bounding_boxes


class FannedSceneGenerator(SceneGenerator):
    def generate_scene(self, n: int) -> Scene:
        cards = self.cards.get_random_cards(n)
        cards_in_fan = []
        for c in cards:
            # new empty canvas
            image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            # TODO: This can be probably be a canvas based solely on the size of the card,
            # and then clamped to the desired size later.
            top = int(self.height / 2 - self.deck.height / 2)
            left = int(self.width / 2 - self.deck.width / 2)
            # paste the card into the middle
            image[top : top + self.deck.height, left : left + self.deck.width] = (
                # TODO: ugh, why is one an integer and the other a float? shouldn't they be the same?
                c.image
                * 255
            )
            cards_in_fan.append(
                CardInFan(
                    name=c.name,
                    image=image,
                    hulls=c.hulls,
                    keypoints=[
                        self.hull_to_keypoints(h, dx=left, dy=top) for h in c.hulls
                    ],
                )
            )

        resize_background = iaa.Resize({"height": self.height, "width": self.width})

        result = resize_background.augment_image(
            self.backgrounds.get_random_background()
        )
        for i, c in enumerate(cards_in_fan):
            # TODO: no idea what's going on here
            mask = c.image[:, :, 3]
            mask = np.stack([mask] * 3, -1)
            result = np.where(mask, c.image[:, :, :3], result)

            augmentation = iaa.Sequential(
                [
                    self._compute_fan_augmentation(c.hulls),
                    self._compute_jitter_augmentation(),
                ]
            )

            for later_card in cards_in_fan[i + 1 :]:
                later_card.augment(augmentation)

        # TODO: reject any fans that obscure both hulls by n%
        # TODO: remove any bounding boxes that are not visible
        # TODO: shrink hulls?

        return (
            result,
            list(
                itertools.chain.from_iterable(
                    c.get_bounding_boxes(self.width, self.height) for c in cards_in_fan
                )
            ),
        )

    def _compute_fan_augmentation(self, hulls: list[ConvexHull]):
        leftmost_hull = min(hulls, key=lambda h: min(h[:, :, 1]))
        rightmost_hull_point = max(leftmost_hull, key=lambda p: p[:, 1])[0]
        cosine = (self.deck.height - rightmost_hull_point[0]) / math.hypot(
            rightmost_hull_point[1],
            self.deck.height - rightmost_hull_point[0],
        )
        min_degrees = math.degrees(math.acos(cosine))
        min_degrees, max_degrees = min_degrees * 0.9, min(
            min_degrees * 1.3, MAX_FAN_ANGLE
        )

        # we want to rotate from the bottom-left corner, so have to translate back and forth
        return iaa.Sequential(
            [
                iaa.Affine(
                    translate_px={
                        "x": int(self.deck.width / 2),
                        "y": int(-self.deck.height / 2),
                    }
                ),
                # 0.9 -> sometimes players hold their cards slightly overlapping
                # 1.2 -> but more often they leave a lot of extra space
                iaa.Affine(
                    rotate=iap.Normal(
                        (min_degrees + max_degrees) / 2, (max_degrees - min_degrees) / 2
                    )
                ),
                iaa.Affine(
                    translate_px={
                        "x": int(-self.deck.width / 2),
                        "y": int(self.deck.height / 2),
                    }
                ),
            ]
        )

    def _compute_jitter_augmentation(self):
        return iaa.Affine(
            translate_px={
                "x": iap.Normal(0, int(self.deck.width * 0.03)),
                "y": iap.Normal(0, int(self.deck.height * 0.05)),
            }
        )
