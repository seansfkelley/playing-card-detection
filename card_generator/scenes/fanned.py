# TODO: in addition to the existing behavior, it should probably also perspective warp to mimic
# the directions that a person might hold the cards in their hard relative to the camera

import math
import numpy as np
from dataclasses import dataclass, field
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
    fan_degrees: int
    keypoint_groups: list[ia.KeypointsOnImage]

    def augment(self, augmentation: iaa.Augmenter):
        aug = augmentation.to_deterministic()
        self.image = aug.augment_image(self.image)
        # do this one at a time in order to ensure determinism is the same for all of them
        self.keypoint_groups = [aug.augment_keypoints(k) for k in self.keypoint_groups]

    def get_bounding_boxes(self, width: int, height: int) -> list[ia.BoundingBox]:
        bounding_boxes = []
        for group in self.keypoint_groups:
            group_x = [k.x for k in group]
            min_x = max(0, int(min(group_x) - BOUNDING_BOX_BUFFER))
            max_x = min(width, int(max(group_x) + BOUNDING_BOX_BUFFER))

            group_y = [k.y for k in group]
            min_y = max(0, int(min(group_y) - BOUNDING_BOX_BUFFER))
            max_y = min(height, int(max(group_y) + BOUNDING_BOX_BUFFER))

            bounding_boxes.append(
                ia.BoundingBox(x1=min_x, y1=min_y, x2=max_x, y2=max_y, label=self.name)
            )
        return bounding_boxes


class FannedSceneGenerator(SceneGenerator):
    def generate_scene(self, n: int) -> Scene:
        cards_in_fan = [
            self._generate_card_in_fan(c) for c in self.cards.get_random_cards(n)
        ]

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
                    self._compute_fan_augmentation(c.fan_degrees),
                    self._compute_jitter_augmentation(),
                ]
            )

            for later_card in cards_in_fan[i + 1 :]:
                later_card.augment(augmentation)

        # TODO: reject any fans that obscure all keypoint_groups by n%
        # TODO: remove any bounding boxes that are not visible
        # TODO: should shrink bounding boxes that are partially obscured?

        return (
            result,
            list(
                itertools.chain.from_iterable(
                    c.get_bounding_boxes(self.width, self.height) for c in cards_in_fan
                )
            ),
        )

    def _generate_card_in_fan(self, card: CardWithMetadata):
        # new empty canvas
        image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        # TODO: This can be probably be a canvas based solely on the size of the card,
        # and then clamped to the desired size later.
        top = int(self.height / 2 - self.deck.height / 2)
        left = int(self.width / 2 - self.deck.width / 2)
        # paste the card into the middle
        image[top : top + self.deck.height, left : left + self.deck.width] = card.image
        return CardInFan(
            name=card.name,
            image=image,
            fan_degrees=self._get_fan_degrees(card.hulls),
            keypoint_groups=[
                self.hull_to_keypoints(h, dx=left, dy=top) for h in card.hulls
            ],
        )

    def _get_fan_degrees(self, card_hulls: list[ConvexHull]):
        # TODO: -1, 2 should reshape this to x, y rather than y, x, right?
        leftmost_hull = min(card_hulls, key=lambda h: min(h[:, :, 1])).reshape(-1, 2)
        max_x = max(leftmost_hull[:, 0])
        max_y = self.deck.height - max(leftmost_hull[:, 1])
        cosine = max_y / math.hypot(max_x, max_y)
        return math.degrees(math.acos(cosine))

    def _compute_fan_augmentation(self, degrees: int):
        # 0.9 -> sometimes players hold their cards slightly overlapping
        # 1.3 -> but more often they leave a lot of extra space
        min_degrees, max_degrees = degrees * 0.9, min(degrees * 1.3, MAX_FAN_ANGLE)

        # we want to rotate from the bottom-left corner, so have to translate back and forth
        return iaa.Sequential(
            [
                iaa.Affine(
                    translate_px={
                        "x": int(self.deck.width / 2),
                        "y": int(-self.deck.height / 2),
                    }
                ),
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
