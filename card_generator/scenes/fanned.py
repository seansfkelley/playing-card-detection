# TODO: in addition to the existing behavior, it should probably also perspective warp to mimic
# the directions that a person might hold the cards in their hard relative to the camera

import math
import numpy as np
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from .base import SceneGenerator, Scene
from ..util import show_images_in_windows
from ..types import ConvexHull

MAX_FAN_ANGLE = 15


class FannedSceneGenerator(SceneGenerator):
    def generate_scene(self, n: int) -> Scene:
        cards = self.cards.get_random_cards(n)

        images = []
        for c in cards:
            # new empty canvas
            image = np.zeros((self.height, self.width, 4), dtype=np.uint8)
            # TODO: This can be probably be a canvas based solely on the size of the card,
            # and then clamped to the desired size later.
            bottom = int(self.height / 2 - self.deck.height / 2)
            left = int(self.width / 2 - self.deck.width / 2)
            # paste the card into the middle
            image[
                bottom : bottom + self.deck.height,
                left : left + self.deck.width,
            ] = (
                # TODO: ugh, why is one an integer and the other a float? shouldn't they be the same?
                c.image
                * 255
            )
            images.append(image)

        resize_background = iaa.Resize({"height": self.height, "width": self.width})

        result = resize_background.augment_image(
            self.backgrounds.get_random_background()
        )
        for i, img in enumerate(images):
            # TODO: no idea what's going on here
            mask = img[:, :, 3]
            mask = np.stack([mask] * 3, -1)
            result = np.where(mask, img[:, :, :3], result)

            augmentation = iaa.Sequential(
                [
                    self._compute_fan_augmentation(cards[i].hulls),
                    self._compute_jitter_augmentation(),
                ]
            )

            for j in range(i + 1, len(images)):
                images[j] = augmentation.augment_image(images[j])

        # TODO: generate keypoints and augment them too
        # TODO: reject any fans that obscure both hulls by n%

        return result

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

    def create3CardsScene():
        kpsa1 = hull_to_kps(hulla1, decalX3, decalY3)
        kpsb1 = hull_to_kps(hullb1, decalX3, decalY3)
        kpsa2 = hull_to_kps(hulla2, decalX3, decalY3)
        kpsb2 = hull_to_kps(hullb2, decalX3, decalY3)
        kpsa3 = hull_to_kps(hulla3, decalX3, decalY3)
        kpsb3 = hull_to_kps(hullb3, decalX3, decalY3)
        self.img3 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img3[decalY3 : decalY3 + cardH, decalX3 : decalX3 + cardW, :] = img3
        self.img3, self.lkps3, self.bbs3 = augment(
            self.img3, [cardKP, kpsa3, kpsb3], trans_rot1
        )
        self.img2 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img2[decalY3 : decalY3 + cardH, decalX3 : decalX3 + cardW, :] = img2
        self.img2, self.lkps2, self.bbs2 = augment(
            self.img2, [cardKP, kpsa2, kpsb2], trans_rot2
        )
        self.img1 = np.zeros((imgH, imgW, 4), dtype=np.uint8)
        self.img1[decalY3 : decalY3 + cardH, decalX3 : decalX3 + cardW, :] = img1

        while True:
            det_transform_3cards = transform_3cards.to_deterministic()
            _img3, _lkps3, self.bbs3 = augment(
                self.img3, self.lkps3, det_transform_3cards, False
            )
            if _img3 is None:
                continue
            _img2, _lkps2, self.bbs2 = augment(
                self.img2, self.lkps2, det_transform_3cards, False
            )
            if _img2 is None:
                continue
            _img1, self.lkps1, self.bbs1 = augment(
                self.img1, [cardKP, kpsa1, kpsb1], det_transform_3cards, False
            )
            if _img1 is None:
                continue
            break
        self.img3 = _img3
        self.lkps3 = _lkps3
        self.img2 = _img2
        self.lkps2 = _lkps2
        self.img1 = _img1

        self.class1 = class1
        self.class2 = class2
        self.class3 = class3
        self.listbba = [
            BBA(self.bbs1[0], class1),
            BBA(self.bbs2[0], class2),
            BBA(self.bbs3[0], class3),
            BBA(self.bbs3[1], class3),
        ]

        # Construct final image of the scene by superimposing: bg, img1, img2 and img3
        self.bg = scaleBg.augment_image(bg)
        mask1 = self.img1[:, :, 3]
        self.mask1 = np.stack([mask1] * 3, -1)
        self.final = np.where(self.mask1, self.img1[:, :, 0:3], self.bg)
        mask2 = self.img2[:, :, 3]
        self.mask2 = np.stack([mask2] * 3, -1)
        self.final = np.where(self.mask2, self.img2[:, :, 0:3], self.final)
        mask3 = self.img3[:, :, 3]
        self.mask3 = np.stack([mask3] * 3, -1)
        self.final = np.where(self.mask3, self.img3[:, :, 0:3], self.final)
