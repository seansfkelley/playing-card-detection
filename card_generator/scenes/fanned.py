# TODO: in addition to the existing behavior, it should probably also perspective warp to mimic
# the directions that a person might hold the cards in their hard relative to the camera

import numpy as np
from imgaug import augmenters
from .base import SceneGenerator, Scene
from ..util import show_images_in_windows

trans_rot1 = augmenters.Sequential(
    [
        augmenters.Affine(translate_px={"x": (10, 20)}),
        augmenters.Affine(rotate=(22, 30)),
    ]
)


class FannedSceneGenerator(SceneGenerator):
    def generate_scene(self, n: int) -> Scene:
        cards = self.cards.get_random_cards(n)

        images = []
        for _, card_image, _ in cards:
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
                card_image
                * 255
            )
            images.append(image)

        resize_background = augmenters.Resize(
            {"height": self.height, "width": self.width}
        )

        result = resize_background.augment_image(
            self.backgrounds.get_random_background()
        )
        for i in images:
            # TODO: no idea what's going on here
            mask = i[:, :, 3]
            mask = np.stack([mask] * 3, -1)
            result = np.where(mask, i[:, :, :3], result)

        return result

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
