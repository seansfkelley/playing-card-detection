from invoke import Collection
from glob import glob
import matplotlib.image as mpimage
import pickle
import os
import cv2
import shutil
import random
from card_generator.extract_card_from_image import (
    extract as extract_card_from_image,
    ExtractionParameters as ImageExtractionParameters,
)
from card_generator.extract_cards_from_video import (
    extract as extract_cards_from_video,
    ExtractionParameters as VideoExtractionParameters,
)
from card_generator.find_convex_hull import (
    find as find_convex_hull_impl,
    FindParameters as FindConvexHullParameters,
)
from card_generator.decks.base import Deck, CardGroup, ARBITRARY_ZOOM_FACTOR
from card_generator.util import show_images_in_windows
from .util import augment_with_task_decorator
from tasks import test

DATA_DIR = "data"

ns = augment_with_task_decorator(Collection())
ns.add_collection(Collection.from_module(test))
task = ns.task


@task
def fetch_backgrounds(c):
    c.run("rm dtd-r1.0.1.tar.gz")
    c.run(
        "wget https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz",
        pty=True,
    )
    c.run("tar xf dtd-r1.0.1.tar.gz")

    images = []
    for f in glob("dtd/images/*/*.jpg"):
        images.append(mpimage.imread(f))

    print(f"loaded {len(images)} images")

    backgrounds_path = f"{DATA_DIR}/backgrounds.pickle"

    with open(backgrounds_path, "wb") as f:
        pickle.dump(images, f)

    print(f"saved to {backgrounds_path}")


@task
def extract_from_videos(
    c, deck_module_name, extension="mov", indir="data/video", outdir="data/cards"
):
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    deck = _get_deck_by_name(deck_module_name)

    parameters = VideoExtractionParameters(
        card_width=deck.width, card_height=deck.height
    )

    for group in deck.cards:
        for c in group.card_names:
            video_path = os.path.join(indir, f"{c}.{extension}")
            if not os.path.exists(video_path):
                print(f"could not find video for card {c} at {video_path}")
                continue

            result = extract_cards_from_video(
                # reusing parameters here is a little risky, but we shouldn't be mutating it!
                cv2.VideoCapture(video_path),
                parameters,
            )

            output_path = os.path.join(outdir, c)
            os.makedirs(output_path)
            for i, image in enumerate(result):
                cv2.imwrite(os.path.join(output_path, f"{i}.png"), image)

            print(f"extracted {len(result)} images for card {c}")


@task
def find_convex_hulls(c, deck_module_name, directory="data/cards"):
    deck = _get_deck_by_name(deck_module_name)

    for group in deck.cards:
        for card in group.card_names:
            card_path = os.path.join(directory, card)
            if not os.path.exists(card_path):
                print(f"could not find images for card {card} at {card_path}")
                continue

            for card_image_path in glob(os.path.join(card_path, "*.png")):
                image = cv2.imread(card_image_path, cv2.IMREAD_UNCHANGED)
                hulls = []
                for r in group.identifiable_rects:
                    parameters = FindConvexHullParameters(
                        rect=r.as_nparray(
                            card_width=deck.width, card_height=deck.height
                        )
                    )
                    hull, debug_output = find_convex_hull_impl(image, parameters)
                    if hull is None:
                        hulls = []
                        break
                    else:
                        hulls.append(hull)
                if not hulls:
                    print(f"could not find all hulls for {card_image_path}; skipping")
                else:
                    with open(
                        os.path.splitext(card_image_path)[0] + ".pickle", "wb"
                    ) as f:
                        pickle.dump(hulls, f)
