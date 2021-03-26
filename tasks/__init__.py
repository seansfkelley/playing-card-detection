from invoke import task
from glob import glob
import matplotlib.image as mpimage
import pickle
import os
import cv2
import shutil
import importlib
from card_generator.extract_card_from_image import (
    extract as extract_card_from_image,
    ExtractionParameters as ImageExtractionParameters,
)
from card_generator.extract_cards_from_video import (
    extract as extract_cards_from_video,
    ExtractionParameters as VideoExtractionParameters,
)
from card_generator.decks.base import Deck, CardGroup, ARBITRARY_ZOOM_FACTOR
from card_generator.util import show_images_in_windows

DATA_DIR = "data"
BACKGROUNDS_FILE = f"{DATA_DIR}/backgrounds.pickle"


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

    with open(BACKGROUNDS_FILE, "wb") as f:
        pickle.dump(images, f)

    print(f"saved to {BACKGROUNDS_FILE}")


@task
def demo_extract_image(
    c, infile, width, height, outfile="example/output/extracted_card.png", debug=False
):
    result, debug_output = extract_card_from_image(
        cv2.imread(infile),
        ImageExtractionParameters(
            card_width=int(width) * ARBITRARY_ZOOM_FACTOR,
            card_height=int(height) * ARBITRARY_ZOOM_FACTOR,
        ),
    )

    if result is not None and not debug:
        print(f"success; extracted image to {outfile}")
        cv2.imwrite(outfile, result)
    else:
        print("focus:", debug_output.focus)
        show_images_in_windows(
            ("Grayscale", debug_output.grayscale),
            ("Edged", debug_output.edged),
            ("Card Contour", debug_output.card_contour),
            ("Alpha Channel", debug_output.alpha_channel),
            ("Result", debug_output.extracted_card),
        )


@task
def demo_extract_video(c, infile, width, height, outdir="example/output/frames/"):
    result = extract_cards_from_video(
        cv2.VideoCapture(infile),
        VideoExtractionParameters(
            card_width=int(width) * ARBITRARY_ZOOM_FACTOR,
            card_height=int(height) * ARBITRARY_ZOOM_FACTOR,
        ),
    )

    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    for i, image in enumerate(result):
        cv2.imwrite(os.path.join(outdir, f"{i}.png"), image)

    print(f"success; extracted {len(result)} images to {outdir}")


@task
def extract_all_videos(
    c, deck_module_name, extension="mov", indir="data/video", outdir="data/cards"
):
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

    # spooky!
    deck: Deck = importlib.import_module(
        f"card_generator.decks.{deck_module_name}"
    ).DECK

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
