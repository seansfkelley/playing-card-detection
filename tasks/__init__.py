from invoke import task
from glob import glob
import matplotlib.image as mpimage
import pickle
import cv2
from card_generator.extract_card import (
    extract_card as extract_card_impl,
    ExtractionParameters,
)
from card_generator.decks import TAROT_DECK, ARBITRARY_ZOOM_FACTOR
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
def extract_card(c, infile, width, height, debug=False):
    result, debug_output = extract_card_impl(
        cv2.imread(infile),
        ExtractionParameters(
            card_width=int(width) * ARBITRARY_ZOOM_FACTOR,
            card_height=int(height) * ARBITRARY_ZOOM_FACTOR,
        ),
    )

    if result is not None and not debug:
        print("success; please locate preview window")
        show_images_in_windows(("Result", result))
    else:
        print("focus:", debug_output.focus)
        show_images_in_windows(
            ("Grayscale", debug_output.grayscale),
            ("Edged", debug_output.edged),
            ("Card Contour", debug_output.card_contour),
            ("Alpha Channel", debug_output.alpha_channel),
            ("Result", debug_output.extracted_card),
        )
