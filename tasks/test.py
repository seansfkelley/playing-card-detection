from invoke import task
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


@task
def extract_image(
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
def extract_video(c, infile, width, height, outdir="example/output/frames/"):
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
def spot_check_rects(c, deck_module_name, directory="data/cards", n=5):
    deck = _get_deck_by_name(deck_module_name)

    all_extracted_images = list(glob(os.path.join(directory, "*", "*.png")))
    selection = random.sample(all_extracted_images, min(len(all_extracted_images), n))

    images = []

    for path in selection:
        *_, card_name, _ = path.split("/")
        for g in deck.cards:
            if card_name in g.card_names:
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                for r in g.identifiable_rects:
                    hull, _ = find_convex_hull_impl(
                        image,
                        FindConvexHullParameters(
                            rect=r.as_nparray(deck.width, deck.height)
                        ),
                    )
                    cv2.drawContours(
                        image,
                        [r.as_nparray(deck.width, deck.height)],
                        0,
                        (0, 255, 0) if hull is not None else (0, 0, 255),
                        1,
                    )
                    if hull is not None:
                        cv2.drawContours(
                            image,
                            [hull],
                            0,
                            (255, 0, 0),
                            1,
                        )
                images.append((path, image))
                break

    show_images_in_windows(*images)
