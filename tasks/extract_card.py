from typing import Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass
from ..decks.abstract import Deck, CardGroup, CardRect
from cached_property import cached_property

ALPHA_BORDER_SIZE = 2
MIN_FOCUS = 100

Image = np.ndarray


@dataclass
class ExtractionParameters:
    card_width: int
    card_height: int

    @cached_property
    def alpha_mask() -> np.ndarray:
        alpha_mask = np.ones((height, width), dtype=np.uint8) * 255
        cv2.rectangle(alpha_mask, (0, 0), (width - 1, height - 1), 0, ALPHA_BORDER_SIZE)
        cv2.line(
            alpha_mask,
            (ALPHA_BORDER_SIZE * 3, 0),
            (0, ALPHA_BORDER_SIZE * 3),
            0,
            ALPHA_BORDER_SIZE,
        )
        cv2.line(
            alpha_mask,
            (width - ALPHA_BORDER_SIZE * 3, 0),
            (width, ALPHA_BORDER_SIZE * 3),
            0,
            ALPHA_BORDER_SIZE,
        )
        cv2.line(
            alpha_mask,
            (0, height - ALPHA_BORDER_SIZE * 3),
            (ALPHA_BORDER_SIZE * 3, height),
            0,
            ALPHA_BORDER_SIZE,
        )
        cv2.line(
            alpha_mask,
            (width - ALPHA_BORDER_SIZE * 3, height),
            (width, height - ALPHA_BORDER_SIZE * 3),
            0,
            ALPHA_BORDER_SIZE,
        )
        return alpha_mask

    @cached_property
    def reference_card_rect() -> np.ndarray:
        return np.array(
            [
                [0, 0],
                [self.card_width, 0],
                [self.card_width, self.card_height],
                [0, self.card_height],
            ],
            dtype=np.float32,
        )

    @cached_property
    def reference_card_rect_rotated() -> np.ndarray:
        np.array(
            [
                [self.card_width, 0],
                [self.card_width, self.card_height],
                [0, self.card_height],
                [0, 0],
            ],
            dtype=np.float32,
        )


def do_stuff(deck: Deck):
    for group in deck.cards:
        do_stuff_again(
            group, ExtractionParameters(card_width=deck.width, card_height=deck.height)
        )


def do_stuff_again(group: CardGroup, parameters: ExtractionParameters):
    rects = [r.as_nparray() for r in group.identifiable_rects]


def score_focus(image: Image) -> float:
    # https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    return cv2.Laplacian(image, cv2.CV_64F).var()


@dataclass
class ExtractCardDebugOutput:
    focus: Optional[int]
    grayscale: Optional[Image]
    edged: Optional[Image]
    card_contour: Optional[Image]
    alpha_channel: Optional[Image]
    extracted_card: Optional[Image]


def extract_card(
    image: Image, parameters: ExtractionParameters
) -> Tuple[Optional[Image], ExtractCardDebugOutput]:
    debug_output = ExtractCardDebugOutput()

    focus = score_focus(image)
    debug_output.focus = focus
    if focus < MIN_FOCUS:
        return None, debug_output

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # reduce noise, but preserve edges
    grayscale = cv2.bilateralFilter(grayscale, 11, 17, 17)
    debug_output.grayscale = grayscale

    edged = cv2.Canny(gray, 30, 200)
    debug_output.edged = edged

    # TODO: should the input be copied here? does this mutate inputs?
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # assume largest contour (by enclosed area) is the card
    card_contour = min(contours, key=cv2.contourArea)

    min_area_bounding_rect = cv2.minAreaRect(card_contour)
    min_area_bounding_rect_corners = np.int0(cv2.boxPoints(min_area_bounding_rect))

    debug_card_contour_image = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(
        debug_card_contour_image, [min_area_bounding_rect_corners], 0, (0, 0, 255), 3
    )
    cv2.drawContours(debug_card_contour_image, [card_contour], 0, (0, 255, 0), -1)
    debug_output.card_contour = debug_card_contour_image

    # make sure the contour is rectangular, i.e., it's very close in size to its own bounding box
    if (
        cv2.contourArea(card_contour) / cv2.contourArea(min_area_bounding_rect_corners)
        < 0.95
    ):
        return None, debug_output

    (_, (rect_width, rect_height), _) = min_area_bounding_rect
    if rect_width > rect_height:
        undo_perspective_transform = cv2.getPerspectiveTransform(
            np.float32(min_area_bounding_rect_corners),
            parameters.reference_card_rect,
        )
    else:
        undo_perspective_transform = cv2.getPerspectiveTransform(
            np.float32(min_area_bounding_rect_corners),
            parameters.reference_card_rect_rotated,
        )

    normalized_image = cv2.warpPerspective(
        image,
        undo_perspective_transform,
        (parameters.card_width, parameters.card_height),
    )
    normalized_image = cv2.cvtColor(normalized_image, cv2.COLOR_BGR2BGRA)

    # reshape from (n, 1, 2) to (1, n, 2) to work with the transform
    normalized_card_contour = cv2.perspectiveTransform(
        card_contour.reshape(1, -1, 2).astype(np.float32), undo_perspective_transform
    ).astype(np.int)

    alpha_channel = np.zeros(normalized_image.shape[:2], dtype=np.uint8)
    cv2.drawContours(alpha_channel, normalized_card_contour, 0, 255, -1)
    alpha_channel = cv2.bitwise_and(alpha_channel, alphamask)
    normalized_image[:, :, 3] = alpha_channel

    debug_output.extracted_card = normalized_image

    return normalized_image, debug_card_contour_image
