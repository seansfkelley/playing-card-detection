from typing import Optional
import numpy as np
import cv2
from dataclasses import dataclass, field

Image = np.ndarray


@dataclass
class FindParameters:
    rect: np.ndarray
    # heuristic: how small is too small for the contour? depends on zoom factor.
    min_contour_area: int = 30
    min_contour_solidity: float = 0.3
    contour_centroid_vertical_window: float = 0.8
    contour_centroid_horizontal_window: float = 0.8
    # TODO: pick better numbers and also these should be part of a CardRect
    # this also depends on the zoom factor
    min_hull_area: int = 300
    max_hull_area: int = 4250

    def __post_init__(self):
        assert 0 <= self.min_contour_solidity <= 1
        assert 0 <= self.contour_centroid_vertical_window <= 1
        assert 0 <= self.contour_centroid_horizontal_window <= 1


@dataclass
class FindConvexHullDebugOutput:
    grayscale: Optional[Image] = None
    # TODO: What does thld stand for and why is it different than just edged?
    thld: Optional[Image] = None
    accepted_contours: Optional[Image] = None
    rejected_contours: Optional[Image] = None
    hull_size: Optional[tuple[bool, int]] = None


def find(
    image: Image, parameters: FindParameters
) -> tuple[Optional[np.ndarray], FindConvexHullDebugOutput]:
    assert parameters.rect.shape == (4, 2)
    assert parameters.rect.dtype == np.int

    debug_output = FindConvexHullDebugOutput()

    kernel = np.ones((3, 3), np.uint8)

    x1 = int(parameters.rect[0][0])
    y1 = int(parameters.rect[0][1])
    x2 = int(parameters.rect[2][0])
    y2 = int(parameters.rect[2][1])
    width = x2 - x1
    height = y2 - y1

    grayscale = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    debug_output.grayscale = grayscale

    thld = cv2.Canny(grayscale, 30, 200)
    thld = cv2.dilate(thld, kernel, iterations=1)
    debug_output.thld = thld

    debug_output.accepted_contours = np.zeros_like(grayscale)
    debug_output.rejected_contours = np.zeros_like(grayscale)

    contours, _ = cv2.findContours(thld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    merged_contour = None

    for c in contours:
        area = cv2.contourArea(c)
        solidity = float(area) / cv2.contourArea(cv2.convexHull(c))

        moments = cv2.moments(c)
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])

        centroid_in_bounds = (
            abs(width / 2 - centroid_x)
            < width * parameters.contour_centroid_horizontal_window / 2
            and abs(height / 2 - centroid_y)
            < height * parameters.contour_centroid_vertical_window / 2
        )

        if (
            area >= parameters.min_contour_area
            and centroid_in_bounds
            and solidity >= parameters.min_contour_solidity
        ):
            cv2.drawContours(debug_output.accepted_contours, [c], 0, 255, 1)
            if merged_contour is None:
                merged_contour = c
            else:
                merged_contour = np.concatenate((merged_contour, c))
        else:
            cv2.drawContours(debug_output.rejected_contours, [c], 0, 255, 1)

    if merged_contour is not None:
        hull = cv2.convexHull(merged_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area < parameters.min_hull_area or hull_area > parameters.max_hull_area:
            hull = None
            debug_output.hull_size = (False, hull_area)
        else:
            # translate back into the coordinate space of the original image
            hull += parameters.rect[0]
            debug_output.hull_size = (True, hull_area)
    else:
        hull = None

    return hull, debug_output
