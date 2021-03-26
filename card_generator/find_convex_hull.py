from typing import Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass

Image = np.ndarray


@dataclass
class FindParameters:
    rect: np.ndarray
    # heuristic: how small is too small for the contour? depends on zoom factor.
    min_area: int = 30
    min_solidity: float = 0.3


@dataclass
class FindConvexHullDebugOutput:
    grayscale: Optional[Image] = None
    # TODO: What does thld stand for and why is it different than just edged?
    thld: Optional[Image] = None


def find(
    image: Image, parameters: FindParameters
) -> Tuple[Optional[np.ndarray], FindConvexHullDebugOutput]:
    assert parameters.rect.shape == (4, 2)
    assert parameters.rect.dtype == np.int

    debug_output = FindConvexHullDebugOutput()

    kernel = np.ones((3, 3), np.uint8)

    x1 = int(parameters.rect[0][0])
    y1 = int(parameters.rect[0][1])
    x2 = int(parameters.rect[2][0])
    y2 = int(parameters.rect[2][1])
    w = x2 - x1
    h = y2 - y1

    zone = image[y1:y2, x1:x2].copy()

    strange_contours = np.zeros_like(zone)
    grayscale = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
    debug_output.grayscale = grayscale

    thld = cv2.Canny(grayscale, 30, 200)
    thld = cv2.dilate(thld, kernel, iterations=1)
    debug_output.thld = thld

    contours, _ = cv2.findContours(thld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    concat_contour = None

    ok = True
    for c in contours:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)

        # TODO: skip contours whose hull includes any of the four corner points (outlines)

        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area
        # Determine the center of gravity (cx,cy) of the contour
        M = cv2.moments(c)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        #  abs(w/2-cx)<w*0.3 and abs(h/2-cy)<h*0.4 : TWEAK, the idea here is to keep only the contours which are closed to the center of the zone
        if (
            area >= parameters.min_area
            and abs(w / 2 - cx) < w * 0.4
            and abs(h / 2 - cy) < h * 0.5
            and solidity > parameters.min_solidity
        ):
            # if debug != "no":
            #     cv2.drawContours(zone, [c], 0, (255, 0, 0), -1)
            if concat_contour is None:
                concat_contour = c
            else:
                concat_contour = np.concatenate((concat_contour, c))
        # if debug != "no" and solidity <= min_solidity:
        #     print("Solidity", solidity)
        #     cv2.drawContours(strange_contours, [c], 0, 255, 2)
        #     cv2.imshow("Strange contours", strange_contours)

    if concat_contour is not None:
        # At this point, we suppose that 'concat_contour' contains only the contours corresponding the value and suit symbols
        # We can now determine the hull
        hull = cv2.convexHull(concat_contour)
        hull_area = cv2.contourArea(hull)
        # If the area of the hull is to small or too big, there may be a problem
        min_hull_area = 940  # TWEAK, deck and 'zoom' dependant
        max_hull_area = 4250  # TWEAK, deck and 'zoom' dependant
        if hull_area < min_hull_area or hull_area > max_hull_area:
            ok = False
            # if debug != "no":
            #     print("Hull area=", hull_area, "too large or too small")
        # So far, the coordinates of the hull are relative to 'zone'
        # We need the coordinates relative to the image -> 'hull_in_img'
        hull_in_img = hull + parameters.rect[0]

    else:
        ok = False

    # if debug != "no":
    #     if concat_contour is not None:
    #         cv2.drawContours(zone, [hull], 0, (0, 255, 0), 1)
    #         cv2.drawContours(img, [hull_in_img], 0, (0, 255, 0), 1)
    #     cv2.imshow("Zone", zone)
    #     cv2.imshow("Image", img)
    #     if ok and debug != "pause_always":
    #         key = cv2.waitKey(1)
    #     else:
    #         key = cv2.waitKey(0)
    #     if key == 27:
    #         return None
    # if ok == False:

    #     return None

    if ok:
        return hull_in_img, debug_output
    else:
        return None, debug_output
