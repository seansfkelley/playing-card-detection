from typing import Iterable, Tuple
import numpy as np
import cv2

Image = np.ndarray


def show_images_in_windows(images: Iterable[Tuple[str, Image]]):
    for name, content in images:
        cv2.imshow(name, content)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    dv2.waitKey(1)
