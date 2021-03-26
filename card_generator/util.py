from typing import Tuple, Optional
import numpy as np
import cv2

Image = np.ndarray


def show_images_in_windows(*images: Tuple[str, Optional[Image]]):
    did_show = False

    for name, content in images:
        if content is not None:
            did_show = True
            cv2.imshow(name, content)

    if did_show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
