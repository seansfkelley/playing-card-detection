from typing import Tuple, Optional
import numpy as np
import cv2

Image = np.ndarray


def show_images_in_windows(*images: Tuple[str, Optional[Image]]):
    for name, content in images:
        if content is not None:
            cv2.imshow(name, content)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
