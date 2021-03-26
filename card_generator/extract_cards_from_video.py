import itertools
from dataclasses import dataclass
import cv2
from .extract_card_from_image import (
    extract as extract_image,
    ExtractionParameters as ImageExtractionParameters,
)
from tqdm import tqdm


@dataclass
class ExtractionParameters(ImageExtractionParameters):
    skip_frames: int = 15


def extract(video: cv2.VideoCapture, parameters: ExtractionParameters):
    extracted_images = []

    for frame_number in tqdm(itertools.count()):
        success, frame = video.read()
        if not success:
            break

        if frame_number % parameters.skip_frames == 0:
            continue

        result, _ = extract_image(frame, parameters)
        if result is not None:
            extracted_images.append(result)

    return extracted_images
