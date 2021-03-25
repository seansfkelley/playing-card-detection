import numpy as np
import cv2
from dataclasses import dataclass
from ..decks.abstract import Deck, CardGroup, CardRect

ALPHA_BORDER_SIZE = 2
MIN_FOCUS = 100

Image = np.ndarray

@dataclass
class ExtractionParameters:
  card_width: int
  card_height: int

  # TODO: cache dis.
  @property
  def alpha_mask() -> np.ndarray:
    alpha_mask = np.ones((height,width),dtype=np.uint8)*255
    cv2.rectangle(alpha_mask,(0,0),(width-1,height-1),0,ALPHA_BORDER_SIZE)
    cv2.line(alpha_mask,(ALPHA_BORDER_SIZE*3,0),(0,ALPHA_BORDER_SIZE*3),0,ALPHA_BORDER_SIZE)
    cv2.line(alpha_mask,(width-ALPHA_BORDER_SIZE*3,0),(width,ALPHA_BORDER_SIZE*3),0,ALPHA_BORDER_SIZE)
    cv2.line(alpha_mask,(0,height-ALPHA_BORDER_SIZE*3),(ALPHA_BORDER_SIZE*3,height),0,ALPHA_BORDER_SIZE)
    cv2.line(alpha_mask,(width-ALPHA_BORDER_SIZE*3,height),(width,height-ALPHA_BORDER_SIZE*3),0,ALPHA_BORDER_SIZE)
    return alpha_mask

  # TODO: cache dis.
  @property
  def reference_card_rect() -> np.ndarray:
    return np.array([
      [0, 0],
      [self.card_width, 0],
      [self.card_width, self.card_height],
      [0, self.card_height],
    ], dtype=np.float32)

  # TODO: cache dis.
  @property
  def reference_card_rect_rotated() -> np.ndarray:
    np.array([
      [self.card_width, 0],
      [self.card_width, self.card_height],
      [0, self.card_height],
      [0, 0],
    ], dtype=np.float32)

def do_stuff(deck: Deck):
  for group in deck.cards:
    do_stuff_again(group, ExtractionParameters(card_width=deck.width, card_height=deck.height))

def do_stuff_again(group: CardGroup, parameters: ExtractionParameters):
  rects = [r.as_nparray() for r in group.identifiable_rects]

def score_focus(image: Image) -> float:
    # https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    return cv2.Laplacian(image, cv2.CV_64F).var()

def extract_card(image: Image, parameters: ExtractionParameters) -> Image:
  focus = score_focus(image)
  if focus < MIN_FOCUS:
    # TODO: Debug logging?
    print(f'focus too low: {focus}')
    return None

  grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # reduce noise, bu preserve edges
  grayscale = cv2.bilateralFilter(grayscale, 11, 17, 17)

  edged = cv2.Canny(gray, 30, 200)
  # TODO: should the input be copied here? does this mutate inputs?
  contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # assume largest contour (by enclosed area) is the card
  card_contour = min(contours, key=cv2.contourArea)

  min_area_bounding_rect = cv2.minAreaRect(card_contour)
  min_area_bounding_rect_corners = np.int0(cv2.boxPoints(min_area_bounding_rect))
  # make sure the contour is rectangular, i.e., it's very close in size to its own bounding box
  if cv2.contourArea(card_contour) / cv2.contourArea(min_area_bounding_rect_corners) < 0.95:
    return None

  (_, (rect_width, rect_height), _) = min_area_bounding_rect
  if rect_width > rect_height:
      perspective_untransform = cv2.getPerspectiveTransform(
        np.float32(min_area_bounding_rect_corners),
        parameters.reference_card_rect,
      )
  else:
      perspective_untransform = cv2.getPerspectiveTransform(
        np.float32(min_area_bounding_rect_corners),
        parameters.reference_card_rect_rotated,
      )

  #############################################################################
  # u r here
  #############################################################################

  # Determine the warped image by applying the transformation to the image
  imgwarp=cv2.warpPerspective(image,perspective_untransform,(cardW,cardH))
  # Add alpha layer
  imgwarp=cv2.cvtColor(imgwarp,cv2.COLOR_BGR2BGRA)

  # Shape of 'cnt' is (n,1,2), type=int with n = number of points
  # We reshape into (1,n,2), type=float32, before feeding to perspectiveTransform
  cnta=cnt.reshape(1,-1,2).astype(np.float32)
  # Apply the transformation 'Mp' to the contour
  cntwarp=cv2.perspectiveTransform(cnta,perspective_untransform)
  cntwarp=cntwarp.astype(np.int)

  # We build the alpha channel so that we have transparency on the
  # external border of the card
  # First, initialize alpha channel fully transparent
  alphachannel=np.zeros(imgwarp.shape[:2],dtype=np.uint8)
  # Then fill in the contour to make opaque this zone of the card
  cv2.drawContours(alphachannel,cntwarp,0,255,-1)

  # Apply the alphamask onto the alpha channel to clean it
  alphachannel=cv2.bitwise_and(alphachannel,alphamask)

  # Add the alphachannel to the warped image
  imgwarp[:,:,3]=alphachannel

  # Save the image to file
  if output_fn is not None:
      cv2.imwrite(output_fn,imgwarp)
