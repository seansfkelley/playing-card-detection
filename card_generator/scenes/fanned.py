# TODO: in addition to the existing behavior, it should probably also perspective warp to mimic
# the directions that a person might hold the cards in their hard relative to the camera

from .base import SceneGenerator, Scene


class FannedSceneGenerator(SceneGenerator):
    def generate_scene(self, n: int) -> Scene:
        pass
