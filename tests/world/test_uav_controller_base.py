import numpy as np

from src.uavnetsim.geometry.coords import Coords3d
from src.uavnetsim.world import BaseUavController


class DummyUavController(BaseUavController):
    def step(self, time_step: float):
        return time_step


def test_base_uav_controller_exposes_location_array_api():
    controller = DummyUavController(Coords3d(1.0, 2.0, 3.0))

    assert controller.locations == [Coords3d(1.0, 2.0, 3.0)]
    assert np.array_equal(
        controller.get_locations_array(),
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )

    locations_array = np.empty((1, 3), dtype=np.float32)
    controller.update_locations_array(locations_array)
    assert np.array_equal(locations_array, controller.get_locations_array())
