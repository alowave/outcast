"""Obstacle-Aware Voronoi Mobility Simulation and Playback Model.

Manages graph-constrained user mobility by generating, caching, and streaming
shortest-path trajectories across map environments. Restricts agent movement paths
to structural Voronoi road networks wrapped around obstacle topologies.
"""

from _pickle import UnpicklingError
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.outcast.geometry.coords import Coords3d
from src.outcast.world.user_model.base import UserModel
from src.outcast.world.user_model.paths import build_voronoi_graph


@dataclass(slots=True)
class ObstacleMobilityCfg:
    duration: int = 3600
    speed_range: tuple[float, float] = (10.0, 20.0)
    pause_range: tuple[float, float] = (0.0, 0.5)

    save_name: str = "Poznan_U8_S20.0_D3600"


class UserWalker:
    """Handles individual movement playback for one user."""

    def __init__(self, uid: int, start_coords: Coords3d, speed: float):
        self.uid = uid
        self.coords = start_coords
        self.speed = speed
        self.waypoints = []
        self.pause_timer = 0.0

    def step_move(self, dt: float) -> bool:
        """Updates location. Returns True if current path/pause is finished."""
        if self.pause_timer > 0:
            self.pause_timer -= dt
            return False

        if not self.waypoints:
            return True

        arrived, _ = self.coords.update(self.waypoints[0], dt * self.speed)

        if arrived:
            self.coords.set(self.waypoints[0])
            self.waypoints.pop(0)
            return len(self.waypoints) == 0

        return False


class ObstacleMobilityUserModel(UserModel):
    """Manages all walkers and handles the loading/saving of waypoint files."""

    def __init__(
        self,
        world_cfg,
        obstacles,
        obstacle_cfg,
        mobility_cfg: ObstacleMobilityCfg,
        rng=None,
    ):
        super().__init__()
        self.world_cfg = world_cfg
        self.obstacles = obstacles
        self.obstacle_cfg = obstacle_cfg
        self.cfg = mobility_cfg
        self.rng = rng or np.random.default_rng()

        self.walkers = []
        self.seeks = {}

        base_dir = Path(__file__).parent.resolve() / "obstacle_mobility_saved"
        self.save_dir = base_dir / self.cfg.save_name

    def reset_users(self, ue_pos: np.ndarray) -> None:
        """Initializes the model and updates the ue_pos array."""
        n_users = ue_pos.shape[0]
        boundary = self.world_cfg.env_boundary

        files_valid = self.save_dir.exists()
        if files_valid:
            for i in range(n_users):
                p = self.save_dir / f"user_{i}.npy"
                if not p.exists() or p.stat().st_size == 0:
                    files_valid = False
                    break

        if not files_valid:
            print(f"Mobility files at {self.save_dir} missing/corrupt. Generating...")
            self._generate_waypoint_files(n_users, boundary)

        self._load_walkers_from_disk(ue_pos)

    def _generate_waypoint_files(self, n_users, boundary):
        self.save_dir.mkdir(parents=True, exist_ok=True)
        graph = build_voronoi_graph(self.obstacles, boundary)
        v_ids = list(graph.vertices.keys())

        if not v_ids:
            raise ValueError("No valid map nodes (v_ids) found.")

        for uid in range(n_users):
            save_path = self.save_dir / f"user_{uid}.npy"
            elapsed = 0
            current_v_id = self.rng.choice(v_ids)

            with open(save_path, "wb") as f:
                while elapsed < self.cfg.duration:
                    target_v_id = self.rng.choice(v_ids)
                    path_coords = graph.get_shortest_path(current_v_id, target_v_id)

                    if not path_coords:
                        current_v_id = self.rng.choice(v_ids)
                        continue

                    path_array = np.array([[c.x, c.y, c.z] for c in path_coords])
                    pause = self.rng.uniform(*self.cfg.pause_range)

                    np.save(f, path_array)
                    np.save(f, np.array(pause))

                    dist = 0.0
                    if len(path_array) > 1:
                        dist = np.sum(
                            np.linalg.norm(np.diff(path_array, axis=0), axis=1)
                        )

                    elapsed += (dist / self.cfg.speed_range[1]) + pause
                    current_v_id = target_v_id

    def _load_walkers_from_disk(self, ue_pos: np.ndarray):
        """Loads data and forces the simulation state to match the saved starting points."""
        self.walkers = []
        self._locations = []
        n_users = ue_pos.shape[0]

        is_pre_randomized = not np.all(ue_pos == 0)

        for uid in range(n_users):
            f_path = self.save_dir / f"user_{uid}.npy"
            runtime_speed = self.rng.uniform(*self.cfg.speed_range)

            with open(f_path, "rb") as f:
                path_arr = np.load(f)
                _ = np.load(f)

                start_pos = Coords3d.from_array(path_arr[0])
                walker = UserWalker(uid, start_pos, runtime_speed)
                walker.waypoints = [Coords3d.from_array(p) for p in path_arr]

                self.walkers.append(walker)
                self._locations.append(start_pos)
                self.seeks[uid] = f.tell()

                if not is_pre_randomized:
                    ue_pos[uid] = path_arr[0]

    def step(self, time_step: float):
        for walker in self.walkers:
            if walker.step_move(time_step):
                self._stream_next_path(walker)

    def _stream_next_path(self, walker):
        f_path = self.save_dir / f"user_{walker.uid}.npy"
        with open(f_path, "rb") as f:
            f.seek(self.seeks[walker.uid])
            try:
                path_arr = np.load(f)
                pause_val = float(np.load(f))

                walker.waypoints = [Coords3d.from_array(p) for p in path_arr]
                walker.pause_timer = pause_val
                self.seeks[walker.uid] = f.tell()
            except (EOFError, ValueError, UnpicklingError):
                f.seek(0)
                self.seeks[walker.uid] = f.tell()
                self._stream_next_path(walker)

    def update_locations_array(self, out_array):
        for i, walker in enumerate(self.walkers):
            out_array[i, 0] = walker.coords.x
            out_array[i, 1] = walker.coords.y
            out_array[i, 2] = walker.coords.z
