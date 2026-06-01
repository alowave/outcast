__all__ = [
    "LoadController",
    "RandomMovementCfg",
    "RandomMovementUserModel",
    "UserModel",
]


def __getattr__(name: str):
    if name == "UserModel":
        from src.outcast.world.user_model.base import UserModel

        return UserModel
    if name == "LoadController":
        from src.outcast.world.user_model.load_ctrl import LoadController

        return LoadController
    if name == "RandomMovementCfg":
        from src.outcast.world.user_model.random_movement import RandomMovementCfg

        return RandomMovementCfg
    if name == "RandomMovementUserModel":
        from src.outcast.world.user_model.random_movement import (
            RandomMovementUserModel,
        )

        return RandomMovementUserModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
