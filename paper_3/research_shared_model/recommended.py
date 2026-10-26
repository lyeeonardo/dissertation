"""Final public55 controller with fixed two-coordinate mean correction."""
from . import defaults
def make_controller(device="cuda"):
    from bsm1_offline_benchmark.stock_joint_control import JointController
    from .efe_exploration.selective_mean_control import SelectiveMeanController
    return SelectiveMeanController(JointController(defaults.CHECKPOINT, device=device, remaining=1344, **defaults.controller_recipe()), device)
