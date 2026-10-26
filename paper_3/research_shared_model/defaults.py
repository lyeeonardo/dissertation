"""User-selected research default: original GenAIF, VFE, refinement and L1.

Historical source and model files remain unchanged. Ablations must be named
explicitly; an exploratory winner does not silently replace this default.
"""

def controller_recipe():
    """Fresh recipe copy; values match the preserved default experiment."""
    return dict(mode='genaif', history_length=1, passes=10, candidates=3,
        particles=8, belief_update=True, refine=True,
        information_weight=.215947119414, tau=.3243581893605516,
        seed=17, horizon=30, discount=.99)

from work3_paths import input_path
CHECKPOINT = input_path("models/joint_diffusion.pt")
