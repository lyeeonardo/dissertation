"""Selected final policies; all model implementations are bundled locally."""
from pathlib import Path
import torch
from work2.native.evaluation import decision_seed
from work2.native.train import load_model
from work2.native.dadp_training import load_policy
from work2.native.unified_comparison_control import UnifiedComparisonEngine, UnifiedOraclePolicy
from .controllers.upright import UprightMapPolicy
from .controllers.map import MapBeliefRepairPolicy
from .io import read

TEMPERATURE = .007242573030413831
METHODS = ('genaif', 'history', 'dadp', 'oracle', 'reward_only')


class BaselinePolicy:
    """The History/DADP branches of the original EpisodePolicy, unchanged."""
    def __init__(self, method, model, seed):
        if method not in ('history', 'dadp'):
            raise ValueError('Only the two dissertation history baselines are included')
        self.method, self.model, self.seed = method, model, seed
        self.device = next(model.parameters()).device
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.history_length = model.config.history

    @torch.no_grad()
    def act(self, history, step, oracle_condition=None):
        if oracle_condition is not None:
            raise ValueError('History baselines do not receive hidden conditions')
        seed = decision_seed(self.seed, step)
        if self.method == 'dadp':
            return self.model.act(history, seed=seed)[0].cpu().numpy(), {}
        self.generator.manual_seed(seed)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            proposal = self.model.sample(history, torch.zeros(1, dtype=torch.long, device=self.device),
                                         horizon=8, generator=self.generator, steps=8, temperature=1.0)
        return proposal['actions'][0, 0].cpu().numpy(), {}

    def observe(self, action, observation, history):
        return {}


def make_policy(method, checkpoint, sampling_seed, device='cuda', high_gravity=3):
    if method not in METHODS:
        raise ValueError('Unknown dissertation controller')
    if method in ('history', 'dadp'):
        model, state = (load_policy if method == 'dadp' else load_model)(Path(checkpoint), device=device)
        model.eval().requires_grad_(False)
        return BaselinePolicy(method, model, sampling_seed), None, state
    if high_gravity == 5:
        from .adaptation_control import load_controller_model
        model, state = load_controller_model(checkpoint, high_gravity, device=device)
    else:
        model, state = load_model(Path(checkpoint), device=device, ema=True)
    if not model.config.condition_input or model.config.history != 8 or model.config.max_horizon != 16:
        raise ValueError('The frozen shared architecture is required')
    engine = UnifiedComparisonEngine(model, batch_size=256)
    if method == 'oracle':
        return UnifiedOraclePolicy(engine, sampling_seed), engine, state
    cls, variant = (MapBeliefRepairPolicy, 'map_replay_evidence') if method == 'reward_only' else (UprightMapPolicy, 'upright_map_replay')
    policy = cls(variant, engine, sampling_seed, planning_temperature=TEMPERATURE, evidence_temperature=TEMPERATURE)
    return policy, policy.engine, state
