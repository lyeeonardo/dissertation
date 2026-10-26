"""Isolated fresh shared/History fits; metadata only unless --execute is supplied.

The original native loss, sampler, AdamW update and EMA are imported unchanged.
H4 describes intended control evaluation, not a restriction of training horizons.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import fcntl
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
from .plateau import Plateau

SEED = 13
DEV_SEED = 130_003
BATCH = 1024
MICROBATCH = 512
DEV_BATCH = 256
DEV_BATCHES = 8
CHECK_EVERY = 2000
SAVE_EVERY = 10_000
MAX_UPDATES = 300_000
SCHEMA = 'gravity3x_native_diffusion_v1'


def file_record(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def inspect_data(root):
    """Read metadata and check split isolation without importing Torch or CUDA."""
    root = Path(root).resolve()
    index_path = root/'splits/R_index.json'
    normalization_path = root/'normalization.json'
    summary_path = root/'data_summary.json'
    collection_path = root/'data_collection_result.json'
    summary = json.loads(summary_path.read_text())
    collection = json.loads(collection_path.read_text())
    if (summary.get('state') != 'complete' or summary.get('gravity_levels') != [1.,3.]
            or collection.get('state') != 'completed'):
        raise ValueError('Require completed collection and the verified [1,3] gravity corpus')
    if (not summary.get('train_dev_cal_test_parent_disjoint')
            or not summary.get('paired_same_roots_actions_prefixes_and_split')
            or summary.get('old_high_gravity_transitions_relabelled') != 0):
        raise ValueError('Data ancestry/branch/law qualification missing or failed')
    expected_splits = dict(train=560000, development=80000, calibration=80000, test=80000)
    arm = summary['arms']['R']
    if (arm['transitions'] != 800000 or arm['splits'] != expected_splits
            or arm['conditions'] != {str(c):100000 for c in range(8)}
            or collection.get('new_dataset_transitions') != 432000
            or collection.get('new_collection_physical_transitions') != 436000):
        raise ValueError('Completed corpus/physical counts differ from the declared R recipe')
    from work2.paths import corpus_rows
    rows = corpus_rows(root)
    normalization = json.loads(normalization_path.read_text())
    if not rows:
        raise ValueError('Empty R index')
    parents = defaultdict(set)
    counts = defaultdict(Counter)
    episodes = set()
    for row in rows:
        identity = (row['path'], row['episode'])
        if identity in episodes:
            raise ValueError(f'Duplicate indexed episode: {identity}')
        episodes.add(identity)
        if row['split'] not in ('train', 'development', 'calibration', 'test'):
            raise ValueError(f'Unexpected split: {row["split"]}')
        condition = int(row['condition'])
        if condition not in range(8) or int(row['length']) <= 0:
            raise ValueError('Invalid indexed condition/length')
        parents[str(row['parent_id'])].add(row['split'])
        counts[row['split']][condition] += int(row['length'])
    if any(len(splits) != 1 for splits in parents.values()):
        raise ValueError('Parent leakage across TRAIN/DEV/CAL/TEST')
    for split in ('train', 'development'):
        if set(counts[split]) != set(range(8)):
            raise ValueError(f'{split} must contain every condition')
    for key in ('observation_mean', 'observation_std', 'delta_mean', 'delta_std',
                'reward_mean', 'reward_std'):
        value = normalization[key]
        values = value if isinstance(value, list) else [value]
        expected = 1 if key.startswith('reward') else 17
        if len(values) != expected or not all(math.isfinite(float(x)) for x in values):
            raise ValueError(f'Invalid normalization shape/value: {key}')
        if key.endswith('std') and not all(float(x) > 0 for x in values):
            raise ValueError(f'Nonpositive normalization: {key}')
    if {split:sum(values.values()) for split,values in counts.items()} != expected_splits:
        raise ValueError('Index transition counts disagree with the completed data summary')
    records = [file_record(index_path), file_record(normalization_path),
               file_record(summary_path), file_record(collection_path)]
    records += [file_record(p) for p in sorted({row['path'] for row in rows})]
    return dict(root=str(root), files=records, normalization=normalization,
                episode_count=len(rows), parent_count=len(parents),
                parent_split_isolation=True,
                transition_counts={s: dict(c) for s, c in sorted(counts.items())},
                split_parent_counts=dict(Counter(next(iter(s)) for s in parents.values())))


def recipe(method, data):
    return dict(schema=SCHEMA, method=method, arm='R', seed=SEED,
                fresh_initialization=True, dataset=data, effective_batch=BATCH,
                microbatch=MICROBATCH, history=8, max_horizon=16,
                roles=dict(proposal=0.5, one_step_prediction=0.25, trajectory_prediction=0.25),
                trajectory_and_proposal_horizons=[4, 8, 16], prediction_horizon=1,
                condition_input=method == 'shared',
                optimizer=dict(name='AdamW', weight_decay=0.01, fused=True),
                learning_rate='Native 10k warm-up to 1e-4, cosine to 1e-5 at 300k',
                precision='Native BF16 forward with FP32 losses/parameters; TF32 enabled',
                gradient_clip=1.0, ema_rate=0.999, compilation='default_forward',
                validation=dict(split='development', seed=DEV_SEED, batch_size=DEV_BATCH,
                                batches=DEV_BATCHES, every_updates=CHECK_EVERY,
                                sampled_windows=DEV_BATCH*DEV_BATCHES, augment=False,
                                monitor='EMA total native diffusion loss',
                                fixed_examples_and_corruption_draws=True,
                                sampled_with_replacement=True),
                plateau=Plateau().state_dict(), maximum_updates=MAX_UPDATES,
                periodic_checkpoint_every=SAVE_EVERY,
                selection='Lowest raw EMA DEV loss; separate from patience threshold',
                execution='One worker; compiler-only eager fallback before optimizer; OOM/fatal stops',
                control_evaluation_horizon=4, control_runs=0)


def atomic_link(source, destination):
    """Give immutable checkpoint bytes another name without duplicating storage."""
    source, destination = Path(source), Path(destination)
    temporary = destination.with_name(destination.name + f'.{os.getpid()}.tmp')
    if temporary.exists():
        temporary.unlink()
    os.link(source, temporary)
    temporary.replace(destination)


@contextmanager
def training_lock(root, method):
    """Shared with the DADP stages and data collector; a dead process releases it."""
    with (root/'training.lock').open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another study worker owns training.lock') from error
        handle.seek(0)
        handle.truncate()
        json.dump(dict(pid=os.getpid(), method=method, started_at=time.time()), handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_development_bank(sampler):
    """Materialize the native deterministic examples, retaining draw identities."""
    import numpy as np
    sampler.rng = np.random.default_rng(DEV_SEED)
    batches, draws = [], []
    native_history_at = sampler.history_at

    def capture_history(anchors):
        draws.append(anchors.copy())
        return native_history_at(anchors)

    sampler.history_at = capture_history
    try:
        for _ in range(DEV_BATCHES):
            batches.append(sampler.sample(DEV_BATCH, augment=False))
    finally:
        sampler.history_at = native_history_at
    if len(draws) != DEV_BATCHES:
        raise AssertionError('Unexpected native sampler history calls')
    ends = np.cumsum([sampler.history - 1 + int(r['length']) + 1 + sampler.horizon
                      for r in sampler.rows])
    drawn_parents = [[sampler.rows[i]['parent_id'] for i in np.searchsorted(ends, a, side='right')]
                    for a in draws]
    return dict(schema=SCHEMA, seed=DEV_SEED, batches=batches,
                anchors_before_native_permutation=draws,
                parents_before_native_permutation=drawn_parents,
                sample_count=DEV_BATCH*DEV_BATCHES,
                distinct_anchor_count=len(np.unique(np.concatenate(draws))),
                distinct_parent_count=len(set(str(p) for batch in drawn_parents for p in batch)),
                source_rows=sampler.rows)


def evaluate_bank(ema, bank, device='cuda'):
    import torch
    from work2.native.windows import to_device
    generator = torch.Generator(device=device).manual_seed(DEV_SEED)
    metrics = {}
    ema.eval()
    with torch.no_grad():
        for batch in bank['batches']:
            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
                loss, values = ema.loss(to_device(batch, device), generator)
            for key, value in dict(loss=loss, **values).items():
                metrics[key] = metrics.get(key, 0.) + float(value) / len(bank['batches'])
    if not all(math.isfinite(v) for v in metrics.values()):
        raise FloatingPointError('Nonfinite held-out validation metric')
    return metrics


def verify_checkpoint(state, expected, config):
    if state.get('schema') != SCHEMA or state.get('recipe') != expected:
        raise ValueError('Resume recipe/data metadata differs')
    if state['model_config'] != config or state['normalization'] != expected['dataset']['normalization']:
        raise ValueError('Resume model or normalization differs')
    if state['arm'] != 'R' or state['seed'] != SEED or state['effective_batch'] != BATCH:
        raise ValueError('Resume arm/seed/effective batch differs')
    completed = int(state['updates'])
    if not 0 <= completed <= MAX_UPDATES:
        raise ValueError('Invalid completed update count')
    steps = {int(s['step'].item()) for s in state['optimizer']['state'].values() if 'step' in s}
    if completed and steps != {completed}:
        raise ValueError(f'Optimizer counters differ from completed updates: {steps}')
    if completed and state['optimizer']['param_groups'][0]['lr'] != native_lr(completed-1):
        raise ValueError('Saved optimizer learning rate differs from native schedule')
    plateau = Plateau.from_state(state['plateau'])
    for name in ('minimum_updates', 'patience', 'relative_improvement'):
        if getattr(plateau, name) != expected['plateau'][name]:
            raise ValueError(f'Resume plateau rule differs: {name}')
    if plateau.last_update > completed:
        raise ValueError('Plateau counter exceeds trained updates')
    if state['development'] and state['development'][-1]['updates'] != plateau.last_update:
        raise ValueError('Validation history and plateau state disagree')
    if completed and not state['rng']['cuda']:
        raise ValueError('CUDA RNG state missing')


def native_lr(update):
    # Kept import-free for metadata checks; compared with the native API in tests.
    if update < 10_000:
        return 1e-4 * (update+1) / 10_000
    fraction = min(1.0, (update-10_000) / 290_000)
    return 1e-5 + 0.5*(1e-4-1e-5)*(1+math.cos(math.pi*fraction))


def optimization_step(model, ema, optimizer, batch, eager_forward, compiled):
    """Native loss/update; a compilation failure can only retry before the step."""
    import torch
    from work2.native.common import restore_rng, rng_state
    from work2.native.train import update_ema
    from work2.native.windows import slice_batch
    random_before = rng_state()
    adjustment = None
    while True:
        optimizer.zero_grad(set_to_none=True)
        model.train()
        metrics = {}
        try:
            n = len(batch['role'])
            for start in range(0, n, MICROBATCH):
                end = min(n, start+MICROBATCH)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss, values = model.loss(slice_batch(batch, start, end))
                fraction = (end-start)/n
                (loss*fraction).backward()
                for key, value in dict(loss=loss.detach(), **values).items():
                    metrics[key] = metrics.get(key, 0) + value*fraction
        except (torch._dynamo.exc.BackendCompilerFailed, torch._dynamo.exc.Unsupported) as error:
            if not compiled:
                raise
            optimizer.zero_grad(set_to_none=True)
            model.forward = eager_forward
            compiled = False
            restore_rng(random_before)
            adjustment = dict(reason='compiler_eager_fallback', error=repr(error),
                              same_batch_and_corruption_rng=True, before_optimizer=True)
            continue
        break
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    update_ema(ema, model)
    metrics['gradient_norm'] = norm.detach()
    return metrics, compiled, adjustment


def execute(root, method, expected, resume=None):
    import numpy as np
    import torch
    from work2.native.common import (
        append_json, atomic_json, atomic_torch_save, require_cuda, restore_rng, rng_state, seed_all)
    from work2.native.model import ModelConfig, SharedDiffusion
    from work2.native.train import learning_rate
    from work2.native.windows import WindowSampler, to_device

    directory = root/'models'/f'{method}_R'
    directory.mkdir(parents=True, exist_ok=True)
    if resume is None and any(directory.iterdir()):
        raise FileExistsError('Existing fit requires explicit --resume; no silent restart')
    if resume is not None and (directory/'result.json').exists():
        raise ValueError('Terminal fit is complete; no automatic extension')
    attempt = f'{time.time_ns()}_{os.getpid()}'
    atomic_json(directory/f'attempt_{attempt}.json', dict(state='started', pid=os.getpid(),
                resume=str(resume) if resume else None, started_at=time.time(), recipe=expected))
    completed, initial_update = 0, 0
    checkpoint_path = None
    started = time.monotonic()
    old_handlers = {}
    stop_requested = []
    try:
        seed_all(SEED)
        gpu = require_cuda()  # Startup computation only; no monitoring probe loop.
        sampler = WindowSampler(root, 'R', 'train', seed=SEED, history=8, horizon=16)
        development = WindowSampler(root, 'R', 'development', seed=DEV_SEED, history=8, horizon=16)
        bank = build_development_bank(development)
        del development
        config = ModelConfig(condition_input=method == 'shared')
        model = SharedDiffusion(config, sampler.normalization).cuda()
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        eager_forward = model.forward
        compiled = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01, fused=True)
        plateau = Plateau()
        dev_metrics = []
        total_elapsed = 0.
        aggregate, recorded = {}, 0
        bank_path = directory/'development_bank.pt'
        if resume is not None:
            checkpoint_path = (directory/'latest.pt') if resume == 'latest' else Path(resume).resolve()
            if not checkpoint_path.is_relative_to(directory):
                raise ValueError('Resume checkpoint must belong to this isolated method')
            state = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
            verify_checkpoint(state, expected, asdict(config))
            if checkpoint_path != directory/'latest.pt':
                latest = torch.load(directory/'latest.pt', map_location='cpu', weights_only=False)
                if latest['updates'] != state['updates']:
                    raise ValueError('Resume must use the last committed update, not an older checkpoint')
                del latest
            model.load_state_dict(state['model'])
            ema.load_state_dict(state['ema'])
            optimizer.load_state_dict(state['optimizer'])
            completed = initial_update = state['updates']
            plateau = Plateau.from_state(state['plateau'])
            dev_metrics = state['development']
            total_elapsed = state['elapsed_seconds']
            compiled = state['forward_compiled']
            aggregate = {k: torch.tensor(v, device='cuda') for k, v in state['aggregate'].items()}
            recorded = state['recorded']
            saved_bank = torch.load(bank_path, map_location='cpu', weights_only=False)
            for old, new in zip(saved_bank['anchors_before_native_permutation'],
                                bank['anchors_before_native_permutation'], strict=True):
                np.testing.assert_array_equal(old, new)
            if saved_bank['source_rows'] != bank['source_rows']:
                raise ValueError('Fixed DEV source rows differ')
            bank = saved_bank
            sampler.rng.bit_generator.state = state['sampler_rng']
            random_state = state['rng']
            random_state['torch'] = random_state['torch'].cpu()
            random_state['cuda'] = [x.cpu() for x in random_state['cuda']]
            restore_rng(random_state)
            if plateau.best_update == completed:
                atomic_link(checkpoint_path, directory/'best.pt')
            elif plateau.best_update is not None:
                best = torch.load(directory/'best.pt', map_location='cpu', weights_only=False)
                if best['updates'] != plateau.best_update:
                    raise ValueError('Saved best checkpoint differs from resumed stopping state')
                del best
            # The committed checkpoint owns the resumed validation prefix. A
            # killed checkpoint save may have left one uncommitted JSONL row.
            log_path = directory/'development.jsonl'
            log_temp = log_path.with_name(log_path.name+f'.{os.getpid()}.tmp')
            log_temp.write_text(''.join(json.dumps(row, allow_nan=False)+'\n' for row in dev_metrics))
            log_temp.replace(log_path)
            del state
        else:
            atomic_torch_save(bank_path, bank)
            atomic_json(directory/'config.json', expected)
            atomic_json(directory/'development_bank.json', {k:v for k,v in bank.items()
                        if k not in ('batches', 'anchors_before_native_permutation',
                                     'parents_before_native_permutation', 'source_rows')})
        if compiled:
            os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', str(root/'.cache/inductor'))
            os.environ.setdefault('TRITON_CACHE_DIR', str(root/'.cache/triton'))
            model.forward = torch.compile(eager_forward, mode='default', fullgraph=True)

        def save(is_best=False, final=False):
            nonlocal checkpoint_path
            # This is called only between complete optimizer updates.
            checkpoint_path = directory/'latest.pt'
            state = dict(schema=SCHEMA, recipe=expected, model=model.state_dict(),
                         ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                         model_config=asdict(config), normalization=sampler.normalization,
                         arm='R', method=method, seed=SEED, updates=completed,
                         effective_batch=BATCH, microbatch=MICROBATCH,
                         rng=rng_state(), sampler_rng=sampler.rng.bit_generator.state,
                         development=dev_metrics, plateau=plateau.state_dict(),
                         elapsed_seconds=total_elapsed+time.monotonic()-started,
                         aggregate={k:float(v) for k,v in aggregate.items()}, recorded=recorded,
                         forward_compiled=compiled, attempt=attempt)
            atomic_torch_save(checkpoint_path, state)
            if is_best:
                atomic_link(checkpoint_path, directory/'best.pt')
            if completed and completed % SAVE_EVERY == 0:
                periodic = directory/f'checkpoint_{completed:07d}.pt'
                if not periodic.exists():
                    atomic_link(checkpoint_path, periodic)
            if final:
                atomic_link(checkpoint_path, directory/'final.pt')
            atomic_json(directory/'resume.json', dict(checkpoint=str(checkpoint_path), updates=completed,
                        attempt=attempt, plateau=plateau.state_dict()))
            if plateau.best_update is not None:
                atomic_json(directory/'selected.json', dict(state='selected', method=method, arm='R',
                            checkpoint=str(directory/'best.pt'), selected_checkpoint=str(directory/'best.pt'),
                            updates=plateau.best_update, selected_update=plateau.best_update,
                            ema=True, criterion='Minimum raw fixed EMA DEV total loss',
                            development_loss=plateau.best_value, seed=SEED,
                            selection_final=bool(final), control_evaluation_completed=False))

        def request_stop(signum, frame):
            stop_requested.append(signum)

        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, request_stop)
        while completed < MAX_UPDATES and not stop_requested and plateau.bad_checks < plateau.patience:
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(completed)
            batch = to_device(sampler.sample(BATCH))
            metrics, compiled, adjustment = optimization_step(
                model, ema, optimizer, batch, eager_forward, compiled)
            if adjustment:
                append_json(directory/'adjustments.jsonl', dict(attempt=attempt, updates=completed,
                            **adjustment))
            completed += 1
            for key, value in metrics.items():
                aggregate[key] = aggregate.get(key, 0) + value
            recorded += 1
            if completed % CHECK_EVERY == 0:
                validation = evaluate_bank(ema, bank)
                decision = plateau.observe(completed, validation['loss'])
                dev = dict(updates=completed, **validation, **decision)
                dev_metrics.append(dev)
                append_json(directory/'development.jsonl', dict(attempt=attempt, **dev))
                report = dict(state='training', attempt=attempt, pid=os.getpid(),
                              updates=completed, maximum_updates=MAX_UPDATES,
                              effective_batch=BATCH, microbatch=MICROBATCH,
                              train_metrics={k:float(v/recorded) for k,v in aggregate.items()},
                              latest_development=dev, best_update=plateau.best_update,
                              elapsed_seconds=total_elapsed+time.monotonic()-started,
                              updates_per_second=(completed-initial_update)/(time.monotonic()-started),
                              gpu_name=gpu['name'])
                if not all(math.isfinite(v) for v in report['train_metrics'].values()):
                    raise FloatingPointError('Nonfinite aggregate training metric')
                aggregate, recorded = {}, 0
                save(is_best=decision['is_best'])
                atomic_json(directory/'progress.json', report)
                append_json(directory/'progress.jsonl', report)
                print(json.dumps(report), flush=True)
        if stop_requested:
            save()
            atomic_json(directory/'interruption.json', dict(state='interrupted', updates=completed,
                        checkpoint=str(checkpoint_path), signals=stop_requested,
                        exact_continuation='Requires explicit --resume latest', automatic_retry=False))
            return 130
        reason = 'validation_plateau' if plateau.bad_checks >= plateau.patience else 'budget_cap'
        # Last scheduled validation already committed the latest checkpoint. Link it
        # directly so periodic paths remain immutable and no duplicate optimizer save occurs.
        if completed != plateau.last_update:
            raise AssertionError('Terminal endpoint lacks scheduled validation')
        atomic_link(directory/'latest.pt', directory/'final.pt')
        selected = json.loads((directory/'selected.json').read_text())
        selected['selection_final'] = True
        atomic_json(directory/'selected.json', selected)
        result = dict(state='completed', stop_reason=reason, updates=completed,
                      best_update=plateau.best_update, best_development_loss=plateau.best_value,
                      selected_update=plateau.best_update, selected_checkpoint=str(directory/'best.pt'),
                      final_development=dev_metrics[-1], plateau=plateau.state_dict(),
                      best_checkpoint=str(directory/'best.pt'), final_checkpoint=str(directory/'final.pt'),
                      selection_manifest=str(directory/'selected.json'),
                      effective_batch=BATCH, sampled_training_windows=completed*BATCH,
                      actual_physical_transitions=0, seed=SEED, method=method,
                      elapsed_seconds=total_elapsed+time.monotonic()-started,
                      no_control_performance_claim=True)
        atomic_json(directory/'result.json', result)
        atomic_json(directory/'progress.json', result)
        print(json.dumps(result), flush=True)
        return 0
    except BaseException as error:
        atomic_json(directory/f'failure_{attempt}.json', dict(state='failed', attempt=attempt,
                    last_completed_update=completed, error=repr(error), traceback=traceback.format_exc(),
                    last_committed_checkpoint=str(checkpoint_path) if checkpoint_path else None,
                    automatic_retry=False, partially_executed_update_not_checkpointed=True))
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=HERE)
    parser.add_argument('--method', choices=('shared', 'history'), required=True)
    parser.add_argument('--resume', help='Explicit latest or a checkpoint path from this method')
    parser.add_argument('--execute', action='store_true', help='Authorize the declared fit')
    args = parser.parse_args()
    data = inspect_data(args.root)
    expected = recipe(args.method, data)
    if not args.execute:
        print(json.dumps(dict(state='metadata_preflight_passed', executes_training=False,
                              recipe=expected), indent=2))
        return 0
    with training_lock(args.root.resolve(), args.method):
        return execute(args.root.resolve(), args.method, expected, args.resume)


if __name__ == '__main__':
    raise SystemExit(main())
