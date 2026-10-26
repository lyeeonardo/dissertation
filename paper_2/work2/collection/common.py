"""Portable file contracts; importing this module executes no science."""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest(path))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def verify_identity(record):
    actual = identity(record['path'])
    require(actual['sha256'] == record['sha256'], 'Input content hash differs')
    require(actual['bytes'] == record.get('bytes', record.get('size_bytes')), 'Input byte count differs')
    return actual


def verify_files(records):
    for record in records:
        verify_identity(record)


@contextmanager
def stage_lock(directory, _parent_lock_held=True):
    """A portable local exclusive lock replaces the historical supervisor."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'stage.lock').open('a+') as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def teacher_bank(path, regenerated=False):
    """A JSON mapping0..7 to policy filenames, relative to this JSON if needed."""
    path = Path(path).resolve()
    mapping = read(path)
    recipe = read(Path(__file__).with_name('recipe.json'))
    require(set(mapping) == set(map(str, range(8))), 'Exactly eight original teachers are required')
    records = []
    for condition in range(8):
        policy = Path(mapping[str(condition)])
        if not policy.is_absolute():
            policy = path.parent / policy
        actual = identity(policy)
        expected = recipe['original_teachers'][str(condition)]
        if not regenerated:
            require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')),
                    'Original teacher identity differs; use --regenerated-teachers for newly trained policies')
        else:
            receipt = read(policy.parent/'result.json')
            require(receipt.get('state') == 'completed'
                    and receipt.get('qualification', {}).get('qualified') is True
                    and receipt.get('is_profile') is False and receipt.get('steps', 0) >= 1000000,
                    'Regenerated teacher must have completed its full budget and passed qualification')
            require(receipt.get('condition') == condition and receipt.get('env_id') == 'HalfCheetah-v5'
                    and receipt.get('seed') == 13,
                    'Regenerated teacher receipt has a different condition, environment or seed')
            selected = receipt.get('selected_policy_identity', {})
            require(Path(receipt.get('selected_policy', '')).name == policy.name
                    and selected.get('sha256') == actual['sha256']
                    and selected.get('size_bytes', selected.get('bytes')) == actual['bytes'],
                    'Regenerated teacher receipt must identify the selected policy by filename, hash and size')
        records.append(dict(condition=condition, resolved_policy=actual['path'],
                            regenerated_teacher=bool(regenerated), **actual,
                            **({'qualification_receipt': identity(policy.parent/'result.json')} if regenerated else {})))
    return records


def relocate_index(root):
    """Bind legacy and relative index cells to the caller's explicit dataset root."""
    from work2.paths import corpus_rows
    return corpus_rows(root)


def bound_identity(record, path):
    """Preserve content identity while binding an old receipt to an explicit current path."""
    return verify_identity(dict(record, path=str(Path(path).resolve())))
