"""Portable input/output and content identities."""
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=digest)


def verify(records):
    for expected in records:
        actual = identity(expected['path'])
        if actual['sha256'] != expected['sha256'] or actual['size_bytes'] != expected.get('size_bytes', expected.get('bytes')):
            raise ValueError('Input content differs: ' + expected['path'])


def bind_checkpoint(path, role, regenerated=False):
    """Bind a support study to its exact historical weights, or label a new fit explicitly."""
    specification=read(Path(__file__).with_name('required_assets.json'))
    expected=specification['support_checkpoint'] if role=='earlier' else specification['checkpoints']['shared']
    actual=identity(path)
    if not regenerated and actual['sha256']!=expected['sha256']:
        raise ValueError('Input is not the archived '+role+' checkpoint; use --regenerated-checkpoints for a new fit')
    return dict(actual,regenerated=bool(regenerated),archived_sha256=expected['sha256'])
