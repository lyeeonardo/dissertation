"""Explicit reproduction inputs; no private project or source-tree lookup."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent
INPUTS = Path(os.environ.get('WORK3_INPUTS', ROOT / 'inputs')).resolve()
def input_path(relative):
    return INPUTS / relative


def verify_relocated_training_data(recorded_path, supplied_path):
    """Accept a moved frozen dataset by content, never by a private path lookup."""
    supplied_path = Path(supplied_path).resolve()
    if Path(recorded_path).resolve() == supplied_path:
        return
    import hashlib
    import json
    expected = json.loads((ROOT / 'configs/inputs.json').read_text())
    for item in expected:
        if not item['path'].startswith('training/'):
            continue
        path = supplied_path / Path(item['path']).name
        if not path.is_file() or hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() != item['sha256']:
            raise ValueError(f'Relocated frozen dataset content differs: {path}')
