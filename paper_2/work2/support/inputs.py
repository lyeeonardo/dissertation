"""Resolve frozen transition identities against a separately supplied corpus."""
from collections import Counter, defaultdict
from pathlib import Path
from work2.io import read

CONFIG = Path(__file__).resolve().parents[1] / 'configs'


def references(kind, split, root):
    root = Path(root).resolve()
    refs = read(CONFIG / 'support_references.json')[kind][split]
    rows = read(root / 'splits/R_index.json')['episodes']
    ancestry = defaultdict(set)
    for row in rows: ancestry[str(row['parent_id'])].add(row['split'])
    if any(len(splits) != 1 for splits in ancestry.values()):
        raise ValueError('Parent ancestry crosses TRAIN/CAL/DEV/TEST')
    expected = {'calibration': {'expert': 16, 'recovery': 8, 'paired': 8},
                'development': {'expert': 8, 'recovery': 4, 'paired': 4}}[split]
    if Counter((r['condition_evaluator_only'], r['kind']) for r in refs) != Counter({(c,k):n for c in range(8) for k,n in expected.items()}):
        raise ValueError('Frozen condition/kind quota differs')
    if len({r['transition_id'] for r in refs}) != len(refs):
        raise ValueError('Duplicate frozen transitions')
    resolved = []
    for ref in refs:
        row = rows[ref['row_index']]
        # Old receipts embed their original checkout path; bind its corpus-relative suffix.
        row_path = str(row['path']).replace('\\', '/')
        relative = 'data/' + row_path.split('/data/', 1)[1] if '/data/' in row_path else row_path
        if (row['split'] != split or ref['role'] != split or relative != ref['path']
            or any(ref[k] != row[k] for k in ('episode','length','kind','parent_id'))
            or ref['condition_evaluator_only'] != row['condition']
            or not 0 <= ref['step'] < row['length']):
            raise ValueError('Frozen transition no longer resolves to its original source')
        path = root / ref['path']
        if not path.is_file(): raise FileNotFoundError(path)
        resolved.append(dict(ref, path=str(path), index_path=str(root / ref['index_path'])))
    return resolved
