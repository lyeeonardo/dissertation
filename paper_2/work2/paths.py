"""Bind omitted runtime assets to explicitly provided roots, independent of original checkout names."""
from pathlib import Path
import json


def corpus_rows(root, arm='R'):
    root=Path(root).resolve()
    rows=json.loads((root/'splits'/f'{arm}_index.json').read_text())['episodes']
    resolved=[]
    for row in rows:
        stored=Path(row['path'])
        candidates=[root/stored] if not stored.is_absolute() else ([stored] if stored.is_relative_to(root) else [])
        if stored.name == f"cell_{row['condition']}.h5":
            candidates.extend([root/'data/HalfCheetah-v5'/stored.name,root/stored.name])
        found={p.resolve() for p in candidates if p.is_file()}
        if len(found)!=1:
            raise FileNotFoundError(f'Expected exactly one declared cell file under {root} for {stored}; got {sorted(map(str,found))}')
        resolved.append(dict(row,path=str(found.pop())))
    return resolved


def runtime_manifest(path):
    """Relative file references in supplied fitting JSON resolve beside that manifest."""
    path=Path(path).resolve()
    def visit(value,key=None):
        if isinstance(value,dict): return {k:visit(v,k) for k,v in value.items()}
        if isinstance(value,list): return [visit(v,key) for v in value]
        if key in ('path','checkpoint') and isinstance(value,str):
            p=Path(value)
            return str(p if p.is_absolute() else path.parent/p)
        return value
    return visit(json.loads(path.read_text()))
