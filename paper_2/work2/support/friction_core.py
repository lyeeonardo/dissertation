"""Original exact root/prefix/action matching for the fixed friction probe."""
import numpy as np
PREFIXES=("prefix_observations","prefix_actions","prefix_rewards","prefix_mask")

def equal(a,b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a,b)


def branch(group):
    if not bool(group.attrs['complete']) or int(group.attrs['length']) != 8:
        raise ValueError('Expected completed native8-step paired branch')
    return dict(integration=group['integration'][0], root=group['observations'][0],
                action=group['actions'][0], next_observation=group['observations'][1],
                reward=float(group['rewards'][0]),
                **{name:group[name][:] for name in PREFIXES},
                metadata={key:group.attrs[key] for key in
                          ('anchor_id','chunk_id','source_condition','parent_id','split','initial_elapsed')})


def identify(old,new):
    """All eight hypotheses must share the complete root, prefix and first action."""
    fields=('integration','root','action',*PREFIXES)
    cross={key:all(equal(a[key],b[key]) for a,b in zip(old,new)) for key in fields}
    within={key:all(equal(rows[0][key],r[key]) for rows in (old,new) for r in rows[1:])
            for key in fields}
    metadata=all(a['metadata']==b['metadata'] for a,b in zip(old,new))
    complete=metadata and all(cross.values()) and all(within.values())
    return complete,dict(metadata_equal=metadata,
                         **{'cross_'+k:v for k,v in cross.items()},
                         **{'within_'+k:v for k,v in within.items()})
