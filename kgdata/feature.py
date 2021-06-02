import numpy as np


def rel_counts(data):
    return (
        data.melt(id_vars="relation", value_name="entity")
        .value_counts(["entity", "relation"])
        .unstack(fill_value=0)
    )


def rel_props(data):
    return rel_counts(data).apply(lambda row: row / row.sum(), axis=1)


def rel_dists(data):
    return (
        rel_counts(data)
        .T.apply(lambda col: col / col.sum())
        .apply(np.exp)
        .apply(lambda col: col / col.sum())
        .T
    )
