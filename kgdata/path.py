import concurrent.futures as cf
import functools as ft
import os

import networkx as nx
import pandas as pd
import tqdm.auto as tqdm


def relation_paths(data, head, tail, min_length=1, max_length=3):
    if isinstance(data, pd.DataFrame):
        graph = nx.MultiDiGraph(zip(data["head"], data["tail"], data["relation"]))
    elif isinstance(data, nx.MultiDiGraph):
        graph = data
    else:
        raise ValueError("data must be either a dataframe of a graph")

    rel_paths = []

    paths = (
        path
        for path in nx.all_simple_edge_paths(graph, head, tail, cutoff=max_length)
        if len(path) >= min_length
    )

    for path in paths:
        rel_path = [relation for _head, _tail, relation in path]

        if rel_path not in rel_paths:
            rel_paths.append(rel_path)

    return rel_paths


def all_relation_paths(data, pairs, max_workers=None, **kwargs):
    if max_workers is None and "SLURM_CPUS_PER_TASK" in os.environ:
        max_workers = os.environ["SLURM_CPUS_PER_TASK"]

    with cf.ProcessPoolExecutor(max_workers) as pool:
        function = ft.partial(_all_relation_paths_worker, data, **kwargs)
        jobs = pool.map(function, *zip(*pairs))
        paths = list(tqdm.tqdm(jobs, total=len(pairs)))

    return pd.DataFrame(
        [entry for entries in paths for entry in entries],
        columns=["ent_1", "ent_2", "path"],
    )


def _all_relation_paths_worker(data, head, tail, depth=2, stochastic=False, **kwargs):
    subdata = data.enclosing(head, tail, depth=depth, stochastic=stochastic)
    subgraph = data.graph.edge_subgraph(
        zip(subdata["head"], subdata["tail"], subdata["relation"])
    )

    return [
        (head, tail, path) for path in relation_paths(subgraph, head, tail, **kwargs)
    ]
