import concurrent.futures as cf
import functools as ft
import os

import networkx as nx
import pandas as pd
import tqdm.auto as tqdm


def nx_rel_seqs(dataset, head, tail, max_paths=None, min_length=1, max_length=3):
    seqs = []

    all_paths = (
        path
        for path in nx.all_simple_edge_paths(
            dataset.graph, head, tail, cutoff=max_length
        )
        if len(path) >= min_length
    )

    for path in all_paths:
        seq = [relation for _head, _tail, relation in path]
        seq = dataset.rel_seq_to_idx(seq)

        if not seq in seqs:
            seqs.append(seq)

            if max_paths and len(seqs) >= max_paths:
                break

    return seqs


def all_nx_rel_seqs(dataset, max_pairs=None, depth=3, **kwargs):
    pairs = dataset.unique_entity_pairs

    if max_pairs:
        pairs = pairs.sample(frac=max_pairs)

    all_rel_seqs = [
        (
            (head, tail),
            nx_rel_seqs(dataset, head, tail, max_length=depth, **kwargs),
        )
        for head, tail in tqdm.tqdm(
            pairs.itertuples(index=False),
            total=len(pairs),
            desc=dataset.__class__.__name__,
        )
    ]
    all_rel_seqs = [
        (index, rel_seqs) for index, rel_seqs in all_rel_seqs if len(rel_seqs) > 0
    ]

    return pd.concat(
        [
            pd.Series(
                rel_seqs,
                index=pd.MultiIndex.from_tuples(
                    [(head, tail)] * len(rel_seqs), names=["ent_1", "ent_2"]
                ),
                name="rel_seq",
            )
            for (head, tail), rel_seqs in all_rel_seqs
        ]
    )


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


def all_relation_paths(data, pairs, max_workers=None, chunk_size=10, **kwargs):
    if max_workers is None and "SLURM_CPUS_PER_TASK" in os.environ:
        max_workers = os.environ["SLURM_CPUS_PER_TASK"]

    with cf.ProcessPoolExecutor(max_workers) as pool:
        function = ft.partial(_all_relation_paths_worker, data, **kwargs)
        jobs = pool.map(function, *zip(*pairs), chunksize=chunk_size)
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
