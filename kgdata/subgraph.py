import collections as cl
import concurrent.futures
import functools as ft
import itertools as it
import multiprocessing as mp
import os
import threading
import typing as tp

import networkx as nx
import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from . import util

rng = np.random.default_rng()


class Extractor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.index_cache = cl.defaultdict(dict)

    @util.cached_property
    def wide_data(self):
        return self.dataset.data

    @util.cached_property
    def long_data(self):
        return self.wide_data.melt(
            id_vars="relation", var_name="role", value_name="entity", ignore_index=False
        )

    @util.cached_property
    def entity_data(self):
        return self.long_data["entity"].reset_index().set_index("entity")["index"]

    @util.cached_property
    def index_data(self):
        return self.entity_data.reset_index().set_index("index")["entity"]

    @util.cached_property
    def graph(self):
        return nx.MultiDiGraph(
            zip(
                self.wide_data["head"],
                self.wide_data["tail"],
                self.wide_data["relation"],
            )
        )

    def neighbourhood(self, entity: str, depth: int = 1, cache=None) -> pd.DataFrame:
        idx = ft.reduce(
            lambda idx, _: idx.union(
                self.entity_data[self.index_data[idx].unique()].unique()
            ),
            range(depth - 1),
            pd.Index(self.entity_data[[entity]]),
        )

        return self.wide_data.loc[idx]

    def enclosing(self, head, tail, **kwargs):
        if head == tail:
            return self.neighbourhood(head)

        idx = self.neighbourhood(head, **kwargs).index.intersection(
            self.neighbourhood(tail, **kwargs).index
        )

        return self.wide_data.loc[idx]

    def all_neighbourhoods(
        self,
        max_entities: float = None,
        seed: int = None,
        max_workers: int = None,
        depth: int = 1,
        **kwargs,
    ):
        entities = self.dataset.entities

        if max_entities is not None:
            params = {
                "n" if max_entities > 1 else "frac": max_entities,
                "random_state": seed,
            }
            entities = entities.sample(**params)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            worker = ft.partial(self._all_neighbourhoods_worker, depth=depth, **kwargs)

            jobs = pool.map(
                worker,
                entities,
                chunksize=min(100, len(entities) // pool._max_workers),
            )
            neighbourhoods = list(
                tqdm.tqdm(
                    jobs,
                    total=len(entities),
                    desc=f"Extracting depth-{depth} neighbourhoods",
                    unit="entities",
                )
            )

        return pd.concat(neighbourhoods)

    def _all_neighbourhoods_worker(self, entity, **kwargs):
        neighbourhood = self.neighbourhood(entity, **kwargs)

        return pd.Series(neighbourhood.index, index=[entity] * len(neighbourhood))

    def all_neighbourhood_sizes(self, depth: int = 1, **kwargs):
        return (
            self.all_neighbourhoods(depth=depth, **kwargs)
            .groupby(level=0)
            .count()
            .to_frame("size")
            .assign(depth=depth, prop=lambda data: data["size"] / len(self.dataset))
        )

    def all_enclosing(
        self,
        depth: int = 1,
        max_pairs: float = None,
        seed: int = None,
        max_workers: int = None,
        **kwargs,
    ):
        pairs = self.dataset.unique_entity_pairs

        if max_pairs is not None:
            params = {
                "n" if max_pairs > 1 else "frac": max_pairs,
                "random_state": seed,
            }
            pairs = pairs.sample(**params)

        pairs = [tuple(pair) for pair in pairs.itertuples(index=False)]

        if max_workers is None and "SLURM_CPUS_PER_TASK" in os.environ:
            max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

        with concurrent.futures.ProcessPoolExecutor(max_workers) as pool:
            jobs = pool.map(
                ft.partial(self._all_enclosing_worker, depth=depth, **kwargs),
                *zip(*pairs),
                chunksize=min(100, len(pairs) // pool._max_workers),
            )
            subgraphs = list(
                tqdm.tqdm(
                    jobs,
                    total=len(pairs),
                    desc=f"Extracting depth-{depth} enclosing subgraphs",
                    unit="pairs",
                )
            )

        return pd.concat(subgraphs)

    def _all_enclosing_worker(self, head, tail, **kwargs):
        enclosing = self.enclosing(head, tail, **kwargs)

        return pd.Series(
            enclosing.index,
            index=pd.MultiIndex.from_tuples([(head, tail)] * len(enclosing)),
        )

        return (head, tail), list(self.enclosing(head, tail, **kwargs).index)

    def all_enclosing_sizes(self, depth: int = 1, **kwargs):
        return (
            self.all_enclosing(depth=depth, **kwargs)
            .groupby(level=[0, 1])
            .count()
            .to_frame("size")
            .assign(depth=depth, prop=lambda data: data["size"] / len(self.dataset))
        )

    def neighbourhood_sizes(self, depths, max_entities=None, seed=None, **kwargs):
        if isinstance(depths, tuple):
            min_depth, max_depth = depths
            depths = range(min_depth, max_depth + 1)

        path = self.dataset.path / "neighbourhood_sizes"

        if self.dataset.split:
            path = path / self.dataset.split

        path.mkdir(exist_ok=True, parents=True)

        path = path / f"depths_{depths}_ents_{max_entities or 'all'}_seed_{seed}.csv"

        if False and path.exists():
            return pd.read_csv(path, index_col=0)

        sizes = pd.concat(
            [
                self.all_neighbourhoods(
                    depth=depth,
                    max_entities=max_entities,
                    seed=seed,
                    **kwargs,
                )
                .map(len)
                .to_frame(name="size")
                .assign(depth=depth, prop=lambda data: data["size"] / len(self.dataset))
                for depth in depths
            ]
        )

        if False:
            sizes.to_csv(path)

        return sizes

    def enclosing_sizes(self, depths, max_pairs=None, seed=None, **kwargs):
        if isinstance(depths, tuple):
            min_depth, max_depth = depths
            depths = range(min_depth, max_depth + 1)

        path = self.dataset.path / "enclosing_sizes"

        if self.dataset.split:
            path = path / self.dataset.split

        path.mkdir(exist_ok=True, parents=True)

        path = path / f"depths_{depths}_pairs_{max_pairs or 'all'}_seed_{seed}.csv"

        if path.exists():
            return pd.read_csv(path, index_col=(0, 1))

        sizes = pd.concat(
            [
                self.all_enclosing(
                    depth=depth, max_pairs=max_pairs, seed=seed, **kwargs
                )
                .map(len)
                .to_frame(name="size")
                .assign(depth=depth, prop=lambda data: data["size"] / len(self.dataset))
                for depth in depths
            ]
        )

        sizes.to_csv(path)

        return sizes
