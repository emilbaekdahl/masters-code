import ast
import concurrent.futures as cf
import functools as ft
import os
import pathlib as pl

import networkx as nx
import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from . import util

rng = np.random.default_rng()


class Extractor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache_path = pl.Path(dataset.path / "neighbourhoods")
        self.cache = {}
        self.stochastic_cache = {}

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

    def cache_lookup(self, entity: str, depth: int, stochastic: bool):
        cache = self.stochastic_cache if stochastic else self.cache

        if depth not in cache:
            path = self.cache_path

            if stochastic:
                path = path / "stochastic"

            path = path / f"{depth}.csv"

            if path.exists():
                cache[depth] = pd.read_csv(
                    path,
                    index_col=0,
                    converters={"index": ast.literal_eval},
                )
            else:
                return None

        try:
            return cache[depth]["index"].loc[entity]
        except KeyError:
            return None

    @ft.lru_cache(maxsize=100_000)
    def stochastic_neighbourhood(self, entity, depth=1) -> pd.DataFrame:
        cached = self.cache_lookup(entity, depth, True)

        if cached:
            idx = pd.Index(cached)
        else:
            idx = ft.reduce(
                self._stochastic_neighbourhood_reducer,
                range(depth - 1),
                pd.Index(self.entity_data[[entity]]),
            ).unique()

        return self.wide_data.loc[idx]

    def _stochastic_neighbourhood_reducer(self, idx, _):
        triples_per_entity = len(self.dataset) // len(self.dataset.entities)

        entities = self.index_data[idx]

        if len(entities) > triples_per_entity:
            entities = entities.sample(triples_per_entity)

        return idx.union(self.entity_data[entities.unique()].unique())

    @ft.lru_cache(maxsize=100_000)
    def neighbourhood(self, entity: str, depth: int = 1) -> pd.DataFrame:
        cached = self.cache_lookup(entity, depth, False)

        if cached:
            idx = pd.Index(cached)
        else:
            idx = ft.reduce(
                lambda idx, _: idx.union(
                    self.entity_data[self.index_data[idx].unique()].unique()
                ),
                range(depth - 1),
                pd.Index(self.entity_data[[entity]]),
            ).unique()

        return self.wide_data.loc[idx]

    def enclosing(self, head, tail, stochastic=False, **kwargs):
        function = self.stochastic_neighbourhood if stochastic else self.neighbourhood

        if head == tail:
            return function(head)

        idx = function(head, **kwargs).index.intersection(
            function(tail, **kwargs).index
        )

        return self.wide_data.loc[idx]

    def all_neighbourhoods(
        self,
        max_entities: float = None,
        seed: int = None,
        max_workers: int = None,
        depth: int = 1,
        chunk_size: int = None,
        **kwargs,
    ):
        entities = self.dataset.entities

        if max_entities is not None:
            params = {
                "n" if max_entities > 1 else "frac": max_entities,
                "random_state": seed,
            }
            entities = entities.sample(**params)

        if max_workers is None and "SLURM_CPUS_PER_TASK" in os.environ:
            max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

        with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
            print(f"Using {pool._max_workers} workers")

            worker = ft.partial(self._all_neighbourhoods_worker, depth=depth, **kwargs)

            if chunk_size is None:
                chunk_size = max(1, len(entities) // pool._max_workers)

            jobs = pool.map(worker, entities, chunksize=chunk_size)
            neighbourhoods = list(
                tqdm.tqdm(
                    jobs,
                    total=len(entities),
                    desc=f"Extracting depth-{depth} neighbourhoods",
                    unit="entities",
                )
            )

        return pd.concat(neighbourhoods)

    def _all_neighbourhoods_worker(self, entity, stochastic: bool = False, **kwargs):
        if stochastic:
            neighbourhood = self.stochastic_neighbourhood(entity, **kwargs)
        else:
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
        chunk_size: int = None,
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

        with cf.ProcessPoolExecutor(max_workers) as pool:
            print(f"Using {pool._max_workers} workers")
            worker = ft.partial(self._all_enclosing_worker, depth=depth, **kwargs)

            if chunk_size is None:
                chunk_size = max(1, len(pairs) // pool._max_workers)

            jobs = pool.map(worker, *zip(*pairs), chunksize=chunk_size)
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
            index=pd.MultiIndex.from_tuples(
                [(head, tail)] * len(enclosing), names=("ent_1", "ent_2")
            ),
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
