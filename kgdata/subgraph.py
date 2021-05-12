import collections as cl
import concurrent.futures
import functools as ft
import threading

import numpy as np
import pandas as pd
import tqdm.autonotebook as tqdm

from . import util

rng = np.random.default_rng()


class Extractor:
    def __init__(self, dataset, use_cache=True):
        self.dataset = dataset
        self.use_cache = use_cache
        self.index_cache = cl.defaultdict(dict)

    @util.cached_property
    def wide_data(self):
        return self.dataset.data

    @util.cached_property
    def long_data(self):
        return self.wide_data.melt(
            id_vars="relation", var_name="role", value_name="entity", ignore_index=False
        )

    def neighbourhood(self, entity, depth=1):
        if not self.use_cache or depth not in self.index_cache[entity]:
            idx = self.long_data[self.long_data["entity"] == entity].index

            for _ in range(depth - 1):
                entities = self.long_data.loc[idx]["entity"]
                idx = idx.union(
                    self.long_data[self.long_data["entity"].isin(entities)].index
                )

            idx = idx.unique()

            if self.use_cache:
                self.index_cache[entity][depth] = idx
        else:
            idx = self.index_cache[entity][depth]

        return self.wide_data.loc[idx]

    def enclosing(self, head, tail, **kwargs):
        idx = self.neighbourhood(head, **kwargs).index.intersection(
            self.neighbourhood(tail, **kwargs).index
        )

        return self.wide_data.loc[idx]

    def all_neighbourhoods(self, max_entities=None, max_workers=None, **kwargs):
        entities = self.dataset.entities

        if max_entities is not None:
            entities = rng.choice(entities, max_entities)

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as pool:
            jobs = pool.map(
                ft.partial(self._all_neighbourhoods_worker, **kwargs),
                entities,
                chunksize=100,
            )
            neighbourhoods = list(tqdm.tqdm(jobs, total=len(entities), unit="entities"))

        index, value = zip(*neighbourhoods)

        return pd.Series(value, index=index)

    def _all_neighbourhoods_worker(self, entity, **kwargs):
        return entity, list(self.neighbourhood(entity, **kwargs).index)

    def all_enclosing(self, max_pairs=None, **kwargs):
        pairs = self.wide_data[["head", "tail"]].drop_duplicates()

        if max_pairs is not None:
            pairs = pairs.sample(max_pairs)

        pairs = [tuple(pair) for pair in pairs.itertuples(index=False)]

        with concurrent.futures.ProcessPoolExecutor() as pool:
            jobs = pool.map(
                ft.partial(self._all_enclosing_worker, **kwargs),
                *zip(*pairs),
                chunksize=100
            )
            subgraphs = list(tqdm.tqdm(jobs, total=len(pairs), unit="pairs"))

        index, value = zip(*subgraphs)

        return pd.Series(value, index=index)

    def _all_enclosing_worker(self, head, tail, **kwargs):
        return (head, tail), list(self.enclosing(head, tail, **kwargs).index)
