import collections as cl
import concurrent.futures
import functools as ft
import threading
import typing as tp

import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from . import util

rng = np.random.default_rng()


class Extractor:
    def __init__(self, dataset):
        self.dataset = dataset

    @util.cached_property
    def wide_data(self):
        return self.dataset.data

    @util.cached_property
    def long_data(self):
        return self.wide_data.melt(
            id_vars="relation", var_name="role", value_name="entity", ignore_index=False
        )

    def neighbourhood(self, entity: str, **kwargs) -> pd.DataFrame:
        return self._neighbourhood_rec([entity], **kwargs)

    def _neighbourhood_rec(
        self, entities: tp.Iterable[str], depth: int = 1
    ) -> pd.DataFrame:
        idx = self.long_data[self.long_data["entity"].isin(entities)].index

        if depth > 1:
            idx = idx.union(
                self._neighbourhood_rec(
                    self.long_data.loc[idx]["entity"], depth=depth - 1
                ).index
            )

        return self.wide_data.loc[idx.unique()]

    def enclosing(self, head, tail, **kwargs):
        if head == tail:
            return self.neighbourhood(head)

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
