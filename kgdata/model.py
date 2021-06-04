import functools as ft
import math
import pathlib as pl
import typing as tp

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as ptl
import torch.nn as nn
import torch.utils.data

from . import feature, util

rng = np.random.default_rng()


class KG:
    def __init__(self, path: tp.Union[str, pl.Path]):
        self.path = path

    def __len__(self) -> int:
        return len(self.data)

    @util.cached_property
    def data(self) -> pd.DataFrame:
        return pd.read_csv(self.path, dtype=str)

    @util.cached_property
    def graph(self) -> nx.MultiDiGraph:
        return nx.MultiDiGraph(
            zip(self.data["head"], self.data["tail"], self.data["relation"])
        )

    @util.cached_property
    def entities(self) -> pd.Series:
        return pd.Series(pd.concat([self.data["head"], self.data["tail"]]).unique())

    @util.cached_property
    def relations(self) -> pd.Series:
        return pd.Series(self.data["relation"].unique())

    @util.cached_property
    def relation_to_index(self):
        return dict(zip(self.relations, self.relations.index))

    @ft.lru_cache(maxsize=100_000)
    def get_rel_seqs(
        self,
        head: str,
        tail: str,
        min_length: int = 1,
        max_length: int = 3,
        max_paths: int = None,
    ) -> tp.List[np.array]:
        seqs = []

        try:
            for path in nx.all_simple_edge_paths(
                self.graph, head, tail, cutoff=max_length
            ):
                if len(path) < min_length:
                    continue

                seq = [
                    self.relation_to_index[relation] for _head, _tail, relation in path
                ]

                if seq not in seqs:
                    seqs.append(seq)

                if max_paths and len(seqs) >= max_paths:
                    break
        except nx.NodeNotFound:
            pass

        return list(map(np.array, seqs))


class Dataset(torch.utils.data.Dataset):
    path: pl.Path
    split: str
    neg_rate: float

    def __init__(
        self,
        path: tp.Union[str, pl.Path],
        split: str = "train",
        neg_rate: float = 1,
        max_paths: int = None,
        min_path_length: int = 1,
        max_path_length: int = 1,
    ):
        if isinstance(path, str):
            self.path = pl.Path(path)
        else:
            self.path = path

        self.split = split
        self.neg_rate = neg_rate
        self.max_paths = max_paths
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length

    @util.cached_property
    def kg(self) -> KG:
        return KG(self.path / "train.csv")

    @util.cached_property
    def pos_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path / f"{self.split}.csv", dtype=str)

    @util.cached_property
    def replace_tail_probs(self) -> pd.Series:
        return self.kg.data.groupby("relation").apply(
            lambda group: pd.Series(
                {
                    "tph": group.groupby("head").size().sum() / group["head"].nunique(),
                    "hpt": group.groupby("tail").size().sum() / group["tail"].nunique(),
                }
            ).agg(lambda data: data["hpt"] / (data["hpt"] + data["tph"]))
        )

    @util.cached_property
    def entity_semantics(self) -> pd.DataFrame:
        return feature.rel_counts(self.kg.data).astype("float32")

    @util.cached_property
    def default_entity_semantics(self) -> np.array:
        return np.repeat(0, len(self.kg.relations)).astype("float32")

    def __len__(self) -> int:
        return round(len(self.pos_data) * (self.neg_rate + 1))

    def __getitem__(
        self, idx: int
    ) -> tp.Tuple[
        str, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        pos_idx = idx % len(self.pos_data)

        pos_sample = self.pos_data.iloc[pos_idx]

        if pos_idx == idx:
            sample = pos_sample
            label = 1
        else:
            sample = self._gen_neg_sample(*pos_sample)
            label = 0

        head, relation, tail = sample

        # Entity semantics
        if head in self.entity_semantics.index:
            head_sem = self.entity_semantics.loc[head].values
        else:
            head_sem = self.default_entity_semantics

        if tail in self.entity_semantics.index:
            tail_sem = self.entity_semantics.loc[tail].values
        else:
            tail_sem = self.default_entity_semantics

        head_sem = torch.from_numpy(head_sem)
        tail_sem = torch.from_numpy(tail_sem)

        # Relation
        relation_idx = torch.tensor(self.kg.relation_to_index[relation] + 1)

        # Relation sequences
        rel_seqs = [
            torch.from_numpy(rel_seq) + 1
            for rel_seq in self.kg.get_rel_seqs(
                head,
                tail,
                min_length=self.min_path_length,
                max_length=self.max_path_length,
                max_paths=self.max_paths,
            )
        ]

        if len(rel_seqs) == 0:
            rel_seqs = [torch.tensor([0])]

        rel_seqs = nn.utils.rnn.pad_sequence(rel_seqs, batch_first=True)

        # Label
        label = torch.tensor(label, dtype=torch.float32)

        return head, tail, head_sem, tail_sem, relation_idx, rel_seqs, label

    def _gen_neg_sample(
        self, head: str, relation: str, tail: str
    ) -> tp.Tuple[str, str, str]:
        replace_tail = rng.binomial(1, self.replace_tail_probs.loc[relation]) == 1

        if replace_tail:
            positive_tails = self.kg.data[
                (self.kg.data["head"] == head) & (self.kg.data["relation"] == relation)
            ]["tail"]
            new_tail = (
                self.kg.entities[~self.kg.entities.isin(positive_tails)]
                .sample(1)
                .iloc[0]
            )

            return head, relation, new_tail

        positive_heads = self.kg.data[
            (self.kg.data["tail"] == tail) & (self.kg.data["relation"] == relation)
        ]["head"]
        new_head = (
            self.kg.entities[~self.kg.entities.isin(positive_heads)].sample(1).iloc[0]
        )

        return new_head, relation, tail

    @staticmethod
    def collate_fn(batch):
        head, tail, head_sem, tail_sem, relation, seqs, label = zip(*batch)

        return (
            head,
            tail,
            torch.stack(head_sem),
            torch.stack(tail_sem),
            torch.stack(relation),
            Dataset.pad_nested_seqs(seqs),
            torch.stack(label),
        )

    @staticmethod
    def pad_nested_seqs(seq_of_seqs):
        # Largest first and second dimension of all padded seqences.
        dim_1 = max([seq.shape[0] for seq in seq_of_seqs])
        dim_2 = max([seq.shape[1] for seq in seq_of_seqs])

        expanded = [
            torch.nn.functional.pad(
                seq, pad=(0, dim_2 - seq.shape[1], 0, dim_1 - seq.shape[0])
            )
            for seq in seq_of_seqs
        ]

        return torch.stack(expanded)


class DataModule(ptl.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        **dataset_kwargs,
    ):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.dataset_kwargs = dataset_kwargs

        super().__init__()

    @util.cached_property
    def kg(self):
        return self.train_dataloader().dataset.kg

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return self._create_dataloader("test")

    def _create_dataloader(
        self, split: str, **data_loader_kwargs
    ) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            Dataset(self.path, split=split, **self.dataset_kwargs),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=Dataset.collate_fn,
        )
