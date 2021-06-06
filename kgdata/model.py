import argparse
import functools as ft
import pathlib as pl
import typing as tp

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as ptl
import pytorch_lightning.callbacks
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchmetrics as tm

from . import feature, util

rng = np.random.default_rng()


class KG:
    def __init__(self, path: tp.Union[str, pl.Path]):
        self.path = path

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return self.data.itertuples()

    @util.cached_property
    def org_data(self) -> pd.DataFrame:
        return pd.read_csv(self.path, dtype=str)

    @util.cached_property
    def data(self) -> pd.DataFrame:
        return self.org_data.assign(
            head=lambda data: self.entity_to_index.loc[data["head"]].values,
            relation=lambda data: self.relation_to_index.loc[data["relation"]].values,
            tail=lambda data: self.entity_to_index.loc[data["tail"]].values,
        )

    @util.cached_property
    def head_relation_data(self) -> pd.Series:
        return self.data.set_index(["head", "relation"])["tail"].sort_index()

    @util.cached_property
    def tail_relation_data(self) -> pd.Series:
        return self.data.set_index(["tail", "relation"])["head"].sort_index()

    @util.cached_property
    def graph(self) -> nx.MultiDiGraph:
        return nx.MultiDiGraph(
            zip(self.data["head"], self.data["tail"], self.data["relation"])
        )

    @util.cached_property
    def entities(self) -> pd.Series:
        return pd.Series(
            pd.concat([self.org_data["head"], self.org_data["tail"]]).unique(),
            name="entity",
        )

    @util.cached_property
    def entity_to_index(self):
        return pd.Series(self.entities.index, index=self.entities).sort_index()

    @util.cached_property
    def relations(self) -> pd.Series:
        return pd.Series(self.org_data["relation"].unique(), name="relation")

    @util.cached_property
    def relation_to_index(self):
        return pd.Series(self.relations.index, index=self.relations).sort_index()

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

                seq = [relation for _head, _tail, relation in path]

                if seq not in seqs:
                    seqs.append(seq)

                if max_paths and len(seqs) >= max_paths:
                    break
        except nx.NodeNotFound:
            pass

        return list(map(np.array, seqs))


class Dataset(torch.utils.data.Dataset):
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
        return KG(self.path / f"{self.split}.csv").data

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

    @util.cached_property
    def idx_map(self):
        pos_idx_map = pd.DataFrame({"pos_index": self.pos_data.index, "label": 1})
        neg_idx_map = pos_idx_map.sample(
            frac=self.neg_rate, replace=self.neg_rate > 1
        ).assign(label=0)

        return pd.concat([pos_idx_map, neg_idx_map]).sort_index().reset_index(drop=True)

    def __len__(self) -> int:
        return round(len(self.pos_data) * (self.neg_rate + 1))

    def __getitem__(
        self, idx: int
    ) -> tp.Tuple[
        str, str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        pos_idx, label = self.idx_map.iloc[idx]

        pos_sample = self.pos_data.iloc[pos_idx]

        if pos_idx == idx:
            sample = pos_sample
        else:
            sample = self._gen_neg_sample(*pos_sample)

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
        relation = torch.tensor(relation + 1)

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

        return head, tail, head_sem, tail_sem, relation, rel_seqs, label

    def _gen_neg_sample(
        self, head: str, relation: str, tail: str
    ) -> tp.Tuple[str, str, str]:
        replace_tail = rng.binomial(1, self.replace_tail_probs[relation]) == 1

        try:
            if replace_tail:
                invalid_entities = self.kg.head_relation_data[head, relation]
            else:
                invalid_entities = self.kg.tail_relation_data[tail, relation]

            candidate_entities = self.kg.entities.index[
                ~self.kg.entities.index.isin(invalid_entities)
            ]
        except KeyError:
            candidate_entities = self.kg.entities.index

        (new_entity,) = rng.choice(candidate_entities, 1)

        if replace_tail:
            return head, relation, new_entity

        return new_entity, relation, tail

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
        neg_rate: float = 1,
        max_paths: int = None,
        min_path_length: int = 1,
        max_path_length: int = 3,
        batch_size: int = 32,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_train: bool = True,
    ):
        self.path = path
        self.neg_rate = neg_rate
        self.max_paths = max_paths
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_train = shuffle_train

        super().__init__()

    @util.cached_property
    def kg(self) -> KG:
        return self.train_dataloader().dataset.kg

    def train_dataloader(self):
        return self._create_dataloader("train")

    def val_dataloader(self):
        return self._create_dataloader("valid")

    def test_dataloader(self):
        return self._create_dataloader("test")

    def _create_dataloader(self, split: str) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            Dataset(
                self.path,
                split=split,
                neg_rate=self.neg_rate,
                max_paths=self.max_paths,
                min_path_length=self.min_path_length,
                max_path_length=self.max_path_length,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=Dataset.collate_fn,
            shuffle=self.shuffle_train and split == "train",
        )


class Model(ptl.LightningModule):
    def __init__(
        self,
        n_rels: int,
        emb_dim: int,
        pool: str = "avg",
        optimiser: str = "sgd",
        early_stopping: bool = True,
        learning_rate: float = 0.0001,
    ):
        """
        Parameters:
            n_rels: Number of relations in the dataset.
            emb_dim: Dimentionality of relation embeddings.
        """
        super().__init__()

        assert pool in ["avg", "lse", "max"], f"pooling function '{pool}' unknown"
        assert optimiser in ["sgd", "adam"], f"optimiser '{optimiser}' unknown"

        self.save_hyperparameters(
            "n_rels", "emb_dim", "pool", "optimiser", "early_stopping", "learning_rate"
        )

        # (n_rels, emb_dim + 1)
        # +1 to account for padding_idx
        self.rel_emb = nn.Embedding(
            self.hparams.n_rels + 1, self.hparams.emb_dim, padding_idx=0
        )

        # (emb_dim, 2 * emb_dim)
        self.comp = nn.Parameter(
            torch.rand(self.hparams.emb_dim, 2 * self.hparams.emb_dim)
        )
        nn.init.xavier_uniform_(self.comp.data)

        # (emb_dim, 2 * n_rels + emb_dim)
        self.ent_comp = nn.Parameter(
            torch.rand(
                self.hparams.emb_dim, 2 * self.hparams.n_rels + self.hparams.emb_dim
            )
        )
        nn.init.xavier_uniform_(self.ent_comp.data)

        # Setup metrics
        metrics = tm.MetricCollection(
            {
                "mrr": tm.RetrievalMRR(),
                **{f"h@{k}": tm.RetrievalPrecision(k=k) for k in [1, 3, 10]},
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, path, relation, head_sem=None, tail_sem=None):
        """
        Parameters:
            path: (batch_size, n_paths, path_length)
            relation: (batch_size)

        Return:
            (batch_size)
        """
        n_paths = path.size()[1]

        # (batch_size, n_paths, emb_dim)
        path_emb = self._encode_emb_path(self.rel_emb(path))

        # (batch_size, emb_dim)
        rel_emb = self.rel_emb(relation)

        path_emb = torch.cat(
            [
                head_sem.unsqueeze(1).repeat_interleave(n_paths, dim=1),
                path_emb,
                tail_sem.unsqueeze(1).repeat_interleave(n_paths, dim=1),
            ],
            dim=2,
        )
        path_emb = torch.matmul(self.ent_comp, path_emb.unsqueeze(-1)).squeeze()

        rel_emb = torch.cat([head_sem, rel_emb, tail_sem], dim=1)
        rel_emb = torch.matmul(self.ent_comp, rel_emb.unsqueeze(-1)).squeeze()

        # (batch_size, n_paths)
        similarities = torch.sigmoid(
            torch.matmul(path_emb, rel_emb.unsqueeze(-1))
        ).squeeze()

        # (batch_size)
        if self.hparams.pool == "avg":
            agg = torch.mean(similarities, dim=1)
        elif self.hparams.pool == "lse":
            agg = torch.sigmoid(torch.logsumexp(similarities, dim=1))
        elif self.hparams.pool == "max":
            agg, _ = torch.max(similarities, dim=1)

        return agg

    def configure_optimizers(self):
        if self.hparams.optimiser == "sgd":
            optim_class = optim.SGD
        elif self.hparams.optimiser == "adam":
            optim_class = optim.Adam

        return optim_class(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self) -> tp.List[ptl.callbacks.Callback]:
        if self.hparams.early_stopping:
            return [ptl.callbacks.EarlyStopping(monitor="val_loss")]

        return []

    def training_step(self, batch, _batch_idx):
        _head, _tail, head_sem, tail_sem, relation, path, label = batch

        pred = self(path, relation, head_sem=head_sem, tail_sem=tail_sem)
        loss = F.binary_cross_entropy(pred, label)

        # Log
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        _head, _tail, head_sem, tail_sem, relation, path, label = batch

        pred = self(path, relation, head_sem=head_sem, tail_sem=tail_sem)
        loss = F.binary_cross_entropy(pred, label)

        # Compute and log metrics.
        label = label.int()
        retr_idx = torch.tensor(batch_idx).expand_as(label)
        metrics = self.val_metrics(pred, label, retr_idx)

        self.log_dict({"val_loss": loss, **metrics})

    def test_step(self, batch, batch_idx):
        _head, _tail, head_sem, tail_sem, relation, path, label = batch

        pred = self(path, relation, head_sem=head_sem, tail_sem=tail_sem)

        # Compute and log metrics.
        label = label.int()
        retr_idx = torch.tensor(batch_idx).expand_as(label)
        metrics = self.test_metrics(pred, label, retr_idx)

        self.log_dict(metrics)

    def _encode_emb_path(self, path):
        """
        Parameters:
            path: (batch_size, n_paths, path_length, emb_dim)

        Return:
            (batch_size, n_paths, emb_dim)
        """
        # (batch_size, n_paths, path_length - 1, emb_dim), (batch_size, n_paths, 1, emb_dim)
        head, tail = torch.split(path, [path.shape[2] - 1, 1], dim=2)

        # (batch_size, n_paths, emb_dim)
        tail = tail.squeeze(2)

        if head.shape[2] == 0:
            return tail

        # (batch_size, n_paths, emb_dim)
        head = self._encode_emb_path(head)

        # (batch_size, n_paths, emb_size * 2)
        stack = torch.cat([head, tail], dim=2)

        # (batch_size, n_paths, emb_dim)
        product = torch.matmul(self.comp, stack.unsqueeze(-1)).squeeze(-1)

        return torch.sigmoid(product)

    @classmethod
    def add_argparse_args(
        cls, parent_parser: argparse.ArgumentParser
    ) -> argparse.ArgumentParser:
        group_parser = parent_parser.add_argument_group("Model")
        group_parser.add_argument("--emb_dim", type=int, default=100)
        group_parser.add_argument("--pooling", type=str, default="avg")
        group_parser.add_argument("--optimiser", type=str, default="sgd")

        return parent_parser
