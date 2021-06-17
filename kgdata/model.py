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
    path: pl.Path

    def __init__(self, path: tp.Union[str, pl.Path]):
        if isinstance(path, str):
            self.path = pl.Path(path)
        else:
            self.path = path

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return self.data.itertuples()

    @util.cached_property
    def all_org_data(self) -> pd.DataFrame:
        files = [
            self.path.parent / file_name
            for file_name in ["train.csv", "valid.csv", "test.csv"]
        ]

        return pd.concat(map(pd.read_csv, files))

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
        return (
            self.data.reset_index().set_index(["head", "relation"])["tail"].sort_index()
        )

    @util.cached_property
    def tail_relation_data(self) -> pd.Series:
        return (
            self.data.reset_index().set_index(["tail", "relation"])["head"].sort_index()
        )

    @util.cached_property
    def head_index_data(self) -> pd.DataFrame:
        return self.data.reset_index().set_index(["head"]).sort_index()

    @util.cached_property
    def head_index_data_idx(self) -> set:
        return set(self.head_index_data.index)

    @util.cached_property
    def graph(self) -> nx.MultiDiGraph:
        return nx.MultiDiGraph(
            zip(
                self.data["head"],
                self.data["tail"],
                self.data["relation"],
                [{"index": index} for index in self.data.index],
            ),
        )

    @util.cached_property
    def entities(self) -> pd.Series:
        return pd.Series(
            pd.concat([self.all_org_data["head"], self.all_org_data["tail"]]).unique(),
            name="entity",
        )

    @util.cached_property
    def entity_to_index(self):
        return pd.Series(self.entities.index, index=self.entities).sort_index()

    @util.cached_property
    def relations(self) -> pd.Series:
        return pd.Series(self.all_org_data["relation"].unique(), name="relation")

    @util.cached_property
    def relation_to_index(self):
        return pd.Series(self.relations.index, index=self.relations).sort_index()

    @util.cached_property
    def degree(self):
        return pd.Series(dict(self.graph.degree))

    @util.cached_property
    def median_degree(self):
        return round(self.degree.median())

    @util.cached_property
    def mean_degree(self):
        return round(self.degree.mean())

    @ft.lru_cache(maxsize=100_000)
    def _neighbourhood_idx(
        self, entity: str, depth: int = 3, sampling: str = None, sample_size: int = None
    ) -> set:
        if entity not in self.head_index_data_idx:
            return set()

        if sample_size is None and sampling is not None:
            if sampling == "mean":
                sample_size = self.mean_degree
            elif sampling == "median":
                sample_size = self.median_degree
            elif sampling is not None:
                raise ValueError(f"subgraph sampling '{sampling}' is unknown")

        data = self.head_index_data.loc[[entity]]

        if sample_size and len(data) > sample_size:
            data = data.sample(sample_size)

        idx = set(data["index"])

        for _ in range(depth - 1):
            tails = data["tail"]

            if len(tails) == 0:
                break

            data = self.head_index_data.loc[
                self.head_index_data_idx.intersection(tails)
            ]

            if sample_size and len(data) > sample_size:
                data = data.sample(sample_size)

            idx.update(data["index"])

        return idx

    def _enclosing_idx(self, head: str, tail: str, **kwargs):
        return self._neighbourhood_idx(head, **kwargs).intersection(
            self._neighbourhood_idx(tail, **kwargs)
        )

    @ft.lru_cache(maxsize=100_000)
    def get_rel_seqs(
        self,
        head: str,
        tail: str,
        min_length: int = 1,
        max_length: int = 3,
        max_paths: int = None,
        subgraph_sampling: str = None,
        no_rel_rep: bool = False,
    ) -> tp.List[np.array]:
        seqs = []

        if subgraph_sampling:
            idx = self._enclosing_idx(
                head, tail, depth=max_length, sampling=subgraph_sampling
            )

            data = self.data.loc[idx]

            graph = self.graph.edge_subgraph(
                zip(data["head"], data["tail"], data["relation"])
            )
        else:
            graph = self.graph

        try:
            for path in nx.all_simple_edge_paths(graph, head, tail, cutoff=max_length):
                if len(path) < min_length:
                    continue

                seq = [relation for _head, _tail, relation in path]

                if seq not in seqs and (
                    not no_rel_rep or KG._is_non_repeating_seq(seq)
                ):
                    seqs.append(seq)

                if max_paths and len(seqs) >= max_paths:
                    break
        except nx.NodeNotFound:
            pass

        return [np.array(seq) for seq in seqs]

    @staticmethod
    def _is_non_repeating_seq(seq: tp.List[int]) -> bool:
        for index in range(1, len(seq)):
            if seq[index - 1] == seq[index]:
                return False

        return True


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: tp.Union[str, pl.Path],
        split: str = "train",
        neg_rate: float = 1,
        max_paths: int = None,
        min_path_length: int = 1,
        max_path_length: int = 1,
        subgraph_sampling: str = None,
        domain_semantics: bool = False,
        no_rel_rep: bool = False,
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
        self.subgraph_sampling = subgraph_sampling
        self.domain_semantics = domain_semantics

        if self.domain_semantics:
            assert "yago" in str(
                self.path
            ), f"kg '{self.path}' does not support domain semantics"

        self.no_rel_rep = no_rel_rep

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
        if self.domain_semantics:
            return pd.read_csv(self.path / "sems.csv", index_col=0)

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

        if label == 1:
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
                subgraph_sampling=self.subgraph_sampling,
                no_rel_rep=self.no_rel_rep,
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
        subgraph_sampling: str = None,
        batch_size: int = 32,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_train: bool = True,
        domain_semantics: bool = False,
        no_rel_rep: bool = False,
    ):
        self.path = path
        self.neg_rate = neg_rate
        self.max_paths = max_paths
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.subgraph_sampling = subgraph_sampling
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_train = shuffle_train
        self.domain_semantics = domain_semantics
        self.no_rel_rep = no_rel_rep

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
                subgraph_sampling=self.subgraph_sampling,
                domain_semantics=self.domain_semantics,
                no_rel_rep=self.no_rel_rep,
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            collate_fn=Dataset.collate_fn,
            shuffle=self.shuffle_train and split == "train",
            pin_memory=True,
        )


class Model(ptl.LightningModule):
    def __init__(
        self,
        n_rels: int,
        emb_dim: int,
        sem_dim: int = None,
        pooling: str = "avg",
        optimiser: str = "sgd",
        no_early_stopping: bool = False,
        early_stopping: str = "val_loss",
        learning_rate: float = 0.0001,
        batch_size: int = None,
        no_semantics: bool = False,
    ):
        """
        Parameters:
            n_rels: Number of relations in the dataset.
            emb_dim: Dimentionality of relation embeddings.
        """
        super().__init__()

        assert pooling in ["avg", "lse", "max"], f"pooling function '{pooling}' unknown"
        assert optimiser in ["sgd", "adam"], f"optimiser '{optimiser}' unknown"

        if sem_dim is None:
            sem_dim = n_rels

        self.save_hyperparameters(
            "n_rels",
            "emb_dim",
            "sem_dim",
            "pooling",
            "optimiser",
            "no_early_stopping",
            "early_stopping",
            "learning_rate",
            "batch_size",
            "no_semantics",
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
        if not self.hparams.no_semantics:
            self.ent_comp = nn.Parameter(
                torch.rand(
                    self.hparams.emb_dim,
                    2 * self.hparams.sem_dim + self.hparams.emb_dim,
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

        if not self.hparams.no_semantics:
            path_emb = torch.cat(
                [
                    head_sem.unsqueeze(1).repeat_interleave(n_paths, dim=1),
                    path_emb,
                    tail_sem.unsqueeze(1).repeat_interleave(n_paths, dim=1),
                ],
                dim=2,
            )
            # (batch_size, n_paths, emb_dim)
            path_emb = torch.sigmoid(
                torch.matmul(self.ent_comp, path_emb.unsqueeze(-1)).squeeze(-1)
            )

            rel_emb = torch.cat([head_sem, rel_emb, tail_sem], dim=1)
            rel_emb = torch.sigmoid(
                torch.matmul(self.ent_comp, rel_emb.unsqueeze(-1)).squeeze(-1)
            )

        # (batch_size, n_paths)
        similarities = torch.matmul(path_emb, rel_emb.unsqueeze(-1)).squeeze(-1)

        # (batch_size)
        if self.hparams.pooling == "avg":
            agg = torch.mean(similarities, dim=1)
        elif self.hparams.pooling == "lse":
            agg = torch.logsumexp(similarities, dim=1)
        elif self.hparams.pooling == "max":
            agg, _ = torch.max(similarities, dim=1)

        return torch.sigmoid(agg)

    def configure_optimizers(self):
        if self.hparams.optimiser == "sgd":
            optim_class = optim.SGD
        elif self.hparams.optimiser == "adam":
            optim_class = optim.Adam

        return optim_class(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self) -> tp.List[ptl.callbacks.Callback]:
        monitor_mode = "min" if self.hparams.early_stopping == "val_loss" else "max"

        callbacks = []

        # callbacks = [
        #     ptl.callbacks.ModelCheckpoint(
        #         monitor=self.hparams.early_stopping, mode=monitor_mode
        #     )
        # ]

        if not self.hparams.no_early_stopping:
            callbacks.append(
                ptl.callbacks.EarlyStopping(
                    monitor=self.hparams.early_stopping, mode=monitor_mode
                )
            )

        return callbacks

    def training_step(self, batch, _batch_idx):
        _head, _tail, head_sem, tail_sem, relation, path, label = batch

        pred = self(path, relation, head_sem=head_sem, tail_sem=tail_sem)
        loss = F.binary_cross_entropy(pred, label)

        # Log
        self.log("train_loss", loss, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _head, _tail, head_sem, tail_sem, relation, path, label = batch

        pred = self(path, relation, head_sem=head_sem, tail_sem=tail_sem)
        loss = F.binary_cross_entropy(pred, label)

        # Compute and log metrics.
        label = label.int()
        retr_idx = torch.tensor(batch_idx).expand_as(label)
        # retr_idx = batch_idx.clone().detach().expand_as(label)
        metrics = self.val_metrics(pred, label, retr_idx)

        self.log("val_loss", loss, sync_dist=True)
        self.log_dict(metrics, sync_dist=True)

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
        group_parser.add_argument("--sem_dim", type=int)
        group_parser.add_argument("--pooling", type=str, default="avg")
        group_parser.add_argument("--optimiser", type=str, default="sgd")
        group_parser.add_argument("--early_stopping", type=str, default="val_loss")
        group_parser.add_argument("--no_early_stopping", action="store_true")
        group_parser.add_argument("--learning_rate", type=float, default=0.0001)
        group_parser.add_argument("--no_semantics", action="store_true")

        return parent_parser
