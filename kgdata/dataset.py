import abc
import itertools as it
import os
import pathlib
import re
import shutil

import networkx as nx
import numpy as np
import pandas as pd
import tqdm

from . import decompress, download, feature, sparql, subgraph, util


@util.delegate(
    "neighbourhood",
    "all_neighbourhoods",
    "enclosing",
    "all_enclosing",
    "all_neighbourhood_sizes",
    "neighbourhood_sizes",
    "enclosing_sizes",
    "all_enclosing_sizes",
    "stochastic_neighbourhood",
    to_attribute="subgraph_extractor",
)
class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    @util.cached_property
    def entities(self):
        return pd.Series(
            pd.concat(
                [self.data["head"], self.data["tail"]], ignore_index=True
            ).unique()
        )

    @util.cached_property
    def relations(self):
        return pd.Series(self.data["relation"].unique())

    @util.cached_property
    def rel_to_idx(self):
        return dict(zip(self.relations, self.relations.index))

    def rel_seq_to_idx(self, rel_seq):
        return [self.rel_to_idx[rel] for rel in rel_seq]

    @util.cached_property
    def stats(self):
        return pd.DataFrame(
            {
                "entities": len(self.entities),
                "relations": len(self.relations),
                "triples": len(self),
            },
            index=[self.__class__.__name__],
        ).assign(triples_per_entity=lambda stats: stats["triples"] / stats["entities"])

    @util.cached_property
    def entity_pairs(self):
        return self.data[["head", "tail"]].drop_duplicates(ignore_index=True)

    @util.cached_property
    def unique_entity_pairs(self):
        sets = set(map(frozenset, self.entity_pairs.itertuples(index=False)))

        data = pd.DataFrame(sets, columns=["entity_1", "entity_2"])

        data.loc[data["entity_1"].isna(), "entity_1"] = data[data["entity_1"].isna()][
            "entity_2"
        ]
        data.loc[data["entity_2"].isna(), "entity_2"] = data[data["entity_2"].isna()][
            "entity_1"
        ]

        return data

    @util.cached_property
    def subgraph_extractor(self):
        return subgraph.Extractor(self)

    def rel_dists(self):
        return feature.rel_dists(self.data)

    @util.cached_property
    def graph(self):
        return nx.MultiDiGraph(
            zip(self.data["head"], self.data["tail"], self.data["relation"])
        )

    @staticmethod
    def load(path):
        return Dataset(pd.read_csv(path, dtype=str))


class PersistedDataset(Dataset):
    NAME = None

    def __init__(self, path, split=None):
        self.path = path

        if not isinstance(self.path, pathlib.Path):
            self.path = pathlib.Path(self.path)

        self.split = split

        if self.split is None:
            self.split = ["train", "valid", "test"]
        elif isinstance(self.split, str):
            self.split = [self.split]

    @util.cached_property
    def name(self):
        return self.__class__.NAME or self.__class__.__name__

    @abc.abstractmethod
    def download(self):
        ...

    def load_split(self, split):
        return self.__class__(self.path, split=split)

    @util.cached_property
    def split_file_names(self):
        return [self.path / f"{split}.csv" for split in self.split]

    @util.cached_property
    def data(self):
        if not self.path.exists():
            self.download()

        return pd.concat(map(pd.read_csv, self.split_file_names), ignore_index=True)

    def subset(self, size, path, force=False):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        if not path.exists() or force:
            path.mkdir(exist_ok=True, parents=True)

            params = {"n" if size > 1 else "frac": size}

            for file in self.split_file_names:
                pd.read_csv(file).sample(**params).to_csv(path / file.name, index=False)

        return self.__class__(path, split=self.split)


class FB15K237Raw(PersistedDataset):
    NAME = "FB15K237"

    def download(self):
        compressed_path = download.download_file(
            "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip",
            self.path,
        )

        decompressed_path = decompress.decompress_zip(
            compressed_path, self.path, keep=True
        )

        source_dir = self.path / "Release"

        for file_name in tqdm.tqdm(
            ["train.txt", "valid.txt", "test.txt"], desc="Moving files", unit="files"
        ):
            pd.read_csv(
                source_dir / file_name,
                sep="\t",
                names=["head", "relation", "tail"],
            ).to_csv((self.path / file_name).with_suffix(".csv"), index=False)

        shutil.rmtree(source_dir)


class FB15K237(PersistedDataset):
    @util.cached_property
    def raw_dataset(self):
        return FB15K237Raw(self.path, split=self.split)

    @util.cached_property
    def data(self):
        data = self.raw_dataset.data.assign(
            head=self.wikidata_labels.loc[self.raw_dataset.data["head"]].values,
            tail=self.wikidata_labels.loc[self.raw_dataset.data["tail"]].values,
        )

        data.loc[data["tail"].isna(), "tail"] = self.raw_dataset.data[
            data["tail"].isna()
        ]["head"]
        data.loc[data["head"].isna(), "head"] = self.raw_dataset.data[
            data["head"].isna()
        ]["tail"]

        return data

    @util.cached_property
    def wikidata_labels(self):
        path = self.path / "wikidata_labels.csv"

        if not path.exists():
            self.get_wikidata_labels().to_csv(path)

        return pd.read_csv(path, index_col=0)

    def get_wikidata_labels(self):
        query = (
            "SELECT ?fb ?itemLabel "
            "WHERE {{ ?item wdt:P646 ?fb. VALUES ?fb {{ {fb_ids} }} "
            "SERVICE wikibase:label {{ bd:serviceParam wikibase:language 'en'. }} }}"
        ).format(
            fb_ids=" ".join([f"'{entity}'" for entity in self.raw_dataset.entities])
        )

        result = sparql.Wikidata().query(query)

        grouped = {
            key: list(value)
            for key, value in it.groupby(
                result.bindings, lambda value: value["fb"]["value"]
            )
        }

        def reduce_group(entity):
            try:
                return list(grouped[entity])[0]["itemLabel"]["value"]
            except (IndexError, ValueError, KeyError):
                return None

        return pd.Series(
            {entity: reduce_group(entity) for entity in self.raw_dataset.entities},
            name="wikidata_label",
        )


class WN18RR(PersistedDataset):
    def download(self):
        compressed_path = download.download_file(
            "https://data.deepai.org/WN18RR.zip", self.path
        )

        decompressed_path = decompress.decompress_zip(
            compressed_path, self.path, keep=True
        )

        for file_name in ["train.txt", "valid.txt", "test.txt"]:
            pd.read_csv(
                self.path / "WN18RR" / "text" / file_name,
                sep="\t",
                names=["head", "relation", "tail"],
            ).to_csv((self.path / file_name).with_suffix(".csv"), index=False)

        shutil.rmtree(self.path / "WN18RR")


class YAGO3(PersistedDataset):
    def download(self):
        compressed_path = download.download_file(
            "https://github.com/TimDettmers/ConvE/raw/5feb358eb7dbd1f534978cdc4c20ee0bf919148a/YAGO3-10.tar.gz",
            self.path,
        )

        decompress.decompress_tar(compressed_path, self.path, keep=True)

        for file_name in tqdm.tqdm(
            ["train.txt", "valid.txt", "test.txt"], desc="Moving files"
        ):
            path = self.path / file_name

            pd.read_csv(
                path,
                sep="\t",
                names=["head", "relation", "tail"],
            ).to_csv(path.with_suffix(".csv"), index=False)

            path.unlink()


class OpenBioLink(PersistedDataset):
    def download(self):
        compressed_path = download.download_file(
            "https://zenodo.org/record/3834052/files/HQ_DIR.zip", self.path
        )

        decompress.decompress_zip(compressed_path, self.path, keep=True)

        file_pairs = [
            ("train", "train"),
            ("val", "valid"),
            ("test", "test"),
            ("negative_train", "train_neg"),
            ("negative_val", "valid_neg"),
            ("negative_test", "test_neg"),
        ]

        for source, target in tqdm.tqdm(file_pairs, desc="Moving files", unit="files"):
            pd.read_csv(
                self.path / "HQ_DIR" / "train_test_data" / f"{source}_sample.csv",
                sep="\t",
                names=["head", "relation", "tail"],
                usecols=[0, 1, 2],
            ).to_csv(self.path / f"{target}.csv", index=False)

        shutil.rmtree(self.path / "HQ_DIR")
