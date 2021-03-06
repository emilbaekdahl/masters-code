import concurrent.futures
import os

import numpy as np
import pandas as pd
import tqdm.auto as tqdm

from . import util

rng = np.random.default_rng()


def gen_neg_samples(dataset):
    data = dataset.data.join(_replace_tail_prob(dataset.data), on="relation").assign(
        replace_tail=lambda data: rng.binomial(1, data["replace_tail_prob"])
    )

    data[data["replace_tail"] == 1]["tail"] = (
        data[data["replace_tail"] == 1]
        .groupby(["head", "relation"], as_index=False)["tail"]
        .apply(
            lambda tail: pd.Series(
                dataset.entities[~dataset.entities.isin(tail)].sample(len(tail)),
                index=["tail"] * len(tail),
            )
        )
    )

    data[data["replace_tail"] == 0]["head"] = (
        data[data["replace_tail"] == 0]
        .groupby(["tail", "relation"], as_index=False)["head"]
        .apply(
            lambda head: pd.Series(
                dataset.entities[~dataset.entities.isin(head)].sample(len(head)),
                index=["head"] * len(head),
            )
        )
    )

    return data[["head", "relation", "tail"]]


def _replace_tail_prob(data):
    return (
        data.groupby("relation")
        .apply(
            lambda group: pd.Series(
                {
                    "tph": group.groupby("head").size().sum() / group["head"].nunique(),
                    "hpt": group.groupby("tail").size().sum() / group["tail"].nunique(),
                }
            )
        )
        .agg(lambda data: data["hpt"] / (data["hpt"] + data["tph"]), axis=1)
    ).rename("replace_tail_prob")


class NegativeSampler:
    def __init__(self, data, seed=None):
        self.data = data
        self.seed = seed

    def __call__(self, head, relation, tail):
        replace_tail = self.rng.binomial(1, self.replace_tail_probs[relation])

        triples = self.data[self.data["relation"] == relation]

        if replace_tail:
            invalid_entities = triples[triples["head"] == head]["tail"]
        else:
            invalid_entities = triples[triples["tail"] == tail]["head"]

        new_entity = (
            self.entities[~self.entities.isin(invalid_entities)]
            .sample(1, random_state=self.seed)
            .iloc[0]
        )

        if replace_tail:
            return head, relation, new_entity

        return new_entity, relation, tail

    def generate(self, triples, chunk_size=100, max_workers=None):
        if max_workers is None and "SLURM_CPUS_PER_TASK" in os.environ:
            max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

        with concurrent.futures.ProcessPoolExecutor(max_workers) as pool:
            jobs = pool.map(self, *zip(*triples), chunksize=chunk_size)
            samples = list(tqdm.tqdm(jobs, total=len(triples)))

        return pd.DataFrame(samples, columns=["head", "relation", "tail"])

    @util.cached_property
    def rng(self):
        return np.random.default_rng(seed=self.seed)

    @util.cached_property
    def entities(self):
        return pd.Series(
            pd.concat(
                [self.data["head"], self.data["tail"]], ignore_index=True
            ).unique()
        )

    @util.cached_property
    def replace_tail_probs(self):
        probs = self.data.groupby("relation").apply(
            lambda group: pd.Series(
                {
                    "tph": group.groupby("head").size().sum()
                    / len(group["head"].unique()),
                    "hpt": group.groupby("tail").size().sum()
                    / len(group["tail"].unique()),
                }
            )
        )

        return probs["hpt"] / (probs["hpt"] + probs["tph"])

    @staticmethod
    def generate_samples(data, neg_rate=1, max_workers=None, **kwargs):
        sampler = NegativeSampler(data, **kwargs)

        pos_samples = data.sample(frac=neg_rate, replace=neg_rate > 1)

        if max_workers is None and "SLURM_CPUS_PER_TASK" in os.environ:
            max_workers = int(os.environ["SLURM_CPUS_PER_TASK"])

        with concurrent.futures.ProcessPoolExecutor(max_workers) as pool:
            neg_samples = list(
                tqdm.tqdm(
                    pool.map(
                        sampler,
                        *zip(*pos_samples.itertuples(index=False)),
                        chunksize=100,
                    ),
                    desc=f"Generating negative samples (rate {neg_rate})",
                    total=len(pos_samples),
                    unit="triples",
                )
            )

        return pd.DataFrame(neg_samples, columns=["head", "relation", "tail"])
