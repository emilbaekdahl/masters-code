import pathlib as pl

import click
import pandas as pd

import kgdata.dataset
import kgdata.path
import kgdata.sample

dataset_choices = click.Choice(["wn", "fb", "yago"])


def dataset_class_from_string(dataset):
    if dataset == "fb":
        return kgdata.dataset.FB15K237Raw
    elif dataset == "wn":
        return kgdata.dataset.WN18RR
    elif dataset == "yago":
        return kgdata.dataset.YAGO3

    raise ValueError(f"dataset '{dataset}' unknown")


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("target", type=click.Path(file_okay=False))
def download(dataset, target):
    dataset_class = dataset_class_from_string(dataset)
    dataset_class(target).download()


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("source", type=click.Path(file_okay=False, exists=True))
@click.option(
    "--split",
    "-s",
    "splits",
    multiple=True,
    default=["train", "valid", "test"],
    show_default=True,
)
@click.option("--neg-rate", "-n", type=float, default=1, show_default=True)
@click.option("--seed", type=int, default=1, show_default=True)
def neg_samples(dataset, source, splits, neg_rate, seed):
    source = pl.Path(source)
    dataset_class = dataset_class_from_string(dataset)

    for split in splits:
        dataset = dataset_class(source, split=split)

        kgdata.sample.NegativeSampler.generate_samples(
            dataset.data, neg_rate=neg_rate, seed=seed
        ).to_csv((source / f"{split}_neg.csv"), index=False)


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--split",
    "-s",
    "splits",
    multiple=True,
    default=["train", "valid", "test", "train_neg", "valid_neg", "test_neg"],
    show_default=True,
)
@click.option("--depth", "-d", type=int, default=2, show_default=True)
@click.option("--length", "-l", type=(int, int), default=(1, 3), show_default=True)
@click.option("--max-pairs", "-m", type=float)
@click.option("--seed", type=int, default=1, show_default=True)
def paths(dataset, source, splits, depth, length, max_pairs, seed):
    source = pl.Path(source)
    dataset_class = dataset_class_from_string(dataset)
    min_length, max_length = length

    for split in splits:
        if not (source / f"{split}.csv").exists():
            continue

        combined_dataset = dataset_class(source, split=set([split, "train"]))
        split_dataset = dataset_class(source, split=split)
        pairs = split_dataset.unique_entity_pairs

        if max_pairs is not None:
            params = {"n" if max_pairs > 1 else "frac": max_pairs, "random_state": seed}
            pairs = pairs.sample(**params)

        pairs = [tuple(pair) for pair in pairs.itertuples(index=False)]

        kgdata.path.all_relation_paths(
            combined_dataset,
            pairs,
            depth=depth,
            min_length=min_length,
            max_length=max_length,
        ).to_csv(source / f"{split}_paths.csv", index=False)


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("source", type=click.Path(file_okay=False))
@click.argument("target", type=click.Path(dir_okay=False, writable=True))
@click.option("--depth", type=(int, int), default=(1, 3))
@click.option("--max-pairs", type=int)
def enclosing_sizes(dataset, source, target, depth, max_pairs):
    dataset_class = dataset_class_from_string(dataset)

    data = dataset_class(source)

    min_depth, max_depth = depth

    pd.concat(
        [
            data.all_enclosing(depth=depth, max_pairs=max_pairs)
            .map(len)
            .to_frame(name="size")
            .assign(depth=depth, prop=lambda size_data: size_data["size"] / len(data))
            for depth in range(min_depth, max_depth + 1)
        ]
    ).to_csv(target)


if __name__ == "__main__":
    cli()
