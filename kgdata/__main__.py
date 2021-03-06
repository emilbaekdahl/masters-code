import pathlib as pl

import click
import pandas as pd

import kgdata.dataset
import kgdata.path
import kgdata.sample

DATASET_MAP = {
    "fb": kgdata.dataset.FB15K237Raw,
    "wn": kgdata.dataset.WN18RR,
    "yago": kgdata.dataset.YAGO3,
    "bio": kgdata.dataset.OpenBioLink,
}


dataset_choices = click.Choice(list(DATASET_MAP.keys()))


def dataset_class_from_string(dataset):
    if dataset not in DATASET_MAP:
        raise ValueError(f"dataset '{dataset}' unknown")

    return DATASET_MAP[dataset]


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("target", type=click.Path(file_okay=False, writable=True))
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
@click.option("--max-workers", type=int)
def neg_samples(dataset, source, splits, neg_rate, seed, max_workers):
    source = pl.Path(source)
    dataset_class = dataset_class_from_string(dataset)

    for split in splits:
        dataset = dataset_class(source, split=split)

        kgdata.sample.NegativeSampler.generate_samples(
            dataset.data, neg_rate=neg_rate, seed=seed, max_workers=max_workers
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
@click.option("--max-workers", type=int)
@click.option("--stochastic/--no-stochastic", default=False)
def paths(
    dataset, source, splits, depth, length, max_pairs, seed, max_workers, stochastic
):
    source = pl.Path(source)
    dataset_class = dataset_class_from_string(dataset)
    min_length, max_length = length

    if max_pairs and int(max_pairs) == max_pairs:
        max_pairs = int(max_pairs)

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
            max_workers=max_workers,
            stochastic=stochastic,
        ).to_csv(source / f"{split}_paths.csv", index=False)


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("source", type=click.Path(file_okay=False))
@click.argument("target", type=click.Path(dir_okay=False, writable=True))
@click.option("--split")
@click.option("--depth", type=int, default=2, show_default=True)
@click.option("--max-pairs", type=float)
@click.option("--max-workers", type=int)
@click.option("--stochastic/--no-stochastic", default=True)
def enclosing_sizes(
    dataset, source, target, split, depth, max_pairs, max_workers, stochastic
):
    if max_pairs and int(max_pairs) == max_pairs:
        max_pairs = int(max_pairs)

    dataset_class = dataset_class_from_string(dataset)
    data = dataset_class(source, split=split)
    data.all_enclosing_sizes(
        depth=depth, max_pairs=max_pairs, max_workers=max_workers, stochastic=stochastic
    ).to_csv(target)


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("source", type=click.Path(file_okay=False))
@click.option("--depth", "-d", type=int, default=1)
@click.option("--max-entities", type=float)
@click.option("--max-workers", type=int)
@click.option("--stochastic/--no-stochastic", default=False)
@click.option("--chunk-size", type=int)
def neighbourhoods(
    dataset, source, depth, max_entities, max_workers, stochastic, chunk_size
):
    dataset_class = dataset_class_from_string(dataset)
    dataset = dataset_class(source)

    target_folder = pl.Path(source) / "neighbourhoods"

    if max_entities and int(max_entities) == max_entities:
        max_entities = int(max_entities)

    if stochastic:
        target_folder = target_folder / "stochastic"

    target_folder.mkdir(exist_ok=True, parents=True)

    dataset.all_neighbourhoods(
        depth=depth,
        max_entities=max_entities,
        max_workers=max_workers,
        stochastic=stochastic,
        chunk_size=chunk_size,
    ).groupby(level=0).apply(list).to_csv(target_folder / f"{depth}.csv")


@cli.command()
@click.argument("dataset", type=dataset_choices)
@click.argument("source", type=click.Path(file_okay=False))
@click.option("--depth", "-d", type=int, default=1)
@click.option("--max-pairs", type=float)
@click.option("--max-paths", type=int)
def nx_paths(dataset, source, depth, max_pairs, max_paths):
    dataset_class = dataset_class_from_string(dataset)
    dataset = dataset_class(source)

    if max_pairs and max_pairs == int(max_pairs):
        max_pairs = int(max_pairs)

    for split in dataset.split:
        split_dataset = dataset_class(source, split=set([split] + ["train"]))

        rel_seqs = kgdata.path.all_nx_rel_seqs(
            split_dataset, max_pairs=max_pairs, max_paths=max_paths, depth=depth
        )

        (dataset.path / "rel_seqs").mkdir(exist_ok=True)

        rel_seqs.to_csv(dataset.path / "rel_seqs" / f"{split}_{depth}.csv")


if __name__ == "__main__":
    cli()
