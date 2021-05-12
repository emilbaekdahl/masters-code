import click
import pandas as pd

import kgdata.dataset

from . import sample


@click.group()
def cli():
    pass


@cli.command()
@click.argument("source", type=click.Path(dir_okay=False, exists=True))
@click.argument("target", type=click.Path(dir_okay=False, writable=True))
@click.option("--neg-rate", "-n", type=float, default=1)
def neg_samples(source, target, neg_rate):
    data = pd.read_csv(source)
    neg_samples = sample.NegativeSampler.generate_samples(data, neg_rate=neg_rate)
    neg_samples.to_csv(target, index=False)


@cli.command()
@click.argument("source", type=click.Path(dir_okay=False, exists=True))
def paths(souce):
    data = pd.read_csv(source)


@cli.command()
@click.argument("dataset", type=click.Choice(["fb"]))
@click.argument("source", type=click.Path(file_okay=False))
@click.argument("target", type=click.Path(dir_okay=False, writable=True))
@click.option("--depth", type=(int, int), default=(1, 3))
@click.option("--max-pairs", type=int)
def enclosing_sizes(dataset, source, target, depth, max_pairs):
    if dataset == "fb":
        data = kgdata.dataset.FB15K237Raw(source)

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
