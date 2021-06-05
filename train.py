import argparse

import pytorch_lightning as ptl

from kgdata.model import DataModule, Model


def main(args):
    data_module = DataModule.from_argparse_args(args)
    trainer = ptl.Trainer.from_argparse_args(args)
    model = Model(n_rels=len(data_module.kg.relations), emb_dim=args.emb_dim)

    trainer.fit(model, data_module)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = ptl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = Model.add_argparse_args(parser)

    main(parser.parse_args())
