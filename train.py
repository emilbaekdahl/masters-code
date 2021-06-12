import argparse
import os.path

import pytorch_lightning as ptl
import pytorch_lightning.loggers

from kgdata.model import DataModule, Model


def main(args):
    data_module = DataModule.from_argparse_args(args)

    trainer = ptl.Trainer.from_argparse_args(
        args,
        logger=ptl.loggers.TensorBoardLogger(
            "lightning_logs", name=os.path.basename(args.path)
        ),
    )

    model = Model(
        n_rels=len(data_module.kg.relations),
        emb_dim=args.emb_dim,
        pooling=args.pooling,
        optimiser=args.optimiser,
        early_stopping=args.early_stopping,
        no_early_stopping=args.no_early_stopping,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
    )

    trainer.fit(model, data_module)
    trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser = ptl.Trainer.add_argparse_args(parser)
    parser = DataModule.add_argparse_args(parser)
    parser = Model.add_argparse_args(parser)

    main(parser.parse_args())
