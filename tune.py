import test_tube

from kgdata.model import DataModule, Model


def main():
    ...


if __name__ == "__main__":
    parser = test_tube.HyperOptArgumentParser(strategy="grid_search")
    parser.opt_list(
        "--emb_dim", type=int, default=100, options=[10, 100, 1000], tunable=True
    )
    parser.opt_list(
        "--optimiser", type=str, default="sgd", options=["sgd", "adam"], tunable=True
    )

    hparams = parser.parse_args()

    print(list(hparams.trials(1)))

    cluster = test_tube.SlurmCluster(
        hyperparam_optimizer=hparams, log_path="test/log/dir"
    )

    cluster.per_experiment_nb_gpus = 2
    cluster.per_experiment_nb_cpus = 4

    cluster.notify_job_status("ebakda16@student.aau.dk", on_done=True, on_fail=True)
