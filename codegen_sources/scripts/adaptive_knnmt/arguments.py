from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "-e",
        "--epochs",
        dest="epochs",
        help="Number of epochs for training",
        type=int,
        default=100
    )

    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        help="Initial learning rate",
        type=float,
        default=3e-4
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        help="Bath size of the data loader",
        type=int,
        default=100
    )

    parser.add_argument(
        "-s",
        "--samples",
        dest="samples",
        help="Number of samples to take from the dataset",
        type=int,
        default=100000
    )

    parser.add_argument(
        "-k",
        "--k",
        dest="k",
        help="Number of neighbors to retrieve from the datastore",
        type=int,
        default=32
    )

    parser.add_argument(
        "-hs",
        "--hidden-size",
        dest="hidden_size",
        help="Hidden size of FFN layers",
        type=int,
        default=32
    )

    parser.add_argument(
        "-t",
        "--temperature",
        dest="temperature",
        help="KNN distance temperature",
        type=int,
        default=10
    )

    parser.add_argument(
        "-v",
        "--vocab-size",
        dest="vocab_size",
        help="Size of vocabulary",
        type=int,
        default=64000
    )

    parser.add_argument(
        "-c",
        "--ckpt-path",
        dest="ckpt_path",
        help="Path to a checkpoint to resume the training",
        type=str,
        default=None
    )

    parser.add_argument(
        "-log",
        "--log-dir",
        dest="log_dir",
        help="Directory to save the training logs",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/dump/adaptive_knnmt/logs"
    )

    parser.add_argument(
        "-ckpt",
        "--checkpoint-dir",
        dest="checkpoint_dir",
        help="Directory to save the model checkpoints",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/dump/adaptive_knnmt/checkpoints"
    )

    parser.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        help="Path to the directory containing the dataset",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/dataset/offline_dataset"
    )

    parser.add_argument(
        "--datastore-dir",
        dest="datastore_dir",
        help="Path to the directory containing the datastore",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/datastore"
    )

    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        help="Path to the directory containing the TransCoder models",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/models"
    )

    parser.add_argument(
        "-ca",
        "--cache-dir",
        dest="cache_dir",
        help="Directory to cache the dataset",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/dump/adaptive_knnmt/cache"
    )

    parser.add_argument(
        "-bpe",
        "--bpe-path",
        dest="bpe_path",
        help="Path to the bpe codes",
        type=str,
        default="/pfs/work7/workspace/scratch/hd_tf268-code-gen/bpe/cpp-java-python/codes"
    )

    parser.add_argument(
        "--language-pair",
        dest="language_pair",
        help="Language pair to train on",
        type=str,
        default="cpp_java"
    )

    return parser.parse_args()
