from argparse import ArgumentParser, Namespace


def parse_arguments() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--language-pair",
        dest="language_pair",
        help="Language pair to train on",
        type=str,
        default="cpp_java"
    )

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
        default=1e-05
    )

    parser.add_argument(
        "-ab",
        "--adam-betas",
        dest="adam_betas",
        help="Beta 1 and 2 of ADAM optimizer",
        type=str,
        default="0.9, 0.98"
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        dest="batch_size",
        help="Batch size of the data loader",
        type=int,
        default=32
    )

    parser.add_argument(
        "-k",
        "--max-k",
        dest="max_k",
        help="Maximum number of neighbors to retrieve from the datastore",
        type=int,
        default=32
    )

    parser.add_argument(
        "--tc_k",
        dest="tc_k",
        help="Number of scores to consider from the TransCoder models",
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
        "--knn-temperature",
        dest="knn_temperature",
        help="KNN distance temperature",
        type=int,
        default=10
    )

    parser.add_argument(
        "--tc-temperature",
        dest="tc_temperature",
        help="TransCoder distance temperature",
        type=int,
        default=3
    )

    parser.add_argument(
        "--vocab-size",
        dest="vocab_size",
        help="Size of vocabulary",
        type=int,
        default=64000
    )

    parser.add_argument(
        "--checkpoint-path",
        dest="checkpoint_path",
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
        default="dump/adaptive_knnmt/logs"
    )

    parser.add_argument(
        "--checkpoint-dir",
        dest="checkpoint_dir",
        help="Directory to save the model checkpoints",
        type=str,
        default="out/adaptive_knnmt/checkpoints"
    )

    parser.add_argument(
        "--dataset-dir",
        dest="dataset_dir",
        help="Path to the directory containing the dataset",
        type=str,
        default="data/test_dataset"
    )

    parser.add_argument(
        "--knnmt-dir",
        dest="knnmt_dir",
        help="Path to the directory containing the KNN datastore and faiss index",
        type=str,
        default="out/knnmt/validation_set_half"
    )

    parser.add_argument(
        "--model-dir",
        dest="model_dir",
        help="Path to the directory containing the TransCoder models",
        type=str,
        default="models"
    )

    parser.add_argument(
        "--cache-dir",
        dest="cache_dir",
        help="Directory to cache the dataset",
        type=str,
        default="out/adaptive_knnmt/cache"
    )

    parser.add_argument(
        "-bpe",
        "--bpe-path",
        dest="bpe_path",
        help="Path to the bpe codes",
        type=str,
        default="data/bpe/cpp-java-python/codes"
    )

    return parser.parse_args()
