from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def get_data_dir(scope, dataset):
    dataset_path = Path(dataset)
    dataset_dir = DATA_ROOT / scope / dataset_path
    nested_dir = dataset_dir / dataset_path.name
    if nested_dir.is_dir():
        return nested_dir
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. "
            f"Expected data under data/{scope}/{dataset}."
        )
    return dataset_dir


def get_common_data_dir(dataset):
    return get_data_dir("common", dataset)


def get_conv_data_dir(dataset):
    return get_data_dir("conv", dataset)


def get_rec_data_dir(dataset):
    return get_data_dir("rec", dataset)


def get_root_data_dir(dataset):
    return get_common_data_dir(dataset)


def require_data_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return path
