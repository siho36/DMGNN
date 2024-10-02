import os

from datasets import CoraDataset


def get_datasets(name, root="data/"):
    """
    Get preloaded datasets by name
    :param name: name of the dataset
    :param root: root path of the dataset
    :return: train_dataset, test_dataset, val_dataset
    """
    if name == "Cora":
        folder = os.path.join(root)
        clean_test_dataset = CoraDataset(folder, mode="clean_testing", name=name)
        atk_test_dataset = CoraDataset(folder, mode="atk_testing", name=name)
        val_dataset = CoraDataset(folder, mode="evaluating", name=name)
        train_dataset = CoraDataset(folder, mode="training", name=name)
    else:
        raise ValueError
    return train_dataset, val_dataset, clean_test_dataset, atk_test_dataset
