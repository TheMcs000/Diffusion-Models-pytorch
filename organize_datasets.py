from pathlib import Path
from random import shuffle

dataset_dir = Path("/data_b/8behrens/datasets")
wanted_datasets = ["stanford-dogs", "cats"]

destination_dir = Path("/data_b/8behrens/datasets-ordered")

train_test_split = 0.7

for dataset_name in wanted_datasets:
    dataset_path = Path(dataset_dir, dataset_name)
    all_images = list(dataset_path.glob("**/*.jpg"))
    shuffle(all_images)  # in-place

    dataset_dest_dir = Path(destination_dir, dataset_name)
    dataset_dest_dir.mkdir(parents=True, exist_ok=False)

    split_index = int(train_test_split * len(all_images))

    # train
    train_dir = Path(dataset_dest_dir, "train")
    train_dir.mkdir(parents=False, exist_ok=False)
    for img_path in all_images[:split_index]:
        img_path.rename(Path(train_dir, img_path.name))
    
    # test
    test_dir = Path(dataset_dest_dir, "test")
    test_dir.mkdir(parents=False, exist_ok=False)
    for img_path in all_images[split_index:]:
        img_path.rename(Path(test_dir, img_path.name))
