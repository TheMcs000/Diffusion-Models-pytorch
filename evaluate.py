import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats


@dataclass
class Args:
    paths: Tuple[str, str]
    num_workers: Optional[int] = None
    device: Optional[str] = "cuda"
    save_stats: bool = False
    batch_size: int = 50
    dims: int = 2048
    

def fid_score_main(path1: Union[str, Path], path2: Union[str, Path]) -> float:
    """Copied from pytorch_fid.fid_score.main"""
    args = Args((str(path1), str(path2)))
    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    if args.num_workers is None:
        try:
            num_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            # os.sched_getaffinity is not available under Windows, use
            # os.cpu_count instead (which may not return the *available* number
            # of CPUs).
            num_cpus = os.cpu_count()

        num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    else:
        num_workers = args.num_workers

    if args.save_stats:
        save_fid_stats(args.paths, args.batch_size, device, args.dims, num_workers)
        return

    fid_value = calculate_fid_given_paths(args.paths,
                                          args.batch_size,
                                          device,
                                          args.dims,
                                          num_workers)
    return fid_value


CIFAR_PATH = Path("/srv/8behrens/cifar/cifar10-64/test")

DATA_PATH = Path("data")
CROSS_IMAGES_PATH = Path(DATA_PATH, "cross_images")
SAMPLED_IMAGES_PATH = Path(DATA_PATH, "sampled_image")

NUM_CROSS_IMAGES = 11
NUM_SAMPLED_IMAGES = 10

CLASS_NAME_TO_ID = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9,
}

CrossImageCombination = Tuple[str, str]

CROSS_IMAGE_COMBINATIONS_TO_ID: Dict[CrossImageCombination, int] = {
    ("automobile", "truck"): 0,
    ("horse", "ship"): 1,
    ("cat", "dog"): 2,
    ("airplane", "bird"): 3,
    ("deer", "frog"): 4,
    ("ship", "truck"): 5,
    ("automobile", "bird"): 6,
    ("ship", "bird"): 7,
    ("deer", "horse"): 8,
    ("truck", "frog"): 9,
    ("dog", "automobile"): 10,
}


def evaluate_sampled_images():
    all_results = {}
    for class_name in CLASS_NAME_TO_ID:
        path1 = Path(CIFAR_PATH, f"class{CLASS_NAME_TO_ID[class_name]}")
        path2 = Path(SAMPLED_IMAGES_PATH, str(CLASS_NAME_TO_ID[class_name]))
        fid_score = fid_score_main(path1, path2)
        # print(f"Class {class_name}, from {path1=} and {path2=}: {fid_score}")
        all_results[class_name] = fid_score
    with open("data/eval/sampled_images.json", "w+") as f:
        json.dump(all_results, f)


def evaluate_cross_images():
    all_results: Dict[str, Dict] = {}
    for cross_image_combination in CROSS_IMAGE_COMBINATIONS_TO_ID:
        try:
            class1, class2 = cross_image_combination

            diff_model_cross_path = Path(CROSS_IMAGES_PATH, str(CROSS_IMAGE_COMBINATIONS_TO_ID[cross_image_combination]))
            cifar_cross_path = Path(CIFAR_PATH, f"cross-{CLASS_NAME_TO_ID[class1]}_{CLASS_NAME_TO_ID[class2]}")
            cifar_class1_path = Path(CIFAR_PATH, f"class{CLASS_NAME_TO_ID[class1]}")
            cifar_class2_path = Path(CIFAR_PATH, f"class{CLASS_NAME_TO_ID[class2]}")
            diff_model_class1_path = Path(SAMPLED_IMAGES_PATH, str(CLASS_NAME_TO_ID[class1]))
            diff_model_class2_path = Path(SAMPLED_IMAGES_PATH, str(CLASS_NAME_TO_ID[class2]))

            this_results = {}

            # diffusion model  - cifar10
            # (class1, class2) - (class1, class2); Wie sehr ähneln unsere cross images den cifar10 cross images
            this_results[f"(class1, class2) - (class1, class2)"] = fid_score_main(diff_model_cross_path, cifar_cross_path)

            # (class1, class2) - class1; Wie sehr ähneln unsere cross images den cifar10 class1 images
            this_results[f"(class1, class2) - class1"] = fid_score_main(diff_model_cross_path, cifar_class1_path)
            # (class1, class2) - class2; Wie sehr ähneln unsere cross images den cifar10 class2 images
            this_results[f"(class1, class2) - class2"] = fid_score_main(diff_model_cross_path, cifar_class2_path)


            # class1 - (class1, class2); Wie sehr ähneln unsere class1 images den cifar10 cross images
            this_results[f"class1 - (class1, class2)"] = fid_score_main(diff_model_class1_path, cifar_cross_path)
            # class2 - (class1, class2); Wie sehr ähneln unsere class2 images den cifar10 cross images
            this_results[f"class2 - (class1, class2)"] = fid_score_main(diff_model_class2_path, cifar_cross_path)


            # class1           - class2; Wie sehr ähneln unsere class1 images den cifar10 class2 images
            this_results[f"class1 - class2"] = fid_score_main(diff_model_class1_path, cifar_class2_path)
            # class2           - class1; Wie sehr ähneln unsere class1 images den cifar10 class2 images
            this_results[f"class2 - class1"] = fid_score_main(diff_model_class2_path, cifar_class1_path)
            
            all_results[str(cross_image_combination)] = this_results
        except Exception as e:
            print(f"Error during {cross_image_combination}:", e)
            print("Skip.")
            continue

    print(repr(all_results))
    with open("data/eval/cross_images.json", "w+") as f:
        json.dump(all_results, f)


def main() -> None:
    # evaluate_sampled_images()
    evaluate_cross_images()

if __name__ == '__main__':
    main()

# 1.
# alle dogs mit dogs, cars mit cars, usw. -> 10

# 2. jeweils mit dem combi folder und jeweils einzeln (4*11=33)
# z.b. (car, truck): (car, truck) - (car, truck), (car, truck) - car, (car, truck) - truck, car - truck, truck - car
# [1, 9],  # Car, Truck
# [7, 8],  # Horse, Ship
# [3, 5],  # cat, dog
# [0, 2],  # airplane, bird
# [4, 6],  # deer, frog
# [8, 9],  # ship, truck
# [1, 2],  # car, bird
# [8, 2],  # ship, bird
# [4, 7],  # deer, horse
# [9, 6],  # truck, frog
# [5, 1],  # dog, car