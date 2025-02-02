"""
Dummy Dataset for Training
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import os
import shutil
from contextlib import ContextDecorator
from functools import wraps
from pathlib import Path
from tempfile import mkdtemp
from typing import Dict, List, Optional, Union

import numpy as np
from skimage.io import imsave

from .shapes import random_shapes


def generate_random_anomaly_image(
    image_width: int,
    image_height: int,
    shapes: List[str] = ["triangle", "rectangle"],
    max_shapes: int = 5,
    generate_mask: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate a random image with the corresponding shape entities.

    Args:
        image_width (int): Width of the image
        image_height (int): Height of the image
        shapes (List[str]): List of shapes to draw in the image. Make sure these are different from `anomalous_shapes`
        max_shapes (int): Maximum shapes of a kind in the image. Defaults to 5.
        max_size (int): Maximum size of the test shapes. Defaults to 10.
        generate_mask (bool): Toggle between train/test split. Train images are restricted to first quadrant.
                    Also generates the mask for the image. Defaults to False.

    Returns:
        Dict[str, np.ndarray]: image if `train` is False. Else return image, mask
    """

    image: np.ndarray = np.full((image_height, image_width, 3), 255, dtype=np.uint8)
    input_region = [0, 0, image_height, image_width]

    for shape in shapes:
        shape_image = random_shapes(input_region, (image_height, image_width), max_shapes=max_shapes, shape=shape)
        image = np.minimum(image, shape_image)  # since white is 255

    result = {"image": image}

    if generate_mask:
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        # if color exists in any channel turn the mask bit white.
        # not sure if there is a better way to do this.
        mask[image[..., 0] < 255] = 255
        mask[image[..., 1] < 255] = 255
        mask[image[..., 2] < 255] = 255
        result["mask"] = mask

    return result


class TestDataset:
    def __init__(
        self,
        num_train: int = 1000,
        num_test: int = 100,
        img_height: int = 128,
        img_width: int = 128,
        max_size: int = 10,
        train_shapes: List[str] = ["triangle", "rectangle"],
        test_shapes: List[str] = ["hexagon", "star"],
        dataset_path: Union[str, Path] = "./datasets/MVTec",
        use_mvtec: bool = False,
    ) -> None:
        """Creates a context for Generating Dummy Dataset. Useful for wrapping test functions.
        NOTE: for MVTec dataset it does not return a category.
        It is adviced to use a default parameter in the function

        Args:
            num_train (int, optional): Number of training images to generate. Defaults to 1000.
            num_test (int, optional): Number of testing images to generate per category. Defaults to 100.
            img_height (int, optional): Height of the image. Defaults to 128.
            img_width (int, optional): Width of the image. Defaults to 128.
            max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
            train_shapes (List[str], optional): List of good shapes. Defaults to ["circle", "rectangle"].
            test_shapes (List[str], optional): List of anomalous shapes. Defaults to ["triangle", "ellipse"].
            path (Union[str, Path], optional): Path to MVTec dataset. Defaults to "./datasets/MVTec".
            use_mvtec (bool, optional): Use MVTec dataset or dummy dataset. Defaults to False.

        Example:
            >>> @TestDataset
            >>> def test_some_function(path, category="leather"):
            >>>     do something
        """
        self.num_train = num_train
        self.num_test = num_test
        self.img_height = img_height
        self.img_width = img_width
        self.max_size = max_size
        self.train_shapes = train_shapes
        self.test_shapes = test_shapes
        self.dataset_path = dataset_path
        self.use_mvtec = use_mvtec

    def __call__(self, func):
        @wraps(func)
        def inner(*args, **kwds):
            # If true, will use MVTech dataset for testing.
            # Useful for nightly builds
            if self.use_mvtec:
                return func(*args, dataset_path=self.dataset_path, **kwds)
            else:
                with GeneratedDummyDataset(
                    num_train=self.num_train,
                    num_test=self.num_test,
                    img_height=self.img_height,
                    img_width=self.img_width,
                    train_shapes=self.train_shapes,
                    test_shapes=self.test_shapes,
                    max_size=self.max_size,
                ) as dataset_path:
                    kwds["category"] = "shapes"
                    return func(*args, dataset_path=dataset_path, **kwds)

        return inner


class GeneratedDummyDataset(ContextDecorator):
    """Context for generating dummy shapes dataset.

    Example:
        >>> with GeneratedDummyDataset(num_train=1000,num_test=100) as dataset_path:
        >>>     some_function()

    Args:
        num_train (int, optional): Number of training images to generate. Defaults to 1000.
        num_test (int, optional): Number of testing images to generate per category. Defaults to 100.
        img_height (int, optional): Height of the image. Defaults to 128.
        img_width (int, optional): Width of the image. Defaults to 128.
        max_size (Optional[int], optional): Maximum size of the test shapes. Defaults to 10.
        train_shapes (List[str], optional): List of good shapes. Defaults to ["circle", "rectangle"].
        test_shapes (List[str], optional): List of anomalous shapes. Defaults to ["triangle", "ellipse"].
    """

    def __init__(
        self,
        num_train: int = 1000,
        num_test: int = 100,
        img_height: int = 256,
        img_width: int = 256,
        max_size: Optional[int] = 10,
        train_shapes: List[str] = ["triangle", "rectangle"],
        test_shapes: List[str] = ["star", "hexagon"],
    ) -> None:
        self.root_dir = mkdtemp()
        self.num_train = num_train
        self.num_test = num_test
        self.train_shapes = train_shapes
        self.test_shapes = test_shapes
        self.image_height = img_height
        self.image_width = img_width
        self.max_size = max_size

    def _generate_dataset(self):
        """Generates dummy dataset in a temporary directory using the same convention as MVTec"""
        # create train images
        train_path = os.path.join(self.root_dir, "shapes", "train", "good")
        os.makedirs(train_path, exist_ok=True)
        for i in range(self.num_train):
            result = generate_random_anomaly_image(
                image_width=self.image_width,
                image_height=self.image_height,
                shapes=self.train_shapes,
                generate_mask=False,
            )
            image = result["image"]
            imsave(os.path.join(train_path, f"{i:03}.png"), image, check_contrast=False)

        # create test images
        for test_category in self.test_shapes:
            test_path = os.path.join(self.root_dir, "shapes", "test", test_category)
            mask_path = os.path.join(self.root_dir, "shapes", "ground_truth", test_category)
            os.makedirs(test_path, exist_ok=True)
            os.makedirs(mask_path, exist_ok=True)
            # anomaly and masks. The idea is to superimpose anomalous shapes on top of correct ones
            for i in range(self.num_test):
                correct_shapes = generate_random_anomaly_image(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    shapes=self.train_shapes,
                    generate_mask=False,
                )
                result = generate_random_anomaly_image(
                    image_width=self.image_width,
                    image_height=self.image_height,
                    shapes=[test_category],
                    generate_mask=True,
                )
                correct_shapes = correct_shapes["image"]
                image, mask = result["image"], result["mask"]
                image = np.minimum(image, correct_shapes)  # since 255 is white
                imsave(os.path.join(test_path, f"{i:03}.png"), image, check_contrast=False)
                imsave(os.path.join(mask_path, f"{i:03}_mask.png"), mask, check_contrast=False)
        # good test
        test_good = os.path.join(self.root_dir, "shapes", "test", "good")
        os.makedirs(test_good, exist_ok=True)
        for i in range(self.num_test):
            result = generate_random_anomaly_image(
                image_width=self.image_width,
                image_height=self.image_height,
                shapes=self.train_shapes,
            )
            image = result["image"]
            imsave(os.path.join(test_good, f"{i:03}.png"), image, check_contrast=False)

    def __enter__(self):
        """Creates the dataset in temp folder"""
        self._generate_dataset()
        return self.root_dir

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        """Cleanup the directory"""
        shutil.rmtree(self.root_dir)
