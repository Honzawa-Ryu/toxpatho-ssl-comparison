# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

utilities from lightly
(https://github.com/lightly-ai/lightly)

@author: Katsuhisa, tadahaya
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Sequence

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from PIL.Image import Image
from PIL import ImageOps, ImageFilter

@dataclass
class Location:
    # The row index of the top-left corner of the crop.
    top: float
    # The column index of the top-left corner of the crop.
    left: float
    # The height of the crop.
    height: float
    # The width of the crop.
    width: float
    # The height of the original image.
    image_height: float
    # The width of the original image.
    image_width: float
    # Whether to flip the image horizontally.
    horizontal_flip: bool = False
    # Whether to flip the image vertically.
    vertical_flip: bool = False

class RandomVerticalFlipWithLocation(transforms.RandomVerticalFlip):  # type: ignore[misc] # Class cannot subclass "RandomVerticalFlip" (has type "Any")
    """See base class."""

    def forward(
        self, img, location
    ):
        """Vertical flip image.

        Vertically flip the given image randomly with a given probability and
        return both the resulting image and the location.

        Args:
            img (PIL Image or Tensor): Image to be flipped..
            Location: Location object linked to the image
        Returns:
            PIL Image or Tensor: Randomly flipped image
            Location: Location object with updated location.vertical_flip parameter
        """

        if torch.rand(1) < self.p:
            img = transforms.functional.vflip(img)
            location.vertical_flip = True
        return img, location

class RandomHorizontalFlipWithLocation(transforms.RandomHorizontalFlip):  # type: ignore[misc] # Class cannot subclass "RandomHorizontalFlip" (has type "Any")
    """See base class."""

    def forward(
        self, img, location
    ):
        """Horizontal flip image.

        Horizontally flip the given image randomly with a given probability and
        return both the resulting image and the location.

        Args:
            img (PIL Image or Tensor): Image to be flipped..
            Location: Location object linked to the image
        Returns:
            PIL Image or Tensor: Randomly flipped image
            Location: Location object with updated location.horizontal_flip parameter
        """

        if torch.rand(1) < self.p:
            img = transforms.functional.hflip(img)
            location.horizontal_flip = True
        return img, location

class RandomResizedCropWithLocation(transforms.RandomResizedCrop):  # type: ignore[misc] # Class cannot subclass "RandomResizedCrop" (has type "Any")
    """
    Do a random resized crop and return both the resulting image and the location. See base class.

    """

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Randomly cropped image
            Location: Location object containing crop parameters

        """
        top, left, height, width = self.get_params(img, self.scale, self.ratio)
        image_width, image_height = transforms.functional.get_image_size(img)
        location = Location(
            top=top,
            left=left,
            height=height,
            width=width,
            image_height=image_height,
            image_width=image_width,
        )
        img = transforms.functional.resized_crop(
            img, top, left, height, width, self.size, self.interpolation
        )
        return img, location

class RandomResizedCropAndFlip(nn.Module):
    """Randomly flip and crop an image.

    A PyTorch module that applies random cropping, horizontal and vertical flipping to an image,
    and returns the transformed image and a grid tensor used to map the image back to the
    original image space in an NxN grid.

    Args:
        grid_size:
            The number of grid cells in the output grid tensor.
        crop_size:
            The size (in pixels) of the random crops.
        crop_min_scale:
            The minimum scale factor for random resized crops.
        crop_max_scale:
            The maximum scale factor for random resized crops.
        hf_prob:
            The probability of applying horizontal flipping to the image.
        normalize:
            A dictionary containing the mean and std values for normalizing the image.
    """

    def __init__(
        self,
        grid_size: int = 7,
        crop_size: int = 224,
        crop_min_scale: float = 0.05,
        crop_max_scale: float = 0.2,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.crop_size = crop_size
        self.crop_min_scale = crop_min_scale
        self.crop_max_scale = crop_max_scale
        self.hf_prob = hf_prob
        self.vf_prob = vf_prob
        self.resized_crop = RandomResizedCropWithLocation(
            size=self.crop_size, scale=(self.crop_min_scale, self.crop_max_scale)
        )
        self.horizontal_flip = RandomHorizontalFlipWithLocation(self.hf_prob)
        self.vertical_flip = RandomVerticalFlipWithLocation(self.vf_prob)

    def forward(self, img):
        """Applies random cropping and horizontal flipping to an image, and returns the
        transformed image and a grid tensor used to map the image back to the original image
        space in an NxN grid.

        Args:
            img: The input PIL image.

        Returns:
            A tuple containing the transformed PIL image and the grid tensor.
        """

        img, location = self.resized_crop.forward(img=img)
        img, location = self.horizontal_flip.forward(img, location)
        img, location = self.vertical_flip.forward(img, location)
        grid = self.location_to_NxN_grid(location=location)

        return img, grid

    def location_to_NxN_grid(self, location: Location) -> torch.Tensor:
        """Create grid from location object.

        Create a grid tensor with grid_size rows and grid_size columns, where each cell represents a region of
        the original image. The grid is used to map the cropped and transformed image back to the
        original image space.

        Args:
            location: An instance of the Location class, containing the location and size of the
                transformed image in the original image space.

        Returns:
            A grid tensor of shape (grid_size, grid_size, 2), where the last dimension represents the (x, y) coordinate
            of the center of each cell in the original image space.
        """

        cell_width = location.width / self.grid_size
        cell_height = location.height / self.grid_size
        x = torch.linspace(
            location.left, location.left + location.width, self.grid_size
        ) + (cell_width / 2)
        y = torch.linspace(
            location.top, location.top + location.height, self.grid_size
        ) + (cell_height / 2)
        if location.horizontal_flip:
            x = torch.flip(x, dims=[0])
        if location.vertical_flip:
            y = torch.flip(y, dims=[0])
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        return torch.stack([grid_x, grid_y], dim=-1)

class RandomRotate(object):
    """Implementation of random rotation.
    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.
    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.
    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.
    
    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.
        Args:
            sample:
                PIL image which will be rotated.
        
        Returns:
            Rotated image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            sample =  transforms.functional.rotate(sample, self.angle)
        return sample

class ImageGridTransform:
    """Transforms an image into multiple views and grids.

    Used for VICRegL.

    Attributes:
        transforms:
            A sequence of (image_grid_transform, view_transform) tuples.
            The image_grid_transform creates a new view and grid from the image.
            The view_transform further augments the view. Every transform tuple
            is applied once to the image, creating len(transforms) views and
            grids.
    """

    def __init__(self, transform_seq: Sequence[transforms.Compose]):
        self.transform_seq = transform_seq

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        """Transforms an image into multiple views.

        Every transform in self.transforms creates a new view.

        Args:
            image:
                Image to be transformed into multiple views and grids.

        Returns:
            List of views and grids tensors or PIL images. In the VICRegL implementation
            it has size:
            [
                [3, global_crop_size, global_crop_size],
                [3, local_crop_size, local_crop_size],
                [global_grid_size, global_grid_size, 2],
                [local_grid_size, local_grid_size, 2]
            ]

        """
        views, grids = [], []
        for image_grid_transform, view_transform in self.transform_seq:
            view, grid = image_grid_transform(image)
            views.append(view_transform(view))
            grids.append(grid)
        views += grids
        return views

class GaussianBlur:
    """Implementation of random Gaussian blur.

    Utilizes the built-in ImageFilter method from PIL to apply a Gaussian
    blur to the input image with a certain probability. The blur is further
    randomized by sampling uniformly the values of the standard deviation of
    the Gaussian kernel.

    Attributes:
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        prob:
            Probability with which the blur is applied.
        scale:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.

    """

    def __init__(
        self,
        kernel_size: Union[float, None] = None,
        prob: float = 0.5,
        scale: Union[float, None] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
    ):
        if scale != None or kernel_size != None:
            warn(
                "The 'kernel_size' and 'scale' arguments of the GaussianBlur augmentation will be deprecated.  "
                "Please use the 'sigmas' parameter instead.",
                DeprecationWarning,
            )
        self.prob = prob
        self.sigmas = sigmas

    def __call__(self, sample):
        """Blurs the image with a given probability.

        Args:
            sample:
                PIL image to which blur will be applied.

        Returns:
            Blurred image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # choose randomized std for Gaussian filtering
            sigma = np.random.uniform(self.sigmas[0], self.sigmas[1])
            # PIL GaussianBlur https://github.com/python-pillow/Pillow/blob/76478c6865c78af10bf48868345db2af92f86166/src/PIL/ImageFilter.py#L154 label the
            # sigma parameter of the gaussian filter as radius. Before, the radius of the patch was passed as the argument.
            # The issue was addressed here https://github.com/lightly-ai/lightly/issues/1051 and solved by AurelienGauffre.
            return sample.filter(ImageFilter.GaussianBlur(radius=sigma))
        # return original image
        return sample

class RandomSolarization(object):
    """Implementation of random image Solarization.

    Utilizes the integrated image operation `solarize` from Pillow. Solarization
    inverts all pixel values above a threshold (default: 128).

    Attributes:
        probability:
            Probability to apply the transformation
        threshold:
            Threshold for solarization.
    """

    def __init__(self, prob: float = 0.5, threshold: int = 128):
        self.prob = prob
        self.threshold = threshold

    def __call__(self, sample):
        """Solarizes the given input image

        Args:
            sample:
                PIL image to which solarize will be applied.

        Returns:
            Solarized image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            # return solarized image
            return ImageOps.solarize(sample, threshold=self.threshold)
        # return original image
        return sample