import functools
import math
import os
from abc import ABC, ABCMeta, abstractmethod
from typing import Tuple, List, Dict, Union, Callable

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ModelMixin(metaclass=ABCMeta):
    model_name: str = ""
    _checked_inputs: bool = True
    _url_map: Dict[str, str]

    def __init__(self, config, **kwargs):
        ...

    def update_config(self, config):
        if config is not None:
            for k, v in config.to_dict().items():
                setattr(self, k, v)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def partial_apply(self, func: Callable, *args, **kwargs) -> List[Tensor]:
        """Partial apply

        Adapted from:
            https://github.com/open-mmlab/mmdetection
        Args:
            func (:obj:`Callable`):
        """
        func = functools.partial(func, **kwargs) if kwargs else func
        results = map(func, *args)

        return map(list, zip(*results))

    @abstractmethod
    def forward(self, inputs) -> None:
        ...

    @classmethod
    def resize_inputs(
        cls,
        inputs: Tensor,
        size: Tuple[int, int],
        mode: str = "nearest",
        preserve_aspect_ratio: bool = False,
    ) -> Tensor:
        """
        Args:
            inputs (:obj:`Tensor`):
            size ()
            mode ()
            preserve_aspect_ratio ()
        Returns:
            images
            image_sizes
        """
        # TODO: modified size
        image_sizes = [tuple(tensor.shape[-2:]) for tensor in inputs]  # H, W

        if preserve_aspect_ratio:
            print("preserve_aspect_ratio")
            images = []
            for tensor, image_size in zip(inputs, image_sizes):
                print(image_size)
                tensor = _resize_image(tensor, size[0], size[1], image_size)
                images.append(tensor)

            resized_sizes = [tuple(img.shape[-2:]) for img in images]
            images = _batch_images(images)

            return images, image_sizes, resized_sizes
        else:
            images = []
            for tensor in inputs:
                # print('??', tensor.size(), tensor.unsqueeze(0).size())
                tensor = F.interpolate(tensor.unsqueeze(0), size=size, mode=mode)
                # tensor = F.interpolate(tensor[None], size=size, mode=mode)[0]
                # print(tensor.size(), tensor.squeeze(0).size())
                images.append(tensor.squeeze(0))
            # inputs = torch.cat([F.interpolate(tensor, size=size, mode=mode) for tensor in inputs])

            images = torch.stack(images, dim=0)

            return images, image_sizes


def _resize_image(
    image: Tensor, min_size: int, max_size: int, image_size: Tuple[int, int]
) -> None:
    _min_size = float(min(image_size))
    _max_size = float(max(image_size))
    scale_factor = min(float(min_size) / _min_size, float(max_size) / _max_size)

    image = F.interpolate(
        image[None],
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )[0]

    return image


def max_by_axis(the_list: List[List[int]]) -> List[int]:
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def _batch_images(images: List[Tensor], size_divisible: int = 32) -> Tensor:
    max_size = max_by_axis([list(img.shape) for img in images])
    print(max_size)
    stride = float(size_divisible)
    max_size = list(max_size)
    max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
    max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

    batch_shape = [len(images)] + max_size
    batched_imgs = images[0].new_full(batch_shape, 0)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


class Backbone(nn.Module, ModelMixin):
    backbone_name: str = ""

    def __init__(self):
        super().__init__()
        self.channels: List[int] = []

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        ...

    @abstractmethod
    def from_pretrained(self, name_or_path, **kwargs):
        ...

    def init_weights(self):
        ...

    @torch.jit.unused
    def eager_outputs(self, *args):
        raise NotImplementedError

    def _from_state_dict(self, *args):
        raise NotImplementedError


class Neck(nn.Module, ModelMixin):
    neck_type: str = ""

    def __init__(self):
        super().__init__()
        self.channels: List[int] = []

    def forward(self, inputs: List[Tensor]) -> List[Tensor]:
        ...

    def eager_outputs(self, *args):
        ...


class Head(nn.Module, ModelMixin):
    def __init__(self):
        super().__init__()
        self.channels: List[int] = []

    def forward(self, inputs: Tensor) -> Dict[str, Tensor]:
        ...


class Model(nn.Module, ModelMixin):
    config_class = None
    base_model_prefix: str = ""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config
        self.name_or_path = ""

    @property
    def base_model(self) -> nn.Module:
        return getattr(self, self.base_model_prefix, self)

    def freeze(self, enable: bool = False):
        """Freeze Batch Nomalization

        Adapted from:
            https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8
        """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    @classmethod
    def get_pretrained_from_file(cls, name_or_path, **kwargs):
        cache_dir = kwargs.get("cache_dir", "cache")

        pretrained_file = os.path.join(
            cache_dir, cls.base_model_prefix, f"{name_or_path}.pth"
        )
        if os.path.isfile(pretrained_file):
            return pretrained_file
        else:
            from urllib import request

            from .file_utils import reporthook

            url = "https://unerue.synology.me/boda/models/"

            # print(f'Downloading {name_or_path}.{extension}...', end=' ')
            request.urlretrieve(
                f"{url}{cls.base_model_prefix}/{name_or_path}.pth",
                pretrained_file,
                reporthook,
            )
            print()
            # pretrained_file = os.path.join(
            #     cache_dir, cls.base_model_prefix, f'{name_or_path}.pth')

            return pretrained_file

    @classmethod
    @abstractmethod
    def from_pretrained(cls, name_or_path: Union[str, os.PathLike], **kwargs) -> None:
        """Create from pretrained model weights"""
        ...

    def load_weights(self, path):
        raise NotImplementedError

    # @classmethod
    # def from_pretrained(
    #     cls,
    #     name_or_path: Union[str, os.PathLike]
    # ) -> None:
    #     if os.path.isfile(name_or_path):
    #         state_dict = torch.load(name_or_path)
    #         return state_dict
    #     else:
    #         cls._url_map[name_or_path]
    #         pass

    # def _check_pretrained_model_is_valid(self, model_name_or_path):
    #     # if model_name_or_path not in
    #     raise NotImplementedError

    # @classmethod
    # def get_config_dict(cls, model_name_or_path, **kwargs):
    #     raise NotImplementedError
    @classmethod
    def check_inputs(cls, inputs: List[Tensor]) -> List[Tensor]:
        """
        Args:
            inputs (List[Tensor]): Size([C, H, W])

        Return:
            outputs (Tensor): Size([B, C, H, W])
        """
        if cls._checked_inputs:
            for image in inputs:
                if isinstance(image, Tensor):
                    if image.dim() != 3:
                        raise ValueError(
                            f"images is expected to be 3d tensors of shape [C, H, W] {image.size()}"
                        )
                else:
                    raise ValueError("Expected image to be Tensor.")
            cls._checked_inputs = False

        # if isinstance(inputs, list):
        #     inputs = torch.stack(inputs)

        return inputs

    def __repr__(self):
        return_str = str(self.config.model_name) + "\n"
        for k, v in self.config.to_dict().items():
            if k not in ["label_map"] and v is not None:
                return_str += f"{k}: {v}\n"

        return return_str


class Matcher(ABC):
    def __init__(self):
        pass

    def encode(self):
        pass

    def deconde(self):
        pass


class LossFunction(nn.Module):
    _checked_targets = True

    def __init__(self, **kwargs) -> None:
        super().__init__()
        # self.config = config

    def forward(
        self, inputs: Dict[str, Tensor], targets: List[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    @classmethod
    def copy_targets(cls, targets: List[Dict[str, Tensor]]) -> List[Dict[str, Tensor]]:
        if targets is not None:
            targets_copy: List[Dict[str, Tensor]] = []
            for target in targets:
                _target: Dict[str, Tensor] = {}
                for key, value in target.items():
                    _target[key] = value
                targets_copy.append(_target)
            targets = targets_copy

        return targets

    @classmethod
    def check_targets(cls, targets: List[Dict[str, Tensor]]) -> None:
        if cls._checked_targets:
            for target in targets:
                if isinstance(target["boxes"], Tensor):
                    boxes = target["boxes"]
                    check_boxes = boxes[:, :2] >= boxes[:, 2:]
                    if boxes.dim() != 2 or boxes.size(1) != 4:
                        raise ValueError(
                            "Expected target boxes to be a tensor of [N, 4]."
                        )
                    elif check_boxes.any():
                        raise ValueError(f"{boxes}")
                    elif target["labels"].dim() != 1:
                        raise ValueError("Expected target boxes to be a tensor of [N].")
                else:
                    raise ValueError("Expected target boxes to be Tensor.")
            cls._checked_targets = False

    def decode(self, targets: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        for target in targets:
            pass


class PostProcess:
    def __init__(self, num_classes, nms, nms_threshold, score_threshold, top_k) -> None:
        self.num_classes = num_classes
        self.top_k = top_k
        self.nms_threshold = 0.5
        self.score_threshold = 0.2
        self.nms = nms

        pass

    def __call__(self, preds: Dict[str, Tensor], **kwargs) -> None:
        if "scores" in preds:
            scores = preds["scores"]
            batch_size = scores.size(0)

        if "prior_boxes" in preds:
            prior_boxes = preds["prior_boxes"]

        if "boxes" in preds:
            boxes = preds["boxes"]

        if "masks" in preds:
            masks = preds["masks"]

    def convert_boxes(self):
        pass

    def convert_scores(self):
        pass

    def convert_masks(self):
        pass


#  "backbone.layers.2.6.conv1.0.weight", "backbone.layers.2.6.bn1.weight",
# class Register(ABCMeta):
#     registry = {}
#     def __new__(cls):
#         new_cls = type.__new__(cls, name, bases, attrs)
#         if not hasattr(new_cls, '_registry_name'):
#             raise Exception('Ay class')

#         cls.register[new_cls._registry_name] = new_cls
#         return ABCMeta.__new__(cls, name, bases, attrs)

#     @classmethod
#     def get_registry(cls):
#         return dict(cls.registry)

# @classmethod
# def __subclasshook__(cls, subclass):
#     return hasattr(subclass, 'from_pretrained') or NotImplementedError


# registry = []

# def register(func):
#     print(f'running register {func}')
#     registry.append(func.__class__.__name__)
#     return func
