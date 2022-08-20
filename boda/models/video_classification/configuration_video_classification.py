from typing import Tuple, List
from ...base_configuration import BaseConfig


class VideoClassificationConfig(BaseConfig):
    """Configuration for Efficient-3DCNN

    Arguments:
        num_classes (:obj:`int`):

        max_size ():
        resize (:resize_h:resize_w):
    """
    def __init__(
        self,
        num_classes: int = 400,
        # resize: List[int, int] = [320, 240],
        cropped_input_size: int = 224,
        # temporal_duration: int = 16,
        # temporal_term: int = 8,
        model_name: str = 'resnet3d_50',
        **kwargs
    ) -> None:

        super().__init__(min_size=cropped_input_size, **kwargs)
        self.num_classes = num_classes
        # self.resize = resize
        self.crop_size = cropped_input_size
        # self.temporal_duration = temporal_duration
        # self.temporal_term = temporal_term
        self.model_name = model_name


