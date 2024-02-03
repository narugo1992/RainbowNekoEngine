from typing import Dict, Any, List, Union
import torch

from PIL import Image
from .base import BasicAction, feedback_input, MemoryMixin
from rainbowneko.ckpt_manager import auto_manager

class LoadImageAction(BasicAction):
    def __init__(self, image_paths:Union[str, List[str]], image_transforms=None):
        super().__init__()
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self.image_paths = image_paths
        self.image_transforms = image_transforms

    def load_one(self, path:str):
        img = Image.open(path).convert('RGB')
        if self.image_transforms:
            img = self.image_transforms(img)
        return img

    @feedback_input
    def forward(self, device, **states):
        input = torch.stack([self.load_one(path) for path in self.image_paths]).to(device)
        input: Dict[str, Any] = {'x':input}
        return {'input':input}

class LoadModelAction(BasicAction, MemoryMixin):
    def __init__(self, part_paths: Union[str, List[str]]):
        super().__init__()
        if isinstance(part_paths, str):
            part_paths = [part_paths]
        self.part_paths=part_paths

    @feedback_input
    def forward(self, memory, **states):
        for path in self.part_paths:
            sd = auto_manager(path).load(path)
            memory.model.load_state_dict(sd['base'])