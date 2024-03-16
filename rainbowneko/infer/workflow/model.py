from typing import Dict, Any, Callable

import torch

from .base import BasicAction, feedback_input, MemoryMixin


class PrepareAction(BasicAction):
    def __init__(self, device='cuda', dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype

    @feedback_input
    def forward(self, **states):
        return {'device': self.device, 'dtype': self.dtype}


class BuildModelAction(BasicAction, MemoryMixin):
    def __init__(self, model_builder: Callable):
        super().__init__()
        self.model_builder = model_builder

    @feedback_input
    def forward(self, memory, device, **states):
        memory.model = self.model_builder().to(device)


class ForwardAction(BasicAction, MemoryMixin):

    @feedback_input
    def forward(self, input: Dict[str, Any], memory, **states):
        with torch.inference_mode():
            memory.model.eval()
            output: Dict[str, Any] = memory.model(**input)
        return {'output': output}


class VisPredAction(BasicAction, MemoryMixin):
    @feedback_input
    def forward(self, output: Dict[str, Any], memory, **states):
        pred = output['pred']
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()

        memory.logits = pred
        return {'logits': pred}
