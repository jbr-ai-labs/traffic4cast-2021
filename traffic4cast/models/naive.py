import torch
import torch.nn as nn


class NaiveRepeatLast(nn.Module):

    def __init__(self):
        super(NaiveRepeatLast, self).__init__()
        self.coef = nn.parameter.Parameter(torch.FloatTensor([1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frame = x[:, 88:, :, :]
        output = frame.repeat(1, 6, 1, 1)
        return self.coef * output
