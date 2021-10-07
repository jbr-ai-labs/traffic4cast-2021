import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

from traffic4cast.models.baseline_unet import UNet


class ReverseLayerF(Function):
    """The gradient reversal layer (GRL)
    This is defined in the DANN paper http://jmlr.org/papers/volume17/15-239/15-239.pdf
    Forward pass: identity transformation.
    Backward propagation: flip the sign of the gradient.
    From https://github.com/criteo-research/pytorch-ada/blob/master/adalib/ada/models/layers.py
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class DomainAdaptationModel(nn.Module):

    def __init__(self, model: nn.Module, emb_dim: int):
        super(DomainAdaptationModel, self).__init__()
        self.model = model
        self.domain_classifier = nn.Sequential(
            nn.Linear(emb_dim, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
        )

    def forward(self, x, *args, **kwargs):
        """ Forward on source domain"""
        return self.model(x, args, kwargs)

    def predict_domain(self, x, *args, **kwargs):
        """ Forward on target domain """
        features = self.model.extract_features(x, args, kwargs)
        features = F.avg_pool2d(features, 2)
        features = features.flatten(start_dim=1)

        reverse_features = ReverseLayerF.apply(features, 1)
        label = self.domain_classifier(reverse_features)

        return label


if __name__ == '__main__':
    input = torch.rand(1, 12 * 8, 496, 448)

    model = UNet(
        in_channels=12 * 8,
        n_classes=6 * 8,
        depth=5,
        wf=6,
        padding=True,
        up_mode="upconv",
        batch_norm=True
    )

    params = sum(p.numel() for p in model.down_path.parameters() if p.requires_grad)
    print(f"Number of UNet down path model parameters: {params}")

    params = sum(p.numel() for p in model.up_path.parameters() if p.requires_grad)
    print(f"Number of UNet up path model parameters: {params}")

    da_model = DomainAdaptationModel(model, 1024 * 15 * 14)
    print("done creating model")

    params = sum(p.numel() for p in da_model.domain_classifier.parameters() if p.requires_grad)
    print(f"Number of DA model parameters: {params}")

    a = da_model.predict_domain(input)
    print(a)
