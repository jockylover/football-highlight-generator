import torch
import clip
from torch import nn


class FineTunedCLIP(nn.Module):
    def __init__(self, freeze_clip=False, unfreeze_layers=2):
        super().__init__()
        self.clip_model, _ = clip.load("ViT-B/32", device="cpu")
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            # 解冻最后几层
            clip_layers = list(self.clip_model.transformer.resblocks)
            for layer in clip_layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
        # self.classifier = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # )


    def forward(self, x):
        features = self.clip_model.encode_image(x)
        return self.classifier(features)