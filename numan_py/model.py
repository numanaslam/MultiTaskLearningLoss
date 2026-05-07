# model.py
import torch
import torch.nn as nn
import torchvision.models as models

class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        # Register hook to capture features[29] (relu5_3) for GradCAM
        self.feature_map = None
        def hook_fn(module, input, output):
            self.feature_map = output
        self.features[29].register_forward_hook(hook_fn)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)