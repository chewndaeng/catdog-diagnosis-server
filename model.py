import torch.nn as nn
import timm

class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, model_name="mobilenetv3_small_100", pretrained=True):
        super(MobileNetClassifier, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)