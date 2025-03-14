import torch
import torch.nn as nn
from torchvision import models


class SkinLesionModel(nn.Module):
    def __init__(self, num_classes=2, model_type='efficient'):
        super(SkinLesionModel, self).__init__()

        # chose resnet, intitially went with resnet50, but resnet18 is computationally cheaper for my gpu
        if model_type == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            num_features = self.model.fc.in_features

            for name, param in self.model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif model_type == 'efficient':
            # EfficientNet-B0 is much faster than ResNet50 with similar performance https://bmcmedimaging.biomedcentral.com/articles/10.1186/s12880-022-00793-7
            self.model = models.efficientnet_b0(pretrained=True)
            num_features = self.model.classifier[1].in_features

            for name, param in self.model.named_parameters():
                if 'features.8' not in name and 'classifier' not in name:
                    param.requires_grad = False

            self.model.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, num_classes)
            )

        else:  # backup to resnet with improvements
            self.model = models.resnet50(pretrained=True)
            num_features = self.model.fc.in_features

            for name, param in self.model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

            self.model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

    def forward(self, x):
        return self.model(x)

    def unfreeze_more_layers(self):
        if hasattr(self.model, 'layer3'):  # ResNet model
            for param in self.model.layer3.parameters():
                param.requires_grad = True
        elif hasattr(self.model, 'features'):  # EfficientNet model
            # unfreeze layers the last few blocks
            for i in range(6, 9):
                for param in self.model.features[i].parameters():
                    param.requires_grad = True