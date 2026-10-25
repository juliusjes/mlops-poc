import torch
from torch import nn


class SimpleTestModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Linear(8, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(torch.relu(emb))
        return {"logits": logits, "emb": emb}



class ResidualBlock(nn.Module):
    def __init__(self, s1, ci1, co1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(
            in_channels=ci1,
            out_channels=co1,
            kernel_size=(3, 3),
            stride=s1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(co1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=co1,
            out_channels=co1,
            kernel_size=(3, 3),
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(co1)

        if s1 != 1 or ci1 != co1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ci1, co1, kernel_size=1, stride=s1, bias=False),
                nn.BatchNorm2d(co1),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class GlobalAveragePool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        return torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)


class SimpleResNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.stages = nn.Sequential(
            ResidualBlock(s1=1, ci1=32, co1=32),
            ResidualBlock(s1=2, ci1=32, co1=64),
            ResidualBlock(s1=2, ci1=64, co1=128),
            ResidualBlock(s1=2, ci1=128, co1=256)
        )
        self.pool = GlobalAveragePool()
        self.lin0 = nn.Linear(256, 128)
        self.lin1 = nn.Linear(128, 64)
        self.lin2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages(x)
        
        x = self.pool(x)
        emb = self.lin0(x)

        x = self.lin1(emb)
        logits = self.lin2(x)

        return {
            "logits": logits,
            "emb": emb
        }
