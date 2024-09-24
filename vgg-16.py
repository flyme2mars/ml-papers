import torch
from torch import nn


device = "cuda" if torch.cuda.is_available() else "cpu"


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )

        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(stride=2, kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25088, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1000),
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.classifier(x)
        return x


image_batch = torch.randn(64, 3, 224, 224).to(device)


model = VGG16().to(device)

logits = model(image_batch)

pred_probs = torch.softmax(logits, dim=1)

print(pred_probs[0])
