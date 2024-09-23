import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"


transform = transforms.Compose(
    [
        transforms.Resize((227, 227)),  # Resize images to 2
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

trainset = datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
testset = datasets.CIFAR100(
    root="./data", train=False, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


class_names = trainset.classes


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(in_features=6 * 6 * 256, out_features=4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=100),
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.classifier(x)
        return x


model = AlexNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)  # type: ignore


for epoch in range(10):
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        pred_prob = logits.softmax(dim=1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/10], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}"
            )


model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        outputs = logits.softmax(dim=0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")
