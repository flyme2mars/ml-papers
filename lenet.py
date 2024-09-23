import torch
from torch import nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.ToTensor(),  # Convert image to tensor
    ]
)

trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)


class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.C1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.Sigmoid = nn.Sigmoid()
        self.S2 = nn.AvgPool2d(kernel_size=2)
        self.C3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.S4 = nn.AvgPool2d(kernel_size=2)
        self.C5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.flatten = nn.Flatten()
        self.F6 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.C1(x)
        x = self.Sigmoid(x)
        x = self.S2(x)
        x = self.C3(x)
        x = self.Sigmoid(x)
        x = self.S4(x)
        x = self.C5(x)
        x = self.Sigmoid(x)
        x = self.flatten(x)
        x = self.F6(x)
        x = self.Sigmoid(x)
        x = self.out(x)
        return x


model = Lenet5()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # type: ignore

for epoch in range(5):
    for i, (inputs, labels) in enumerate(trainloader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(
                f"Epoch [{epoch + 1}/5], Step [{i + 1}/{len(trainloader)}], Loss: {loss.item():.4f}"
            )

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in testloader:
        logits = model(inputs)
        outputs = logits.softmax(dim=0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy of the model on the test set: {100 * correct / total:.2f}%")
