import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x = torch.randn(16, 3, 224, 224).to(device)
y = torch.randint(low=0, high=3, size=(16,), device=device).to(device)

model = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Flatten(),
    nn.Linear(in_features=16, out_features=3)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

model.train()

for epoch in range(10):
    optimizer.zero_grad()

    predictions = model(x)
    loss = criterion(predictions, y)

    loss.backward()
    optimizer.step()

    print(f"Época: {epoch + 1}, Perda: {loss.item():.4f}")