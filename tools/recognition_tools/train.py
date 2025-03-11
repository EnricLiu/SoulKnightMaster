import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # Add this import
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import EquipClassifierCNN
# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

# 加载数据集
train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型


# 初始化模型、损失函数和优化器
model = EquipClassifierCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # 将标签转换为浮点型
        labels = labels.float().unsqueeze(1)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.10f}')

print("Training complete.")
# 保存模型
torch.save(model.state_dict(), 'skill_cnn.pth')
print("Model saved successfully.")