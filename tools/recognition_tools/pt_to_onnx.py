import torch
import torch.onnx

from src.model import EquipClassifierCNN


# 加载模型
model = EquipClassifierCNN()
checkpoint = torch.load('./skill_cnn.pth', map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 64, 64)
print(dummy_input.shape)

# 导出模型为 ONNX 格式
torch.onnx.export(model, dummy_input, "model.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("模型已成功导出为 ONNX 格式")