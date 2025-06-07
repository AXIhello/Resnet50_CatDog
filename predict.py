import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import time # 如果需要计时可以保留

# ----------------------------------------------------------------------
# 1. 定义与训练时完全相同的网络结构
#    这部分必须与训练脚本中 Dog_Cat_Net 的定义一模一样
# ----------------------------------------------------------------------
class Dog_Cat_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Dog_Cat_Net, self).__init__()
        # 确保 pretrained 参数与训练时一致（通常是 True）
        self.model = models.resnet50(pretrained=False) # 预测时可以不用下载预训练权重，因为我们会加载自己的权重
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=fc_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

# ----------------------------------------------------------------------
# 2. 主预测逻辑
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print("--- 启动模型预测脚本 ---")

    # 指定模型权重文件路径
    model_path = 'Model.pth' # 确保这个路径指向你训练脚本保存的模型文件

    # 指定设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 实例化模型
    model_pred = Dog_Cat_Net(num_classes=2)
    model_pred.to(device)

    # 加载训练好的模型权重
    try:
        # 加载 state_dict，注意 map_location 参数以兼容不同设备
        model_pred.load_state_dict(torch.load(model_path, map_location=device))
        print(f"成功加载模型权重: {model_path}")
    except FileNotFoundError:
        print(f"错误: 模型权重文件未找到在 {model_path}")
        print("请先运行训练脚本生成 Model.pth 文件。")
        exit() # 退出脚本

    # 设置模型为评估模式
    model_pred.eval()

    # ----------------------------------------------------------------------
    # 3. 定义图像预处理步骤
    #    这些步骤必须与训练/测试时数据加载器中使用的预处理步骤完全一致！
    # ----------------------------------------------------------------------
    preprocess = transforms.Compose([
        transforms.Resize(256),        # 先缩放到 256x256
        transforms.CenterCrop(224),    # 中心裁剪到 224x224 (ResNet的标准输入尺寸)
        transforms.ToTensor(),         # 转换为 Tensor，范围 [0, 1]
        # 标准化参数必须与训练时一致
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 统计量
    ])

    # ----------------------------------------------------------------------
    # 4. 准备待预测的图片并进行预测
    # ----------------------------------------------------------------------

    # *** 请将下面的列表替换为你想要预测的图片文件的实际路径 ***
    image_paths_to_predict = [
        'test/test_cat.jpg',
        'test/test_cat2.jpg',
        'test/test_cat3.jpg',
        'test/test_dog.jpg',
        # 可以添加更多图片路径
    ]

    # 假设类别索引到标签的映射 (0: Cat, 1: Dog)
    # *** 请根据你的实际数据加载器中类别的顺序来确定这里的映射 ***
    class_labels = ["Cat", "Dog"] # 示例：索引 0 对应 Cat，索引 1 对应 Dog

    print(f"\n待预测图片数量: {len(image_paths_to_predict)}")

    for i, image_path_to_predict in enumerate(image_paths_to_predict):
        print(f"\n--- 预测图片 {i+1}/{len(image_paths_to_predict)}: {image_path_to_predict} ---")
        try:
            # 加载图片并确保是 RGB 格式
            image = Image.open(image_path_to_predict).convert('RGB')
        except FileNotFoundError:
            print(f"错误: 图片未找到在 {image_path_to_predict}，跳过。")
            continue # 跳过当前图片，处理下一张

        # 对图片进行预处理，得到用于模型输入的 Tensor
        image_tensor = preprocess(image)

        # 添加一个 batch 维度 (因为模型期望 batch 输入)
        image_tensor = image_tensor.unsqueeze(0) # shape 从 [C, H, W] 变为 [1, C, H, W]

        # 将图片 Tensor 移动到与模型相同的设备
        image_tensor = image_tensor.to(device)

        # 进行预测
        with torch.no_grad(): # 预测时不需要计算梯度
            outputs = model_pred(image_tensor) # 前向传播，得到模型的原始输出 (logits)

        # 解析输出结果
        # outputs 是 [1, num_classes] 的 Tensor，包含每个类别的得分
        probabilities = F.softmax(outputs, dim=1) # 应用 Softmax 转换为概率
        predicted_prob, predicted_index = torch.max(probabilities, 1) # 找到最大概率及其索引

        # 获取预测的类别标签和对应的概率
        predicted_label = class_labels[predicted_index.item()]
        predicted_confidence = predicted_prob.item()

        # 打印预测结果
        print(f"预测的类别索引: {predicted_index.item()}")
        print(f"预测的类别标签: {predicted_label}")
        print(f"预测的置信度: {predicted_confidence:.4f}")
        # print(f"原始输出 (Logits): {outputs.squeeze().cpu().numpy()}") # 可选打印原始输出
        # print(f"概率分布: {probabilities.squeeze().cpu().numpy()}") # 可选打印概率分布

    print("\n--- 模型预测脚本结束 ---")