import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import os


class Dog_Cat_Net(nn.Module):
    def __init__(self, num_classes=2):
        super(Dog_Cat_Net, self).__init__()
        self.model = models.resnet50(pretrained=False)
        fc_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=fc_features, out_features=num_classes)

    def forward(self, x):
        return self.model(x)


class PredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("猫狗图片分类器")
        self.root.geometry("900x750")

        # 设置窗口背景色
        self.root.configure(bg='#f0f0f0')

        # 初始化模型
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Dog_Cat_Net(num_classes=2)
        self.model.to(self.device)

        # 加载模型权重
        try:
            self.model.load_state_dict(torch.load('Model.pth', map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            print("错误: 模型权重文件未找到")
            exit()

        # 创建界面元素
        self.create_widgets()

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.class_labels = ["猫", "狗"]

    def create_widgets(self):
        # 创建主框架并居中
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.place(relx=0.5, rely=0.5, anchor="center")

        # 创建标题标签
        title_label = ttk.Label(
            main_frame,
            text="猫狗图片分类器",
            font=('Arial', 24, 'bold')
        )
        title_label.grid(row=0, column=0, pady=(0, 20))

        # 创建选择图片按钮（使用自定义样式）
        style = ttk.Style()
        style.configure(
            'Custom.TButton',
            font=('Arial', 14),
            padding=10
        )

        self.select_btn = ttk.Button(
            main_frame,
            text="选择图片",
            command=self.select_image,
            style='Custom.TButton',
            width=20
        )
        self.select_btn.grid(row=1, column=0, pady=20)

        # 创建图片显示区域（无边框）
        self.image_label = ttk.Label(main_frame)
        self.image_label.grid(row=2, column=0, pady=20)

        # 创建预测结果区域
        result_frame = ttk.Frame(main_frame)
        result_frame.grid(row=3, column=0, pady=10)

        self.result_label = ttk.Label(
            result_frame,
            text="预测结果将在这里显示",
            font=('Arial', 16, 'bold'),
            foreground='#2c3e50'
        )
        self.result_label.pack()

        # 置信度显示
        self.confidence_label = ttk.Label(
            result_frame,
            text="",
            font=('Arial', 14),
            foreground='#34495e'
        )
        self.confidence_label.pack(pady=(5, 0))

    def select_image(self):
        # 打开文件选择对话框
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )

        if file_path:
            # 显示选中的图片
            self.display_image(file_path)
            # 进行预测
            self.predict_image(file_path)

    def display_image(self, image_path):
        # 打开并调整图片大小以适应显示
        image = Image.open(image_path)
        # 调整图片大小，保持比例
        image.thumbnail((400, 400))
        photo = ImageTk.PhotoImage(image)

        self.image_label.configure(image=photo)
        self.image_label.image = photo  # 保持引用

    def predict_image(self, image_path):
        # 加载并预处理图片
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)

        # 进行预测
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # 获取预测结果
        probabilities = F.softmax(outputs, dim=1)
        predicted_prob, predicted_index = torch.max(probabilities, 1)

        # 更新界面显示
        predicted_label = self.class_labels[predicted_index.item()]
        confidence = predicted_prob.item() * 100

        # 根据预测结果设置不同的颜色
        result_color = '#27ae60' if predicted_label == "猫" else '#e74c3c'

        self.result_label.configure(
            text=f"预测结果: {predicted_label}",
            font=('Arial', 18, 'bold'),
            foreground=result_color
        )
        self.confidence_label.configure(
            text=f"置信度: {confidence:.2f}%",
            font=('Arial', 16),
            foreground='#34495e'
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionGUI(root)
    root.mainloop()