import torch
import torchvision.models as models
from torchvision import datasets, transforms
from PIL import Image

# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)
model.eval()


# 定义 R1 攻击函数
def r1_attack(model, image, label, max_queries=1000, target=None, epsilon=0.03):
    original_image = image.clone().detach()
    query_count = 0
    
    while query_count < max_queries:
        perturbation = torch.FloatTensor(image.shape).uniform_(-epsilon, epsilon)
        perturbed_image = image + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 确保像素值在合法范围内

        output = model(perturbed_image)    
        _, predicted = torch.max(output, 1)
        
        query_count += 1
        if target is not None and predicted.item() == target:
            # 如果是有目标攻击并且攻击成功，则返回对抗样本和查询次数
            print("adv ", predicted)
            return perturbed_image, query_count
        elif target is None and predicted.item() != label:
            # 如果是无目标攻击并且攻击成功，则返回对抗样本和查询次数
            print("adv ", predicted)
            return perturbed_image, query_count
    
    # 如果达到最大查询次数仍未成功，则返回原始图像和最大查询次数
    return original_image, max_queries

# 加载图像并进行预处理
data_transform = {
    "train":transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]),
    "val":transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
}
BATCH_SIZE = 1
# 加载数据集，指定训练或测试数据，指定于处理方式
train_data = datasets.CIFAR10(root='./CIFAR10/', train=True, transform=data_transform["train"], download=False)
test_data = datasets.CIFAR10(root='./CIFAR10/', train=False, transform=data_transform["val"], download=False)

train_dataloader = torch.utils.data.DataLoader(train_data, BATCH_SIZE, True, num_workers=0)
test_dataloader = torch.utils.data.DataLoader(test_data, BATCH_SIZE, False, num_workers=0)


# image_path = 'OIP-C.png'  # 替换为您的图像路径
# image = Image.open(image_path)
# image = data_transform(image).unsqueeze(0)  # 添加批次维度
for image,label in train_dataloader:
    print(label)
    output = model(image)    
    _, predicted = torch.max(output, 1)
    print("orange", predicted)
    # 执行 R1 攻击
    adversarial_image, queries = r1_attack(model, image, label, target=None, epsilon=0.03)


    # 保存对抗样本
    adversarial_image = adversarial_image.squeeze(0)
    adversarial_image = transforms.ToPILImage()(adversarial_image)
    adversarial_image.save('adversarial_image.png')
    image = transforms.ToPILImage()(image[0])
    image.save("image.png")
    print(f"攻击成功！查询次数：{queries}")
    break
