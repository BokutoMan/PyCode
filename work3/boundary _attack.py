import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 加载 MNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# 定义 ResNet-18 模型作为被攻击模型
model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)  # 将最后的全连接层修改为输出10个类别
model = model.cuda()  # 将模型移动到GPU上
model.train()

# 定义边界攻击函数
def boundary_attack(model, image, label, max_queries=1000, epsilon=0.03):
    original_image = image.clone().detach()
    query_count = 0
    
    while query_count < max_queries:
        perturbation = torch.FloatTensor(image.shape).uniform_(-epsilon, epsilon)
        perturbed_image = image + perturbation
        perturbed_image = torch.clamp(perturbed_image, 0, 1)  # 确保像素值在合法范围内

        output = model(perturbed_image.cuda())
        _, predicted = torch.max(output, 1)
        
        query_count += 1
        
        if predicted.item() == label:
            return True, query_count  # 有目标攻击成功
        elif predicted.item() != label:
            return False, query_count  # 无目标攻击成功
    
    return False, max_queries  # 攻击失败

# 统计攻击成功率
total_samples = len(testset)
targeted_attack_successes = 0
untargeted_attack_successes = 0

for images, labels in testloader:
    images = images.cuda()
    labels = labels.cuda()
    
    for i in range(len(images)):
        image = images[i:i+1]
        label = labels[i].item()
        
        # 有目标攻击
        targeted_success, _ = boundary_attack(model, image, label)
        if targeted_success:
            targeted_attack_successes += 1
        
        # 无目标攻击
        untargeted_success, _ = boundary_attack(model, image, label, target=np.random.choice([i for i in range(10) if i != label]))
        if untargeted_success:
            untargeted_attack_successes += 1

print(f"有目标攻击成功率: {targeted_attack_successes / total_samples * 100}%")
print(f"无目标攻击成功率: {untargeted_attack_successes / total_samples * 100}%")
