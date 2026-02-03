import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

transform = transforms.ToTensor()
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform)

img, label = dataset[0]

plt.imshow(img.permute(1,2,0))
plt.title(f"Label: {dataset.classes[label]}")
plt.axis("off")
plt.show()
