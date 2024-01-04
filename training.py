import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms as T
from timm.models import VisionTransformer
import time

def train(device, batch_size, patch_size, embedding_dim,
          learning_rate, epochs, dataset_root, print_log=False):
    
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,),)
    ])

    train_set = FashionMNIST(root=dataset_root, download=False, train=True, transform=transform)
    test_set = FashionMNIST(root=dataset_root, download=False, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = VisionTransformer(img_size=28, patch_size=patch_size,
                            num_classes=10, in_chans=1, 
                            embed_dim=embedding_dim,
                            depth=12, num_heads=12)
    model = model.to(device)

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    epoch_start = time.time()
    epoch_durations = []
    epoch_accuracies = []
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            l = loss(outputs, labels)
            
            optimizer.zero_grad()

            l.backward()
            optimizer.step()

        epoch_end = time.time()
        correct = 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(test_loader):

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                pred = outputs.argmax(dim=1)
                correct += (labels==pred).sum().item()
        
        epoch_durations.append(epoch_end - epoch_start)
        epoch_accuracies.append(correct / len(test_set))
        
        if print_log:
            print(epoch_durations)
            print(epoch_accuracies)

        model.train()

    return epoch_accuracies, epoch_durations
