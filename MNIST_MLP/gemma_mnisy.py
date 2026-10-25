import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DeepMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dims=[1024, 512, 256], num_classes=10, dropout_rate=0.3):
        super(DeepMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
            
        # Final output projection
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten the 28x28 grayscale images into a 784 vector
        x = x.view(x.size(0), -1)
        return self.network(x)
def main():
    # 1. Device Diagnostics
    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )
    print(f"Using device: {device}")

    # 2. Data Preparation & Augmentation
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),  # Minor rotational variance
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # Standard MNIST mean/std

    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model Initialization
    model = DeepMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # AdamW introduces decoupled weight decay for cleaner stabilization
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Cosine Annealing dynamically cools down learning rate over time
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    # 4. Loop Execution
    epochs = 40
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True) # Optimized memory usage
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        scheduler.step()
        train_loss /= len(train_loader.dataset)

        # Evaluation Phase
        model.eval()
        test_loss = 0
        correct = 0
        with torch.inference_mode(): # Cleaner & faster than torch.no_grad()
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        if accuracy > best_acc:
            best_acc = accuracy

        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Accuracy: {accuracy:.2f}% (Best: {best_acc:.2f}%)")
        
        if accuracy >= 99.0:
            print("Target 99% accuracy accomplished!")

if __name__ == '__main__':
    main()
# **Data Augmentation:** Apply minor random rotations, scaling, and translations to the 60,000 training images so the network generalizes better to unseen handwriting variations, can also add some gaussian noise **Hidden Layers:** 3 to 6 fully connected (dense) layers with 512 to 1024 neurons per layer. , Batch Normalization after each hidden layer and Dropout (rate of 0.2 to 0.4) to prevent overfitting, 10 nodes with Softmax activation for digit classification. [[1] create a new file that tries this as well The model processes images by splitting them into a grid of non-overlapping patches, projecting them linearly into hidden dimensions, and passing them through repeated mixer layers. Each layer contains two distinct MLP block