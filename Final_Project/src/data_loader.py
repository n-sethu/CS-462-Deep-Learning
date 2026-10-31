# data_loader.py
import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def unpickle(file):
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def get_cifar10_loaders(data_dir, batch_size=64):
    train_images = []
    train_labels = []
    
    # Load and stack training batches
    for i in range(1, 6):
        batch = unpickle(os.path.join(data_dir, f'../data/cifar-10-batches-py/data_batch_{i}'))
        train_images.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
        
    train_images = np.vstack(train_images)
    train_labels = np.array(train_labels)
    
    # Load test batch
    test_batch = unpickle(os.path.join(data_dir, 'test_batch'))
    test_images = test_batch[b'data']
    test_labels = np.array(test_batch[b'labels'])
    
    # Reshape and normalize
    x_train = train_images.reshape(-1, 3, 32, 32).astype('float32') / 255.0
    x_test = test_images.reshape(-1, 3, 32, 32).astype('float32') / 255.0
    
    # Convert to PyTorch Tensors
    x_train_tensor = torch.tensor(x_train)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.long)
    
    # Wrap in Datasets
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader