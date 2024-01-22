import numpy as np
import time
import os

from tqdm import tqdm, trange

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from torchvision.transforms import ToTensor
from torchvision.datasets.mnist import MNIST
from torchvision.datasets import Caltech256
from torchvision.datasets import CIFAR10

from torch_geometric.nn import GATConv


# np.random.seed(0)
# torch.manual_seed(0)
MODELNAME = "last_model.mod"

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

def incr(lst, i):
    return [x+i for x in lst]

def get_full_edge_index(n_patches):
    edge_index = [[],
                  []]
    
    for a in range(n_patches*n_patches+1):
        for b in range(n_patches*n_patches+1):
            # if a == 0:
            #     continue
            # if a != b:
                edge_index[0].append(a)
                edge_index[1].append(b)
    return edge_index


def get_edge_index(n_patches):
    edge_index = [[],
                  []]
    
    for i in range(n_patches):
        for j in range(n_patches):
            a = i + j * n_patches
            
            if i > 0:
                b = (i-1) + j * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if j > 0:
                b = i + (j-1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if i < n_patches - 1:
                b = (i+1) + j * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if j < n_patches - 1:
                b = i + (j+1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
    
    edge_index[0] = incr(edge_index[0],1)
    edge_index[1] = incr(edge_index[1],1)

    for i in range(n_patches):
        for j in range(n_patches):
            a = 0
            b = i + j * n_patches + 1
            edge_index[0].append(b)
            edge_index[1].append(a)
    
    return edge_index

def get_squere_edge_index(n_patches):
    edge_index = [[],
                  []]
    
    for i in range(n_patches):
        for j in range(n_patches):
            a = i + j * n_patches
            
            # cross
            if i > 0:
                b = (i-1) + j * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
                
            if i > 0 and j > 0:
                b = (i-1) + (j-1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if j > 0:
                b = i + (j-1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if i < n_patches - 1 and j > 0:
                b = (i+1) + (j-1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if i < n_patches - 1:
                b = (i+1) + j * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
            if i < n_patches - 1 and j < n_patches - 1:
                b = (i+1) + (j+1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)

            if j < n_patches - 1:
                b = i + (j+1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            

            if i > 0 and j < n_patches - 1:
                b = (i-1) + (j+1) * n_patches
                edge_index[0].append(a)
                edge_index[1].append(b)
            
    
    edge_index[0] = incr(edge_index[0],1)
    edge_index[1] = incr(edge_index[1],1)

    for i in range(n_patches):
        for j in range(n_patches):
            a = 0
            b = i + j * n_patches + 1
            edge_index[0].append(b)
            edge_index[1].append(a)
    
    return edge_index

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n_heads):
        super(GNNLayer, self).__init__()

        assert out_channels % n_heads == 0, f"Can't divide dimension {in_channels} into {n_heads} heads"
        
        self.conv = GATConv(in_channels, int(out_channels/n_heads), heads = n_heads)
        # self.conv = GATConv(in_channels, out_channels,1)
        # self.conv = CuGraphGATConv(in_channels, out_channels,1)
        # self.conv = GATv2Conv(in_channels, out_channels,1)

        

    def forward(self, x, edge_index):
        # print(x.shape)
        # print(edge_index.shape)
        ret = []
        # ret = torch.tensor(ret)
        # print(x.shape)
        for i in range(x.shape[0]):
            # print(x[i].shape)
            # print(x[i].shape)
            
            ret.append(self.conv(x[i], edge_index))
            
        return torch.stack(ret)
    
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4, n_patches=7):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        # self.mhsa = MyMSA(hidden_d, n_heads)
        self.Gnn = GNNLayer(hidden_d, hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

        self.edge_index = torch.tensor(get_squere_edge_index(n_patches))

    def forward(self, x):
        # out = x + self.mhsa(self.norm1(x))
        # print(x.shape)
        # print(self.Gnn(self.norm1(x),self.edge_index).shape)
        out = x + self.Gnn(self.norm1(x),self.edge_index)

        out = out + self.mlp(self.norm2(out))
        return out
        
class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Softmax(dim=-1)
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution

def save_progress(model, data, epoch: int, path: str):

    torch.save(model.state_dict(), path+MODELNAME)

    with open(path+"current_epoch", "w") as f:
        f.write(str(epoch))

    with open(path+"data.dat", "a") as f:
        f.write(";".join([str(d) for d in data])+"\n")
    
def execution(model, train_loader: DataLoader, test_loader: DataLoader, device, path: str, n_EPOCHS: int, current_epoch = 0, LR = 0.005):
    # Training loop
    optimizer = Adam(model.parameters(), lr=LR)
    criterion = CrossEntropyLoss()
    
    loss_list = []
    for epoch in trange(current_epoch,n_EPOCHS, desc="Training"):
        train_loss = 0.0
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            
            train_loss += loss.detach().cpu().item() / len(train_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc = correct / total * 100
        print(f"\nEpoch {epoch + 1}/{n_EPOCHS} train loss: {train_loss:.2f}\n")
        
        # Test loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(test_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(test_loader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
            test_acc = correct / total * 100
            # print(f"Test loss: {test_loss:.2f}")
            # print(f"Test accuracy: {correct / total * 100:.2f}%")
        print(f"\nEpoch {epoch + 1}/{n_EPOCHS} test loss: {test_loss:.2f}\n")

        save_progress(model, [train_loss, train_acc, test_loss, test_acc], epoch, path)
            

def make_experiment(n_EPOCHS: int, test_name = "Test_001"):

    #preparing data
    DATA = CIFAR10
    transform = ToTensor()

    train_set = CIFAR10(root='./../datasets', train=True, download=True, transform=transform)
    test_set = CIFAR10(root='./../datasets', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=128)

    #getting picture size and number od classes
    subset_indices = [0,1]
    subset = torch.utils.data.Subset(train_set, subset_indices)
    pic_size = tuple(next(iter(DataLoader(subset)))[0].shape)[1:]
    classes_num = len(train_set.classes)

    #preparing data saving
    path = f"results/{test_name}/"
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

    #params
    n_patches = 8
    n_blocks = 2
    hidden_d = 64
    n_heads = 8
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    #creating model
    model = MyViT(pic_size, n_patches=n_patches, n_blocks=n_blocks, hidden_d=hidden_d, n_heads=n_heads, out_d=classes_num).to(device)
    
    if not isExist:
        torch.save(model.state_dict(), path+MODELNAME)
        with open(path+"current_epoch","a+") as f: f.write("-1")
        with open(path+"data.dat","a+"): pass
        current_epoch = 0
        print("Tworzę nowy model")
    else:
        model.load_state_dict(torch.load(path+MODELNAME))
        model.eval()
        with open(path+"current_epoch") as f:
            current_epoch = int(f.read()) + 1
        print("Pomyślnie wczytano model")

    execution(model, train_loader, test_loader, device, path, n_EPOCHS = n_EPOCHS, current_epoch = current_epoch)

if __name__ == '__main__':
  make_experiment(50, "CIFAR10_gnn_squere_transformer_01")