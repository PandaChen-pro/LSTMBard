import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PoemDataset(Dataset):
    def __init__(self, poems, seq_length):
        self.poems = poems
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.poems) - self.seq_length
        
    def __getitem__(self, idx):
        x = self.poems[idx:idx + self.seq_length]
        y = self.poems[idx + 1:idx + self.seq_length + 1]
        return x, y

def load_data(file_path='./data/tang.npz', seq_length=48, batch_size=64):
    """加载唐诗数据集并创建数据加载器"""
    # 加载数据
    data = np.load(file_path, allow_pickle=True)
    print(data.keys())
    poems = data['poems']
    
    # 合并所有诗句为一个长字符串
    text = ''.join(poems)
    
    # 创建字符到索引的映射
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # 将文本转换为索引序列
    indices = [char_to_idx[c] for c in text]
    indices = torch.tensor(indices, dtype=torch.long)
    
    # 创建数据集和数据加载器
    dataset = PoemDataset(indices, seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, char_to_idx, idx_to_char, len(chars)

def prepare_input(text, char_to_idx, device='cuda'):
    """将输入文本转换为模型可接受的格式"""
    indices = [char_to_idx.get(c, 0) for c in text]  # 使用0作为未知字符的索引
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

if __name__ == "__main__":
    dataloader, char_to_idx, idx_to_char, vocab_size = load_data()
    print(dataloader)
    print(char_to_idx)
    print(idx_to_char)
    print(vocab_size)