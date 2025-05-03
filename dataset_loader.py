import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PoemDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        poem = self.data[idx]
        # 取出不包含最后一个字符的序列作为输入
        x = poem[:-1][:self.seq_length]
        # 取出不包含第一个字符的序列作为目标
        y = poem[1:][:self.seq_length]
        
        # 创建长度一致的序列（填充或截断）
        if len(x) < self.seq_length:
            x = np.pad(x, (0, self.seq_length - len(x)), 'constant', constant_values=8292)
        if len(y) < self.seq_length:
            y = np.pad(y, (0, self.seq_length - len(y)), 'constant', constant_values=8292)
            
        return torch.LongTensor(x), torch.LongTensor(y)

def load_data(file_path='./data/tang.npz', seq_length=64, batch_size=64, val_split=0.1):
    """加载唐诗数据集并创建数据加载器"""
    # 加载数据
    data = np.load(file_path, allow_pickle=True)
    
    # 获取数据集的三个部分
    poem_data = data['data']  # 诗词数据
    ix2word = data['ix2word'].item()  # 序号到字的映射
    word2ix = data['word2ix'].item()  # 字到序号的映射
    vocab_size = len(word2ix)
    
    # 划分训练集和验证集
    indices = np.random.permutation(len(poem_data))
    val_size = int(len(poem_data) * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_data = poem_data[train_indices]
    val_data = poem_data[val_indices]
    
    # 创建数据集和数据加载器
    train_dataset = PoemDataset(train_data, seq_length)
    val_dataset = PoemDataset(val_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, word2ix, ix2word, vocab_size

def prepare_input(text, word2ix, pad_idx=8292, device='cuda'):
    """将输入文本转换为模型可接受的格式"""
    indices = []
    for c in text:
        if c in word2ix:
            indices.append(word2ix[c])
        else:
            # 对于不在词汇表中的字符，使用一个特殊标记
            indices.append(pad_idx)
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)