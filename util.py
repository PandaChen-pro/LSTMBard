import torch
import numpy as np

def set_seed(seed=42):
    """设置随机种子以确保结果可复现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def format_poem_output(poem):
    """格式化诗歌输出，使其更美观"""
    # 分割成行
    lines = []
    line = ""
    for char in poem:
        line += char
        if char in ['。', '！', '？']:
            lines.append(line)
            line = ""
        elif char == '，' and len(line) >= 5:
            lines.append(line)
            line = ""
    
    if line:  # 添加剩余未完成的行
        lines.append(line)
    
    return '\n'.join(lines)

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