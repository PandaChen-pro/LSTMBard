import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset_loader import prepare_input
import numpy as np

class PoemLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5, pad_idx=8292):
        super(PoemLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pad_idx = pad_idx
        
        # 定义嵌入层和LSTM层
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        # x: [batch_size, seq_length]
        batch_size = x.size(0)
        
        # 初始化隐藏状态
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)
            
        # 嵌入和LSTM前向传播
        embeds = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        lstm_out, hidden = self.lstm(embeds, hidden)  # [batch_size, seq_length, hidden_dim]
        
        # 全连接层输出
        output = self.dropout(lstm_out)
        output = self.fc(output)  # [batch_size, seq_length, vocab_size]
        
        return output, hidden
    
    def init_hidden(self, batch_size, device):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        return (h0, c0)
    
    def generate(self, initial_text, word2ix, ix2word, max_length=100, 
                temperature=1.0, device='cuda'):
        """生成诗句"""
        self.eval()
        result = initial_text
        
        # 将初始文本转换为索引序列
        input_seq = prepare_input(initial_text, word2ix, self.pad_idx, device)[0].tolist()
        
        hidden = None
        with torch.no_grad():
            for _ in range(max_length):
                # 准备输入
                # 限制输入序列长度，防止过长
                current_input_seq = input_seq[-self.lstm.input_size:] 
                x = torch.tensor(current_input_seq).unsqueeze(0).to(device)
                
                # 前向传播
                output, hidden = self(x, hidden)
                
                # 获取最后一个时间步的输出
                output = output[:, -1, :] / temperature
                
                # 转换为概率分布，并忽略填充标记
                output[:, self.pad_idx] = -float('inf')  # 避免生成填充标记
                probs = F.softmax(output, dim=-1).squeeze()
                
                # 采样下一个字符
                pred_idx = torch.multinomial(probs, 1).item()
                
                # 如果生成了填充标记或无效索引，尝试重新采样或跳过
                if pred_idx == self.pad_idx or pred_idx not in ix2word:
                    continue
                
                # 添加到结果
                pred_char = ix2word[pred_idx] 
                result += pred_char
                
                # 更新输入序列
                input_seq.append(pred_idx) # 只添加新的索引
                
                # 如果生成了特定标点或达到足够长度，可以结束生成
                if pred_char in ['。', '！', '？'] and len(result) > len(initial_text) + 20:
                    break
                if len(result) >= len(initial_text) + max_length: # 添加显式最大长度停止条件
                    break

        return result
    
    def generate_acrostic(self, head_chars, word2ix, ix2word, 
                         line_length=7, temperature=1.0, device='cuda'):
        """生成藏头诗"""
        self.eval()
        result = ""
        
        with torch.no_grad():
            # 为每一行生成诗句
            for i, head_char in enumerate(head_chars):
                # 确保首字是给定的字
                if head_char in word2ix:
                    current_line = head_char
                    # 初始输入序列
                    input_seq = [word2ix[head_char]]
                else:
                    # 如果首字不在词汇表中，选择一个随机字符
                    random_idx = np.random.randint(0, self.vocab_size) # 0 到 vocab_size-1
                    while random_idx == self.pad_idx or random_idx not in ix2word: # 确保不是pad且有效
                        random_idx = np.random.randint(0, self.vocab_size)
                    current_line = ix2word[random_idx] # <--- 修改处
                    input_seq = [random_idx]
                    print(f"Warning: Head character '{head_char}' not in vocabulary. Using '{current_line}' instead.")

                
                hidden = None
                
                # 生成剩余的字符
                for j in range(line_length - 1):
                    # 限制输入序列长度
                    current_input_seq = input_seq[-self.lstm.input_size:]
                    x = torch.tensor(current_input_seq).unsqueeze(0).to(device)
                    output, hidden = self(x, hidden)
                    output = output[:, -1, :] / temperature
                    
                    # 避免生成填充标记
                    output[:, self.pad_idx] = -float('inf')
                    probs = F.softmax(output, dim=-1).squeeze()
                    pred_idx = torch.multinomial(probs, 1).item()
                    
                    # 如果生成了填充标记或无效索引，尝试重新采样
                    retry_count = 0
                    while (pred_idx == self.pad_idx or pred_idx not in ix2word) and retry_count < 5:
                        pred_idx = torch.multinomial(probs, 1).item()
                        retry_count += 1
                    
                    if pred_idx == self.pad_idx or pred_idx not in ix2word: # 重试后仍无效，跳过本次生成
                        print("Warning: Could not generate a valid character, skipping.")
                        continue # 或者 break，取决于希望的行为
                        
                    pred_char = ix2word[pred_idx] # <--- 修改处
                    current_line += pred_char
                    input_seq.append(pred_idx)
                
                # 添加标点和换行
                if i < len(head_chars) - 1:
                    if (i + 1) % 2 == 0:
                        current_line += "。" # 五绝/七绝 通常是 逗号/句号 交替
                    else:
                        current_line += "，"
                else:
                    current_line += "。" # 最后一句用句号
                
                result += current_line + "\n" # 每句后加换行符

        # 移除最后一个多余的换行符
        return result.strip()