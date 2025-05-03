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

def preprocess_text(text):
    """预处理文本，去除特殊字符和标点符号"""
    # 保留常用的标点符号如句号、逗号，但去除其他特殊字符
    chars_to_remove = set("!@#$%^&*()_+-=[]{}|;':\"<>/?.`~")
    return ''.join([c for c in text if c not in chars_to_remove])

def split_poem(poem, line_length=7):
    """将长诗分割为多行"""
    lines = []
    for i in range(0, len(poem), line_length):
        lines.append(poem[i:i+line_length])
    return '\n'.join(lines)

def validate_poem_grammar(poem):
    """简单验证诗歌语法和结构（此处为简化实现）"""
    # 检查长度
    if len(poem) < 10:
        return False
    
    # 检查是否包含标点符号
    if not any(p in poem for p in ['。', '，']):
        return False
    
    return True

def format_poem_output(poem):
    """格式化诗歌输出，使其更美观"""
    # 分割成行
    lines = poem.strip().split('\n')
    
    # 格式化每一行
    formatted_lines = []
    for line in lines:
        # 去除空行
        if not line.strip():
            continue
        # 确保每行都有标点
        if not line.strip()[-1] in ['。', '，', '！', '？']:
            line = line.strip() + '，'
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def sample_with_temperature(logits, temperature=1.0):
    """使用温度参数进行采样"""
    logits = logits / temperature
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

def is_chinese_char(char):
    """判断是否为汉字"""
    return '\u4e00' <= char <= '\u9fff'

def count_chinese_chars(text):
    """计算汉字数量"""
    return sum(1 for c in text if is_chinese_char(c))