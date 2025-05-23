import argparse
import torch
import numpy as np
import wandb
from dataset_loader import load_data, prepare_input
from model import PoemLSTM
from trainer import PoemTrainer
from util import set_seed, format_poem_output

def main():
    parser = argparse.ArgumentParser(description='LSTM 唐诗生成')
    parser.add_argument('--data_path', type=str, default='./data/tang.npz', help='数据集路径')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate', 'acrostic'], 
                        help='运行模式：训练模型或生成诗句')
    parser.add_argument('--model_path', type=str, default='checkpoints/final_model.pth', 
                        help='用于生成的模型路径')
    parser.add_argument('--initial_text', type=str, default='湖光秋月两相和', 
                        help='用于生成的初始文本')
    parser.add_argument('--head_chars', type=str, default='春夏秋冬', 
                        help='藏头诗的首字')
    parser.add_argument('--embedding_dim', type=int, default=256, help='嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=3, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout率')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='批大小')
    parser.add_argument('--seq_length', type=int, default=64, help='序列长度')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--use_wandb', action='store_true', help='是否使用wandb记录')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='运行设备')
    parser.add_argument('--resume', type=str, default=None, help='从检查点恢复训练')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--temperature', type=float, default=0.8, help='生成时的温度参数')
    parser.add_argument('--pad_idx', type=int, default=8292, help='填充标记的索引')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减 (L2 正则化)') 
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    train_loader, val_loader, word2ix, ix2word, vocab_size = load_data(
        args.data_path, args.seq_length, args.batch_size)
    
    model = PoemLSTM(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_idx=args.pad_idx
    ).to(args.device)
    
    if args.mode == 'train':
        trainer = PoemTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=args.lr,
            device=args.device,
            pad_idx=args.pad_idx,
            weight_decay=args.weight_decay,
        )
        
        trainer.train(
            epochs=args.epochs,
            word2ix=word2ix,
            ix2word=ix2word,
            use_wandb=args.use_wandb,
            resume_path=args.resume,
        )
        
    elif args.mode == 'generate':
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        generated_poem = model.generate(
            initial_text=args.initial_text,
            word2ix=word2ix,
            ix2word=ix2word,
            max_length=100,
            temperature=args.temperature,
            device=args.device
        )
        
        print("\n生成的诗句:")
        print(format_poem_output(generated_poem))
        
    elif args.mode == 'acrostic':
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        acrostic_poem = model.generate_acrostic(
            head_chars=args.head_chars,
            word2ix=word2ix,
            ix2word=ix2word,
            temperature=args.temperature,
            device=args.device
        )
        
        print(f"\n藏头诗 (头字: {args.head_chars}):")
        print(acrostic_poem)

if __name__ == '__main__':
    main()