import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


class PoemTrainer:
    def __init__(self, model, train_loader, val_loader=None, 
                learning_rate=0.001, device='cuda', pad_idx=8292,weight_decay=1e-5,
                t_max=80, eta_min=1e-5):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.pad_idx = pad_idx
        # 使用交叉熵损失，忽略填充标记
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)

        
    def train(self, epochs, word2ix, ix2word, save_path='checkpoints',
             log_interval=10, save_interval=10, use_wandb=True, resume_path=None):
        """训练模型"""
        wandb_run_id = None
        if resume_path:
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if 'wandb_run_id' in checkpoint:
                wandb_run_id = checkpoint['wandb_run_id']
            
            print(f"Resuming training from {resume_path}")
        # 将模型移动到设备
        self.model.to(self.device)
        if use_wandb:
            wandb_config = {
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "epochs": epochs,
                "model_type": self.model.__class__.__name__,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "vocab_size": self.model.vocab_size
            }
            if wandb_run_id:
                wandb.init(project="lstm-bard", id=wandb_run_id, resume="must", config=wandb_config)
                print(f"Resuming wandb run with id: {wandb_run_id}")
            else:
                wandb.init(project="lstm-bard", config=wandb_config)
            
            wandb.watch(self.model, log_freq=100)

            
        os.makedirs(save_path, exist_ok=True)
        
        best_val_loss = float('inf')
        
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            start_time = time.time()
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (data, target) in enumerate(progress_bar):
                data, target = data.to(self.device), target.to(self.device)
                
                # 梯度归零
                self.optimizer.zero_grad()
                
                # 前向传播
                output, _ = self.model(data)
                
                # 调整形状以适应损失函数
                loss = self.criterion(output.reshape(-1, self.model.vocab_size), target.reshape(-1))
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                
                # 更新参数
                self.optimizer.step()
                
                total_train_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                # 记录到wandb
                if use_wandb and batch_idx % log_interval == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx + epoch * len(self.train_loader)
                    })
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_train_loss:.4f}, Time: {elapsed:.2f}s")
            
            current_val_loss = float('inf')

            # 验证
            if self.val_loader:
                current_val_loss = self.evaluate()
                print(f"Validation Loss: {current_val_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "val_loss": current_val_loss,
                        "train_loss": avg_train_loss,
                        "epoch": epoch,
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })
                    
                if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        self.save_model(os.path.join(save_path, 'best_model.pth'))
                        epochs_no_improve = 0 # 重置计数器
                        print(f"Validation loss improved. Saved best model.")
                else:
                    epochs_no_improve += 1
                    print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")

            else: # 如果没有验证集，按间隔保存
                if (epoch + 1) % save_interval == 0:
                    self.save_model(os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))

            # 每个epoch后更新学习率
            self.scheduler.step()
            if use_wandb:
                wandb.log({"learning_rate": self.optimizer.param_groups[0]['lr'], "epoch": epoch})

            # 生成示例诗句 (可以在验证后进行)
            if use_wandb and epoch % log_interval == 0:
                try: # 添加 try-except 以防生成失败
                    example_poem = self.model.generate("湖光秋月两相和", word2ix, ix2word,
                                                    max_length=60, device=self.device, temperature=0.8) # 降低一点温度和长度
                    wandb.log({"example_poem": wandb.Html(f"<pre>{example_poem.encode('utf-8').decode('utf-8')}</pre>"), "epoch": epoch}) # 使用 pre 标签保持格式
                    os.makedirs(os.path.join(save_path, 'poems'), exist_ok=True)
                    # 使用 epoch+1 作为文件名
                    with open(os.path.join(save_path, f'./poems/example_poem_epoch_{epoch+1}.txt'), 'w', encoding='utf-8') as f:
                        f.write(example_poem)
                    
                except Exception as e:
                    print(f"Error generating example poem: {e}")
        
        # 保存最终模型
        self.save_model(os.path.join(save_path, 'final_model.pth'))
        
        if use_wandb:
            wandb.finish()
    
    def evaluate(self):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                loss = self.criterion(output.reshape(-1, self.model.vocab_size), target.reshape(-1))
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_model(self, path):
        """保存模型"""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),  # 保存调度器状态
        }
        
        # 如果 wandb 已初始化，保存 run ID
        if wandb.run is not None:
            save_dict['wandb_run_id'] = wandb.run.id
        
        torch.save(save_dict, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        wandb_run_id = None
        if 'wandb_run_id' in checkpoint:
            wandb_run_id = checkpoint['wandb_run_id']
        
        print(f"Model loaded from {path}")
        return wandb_run_id