import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm

class PoemTrainer:
    def __init__(self, model, train_loader, val_loader=None, 
                 learning_rate=0.001, device='cuda'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
    def train(self, epochs, char_to_idx, idx_to_char, save_path='checkpoints',
              log_interval=10, save_interval=1, use_wandb=True):
        """训练模型"""
        if use_wandb:
            wandb.init(project="lstm-poem-generator", config={
                "learning_rate": self.optimizer.param_groups[0]['lr'],
                "epochs": epochs,
                "model_type": self.model.__class__.__name__,
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_layers": self.model.num_layers,
                "vocab_size": self.model.vocab_size
            })
            
        os.makedirs(save_path, exist_ok=True)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
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
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
                
                # 记录到wandb
                if use_wandb and batch_idx % log_interval == 0:
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch": batch_idx + epoch * len(self.train_loader)
                    })
            
            avg_loss = total_loss / len(self.train_loader)
            elapsed = time.time() - start_time
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {elapsed:.2f}s")
            
            # 验证
            if self.val_loader:
                val_loss = self.evaluate()
                print(f"Validation Loss: {val_loss:.4f}")
                if use_wandb:
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch
                    })
                    
                # 如果是最佳模型，保存它
                if val_loss < best_loss:
                    best_loss = val_loss
                    self.save_model(os.path.join(save_path, 'best_model.pth'))
            
            # 记录到wandb
            if use_wandb:
                wandb.log({
                    "train_loss": avg_loss,
                    "epoch": epoch
                })
                
                # 生成示例诗句
                if epoch % log_interval == 0:
                    example_poem = self.model.generate("湖光秋月两相和", char_to_idx, idx_to_char, 
                                                      max_length=100, device=self.device)
                    wandb.log({
                        "example_poem": example_poem,
                        "epoch": epoch
                    })
            
            # 定期保存模型
            if (epoch + 1) % save_interval == 0:
                self.save_model(os.path.join(save_path, f'model_epoch_{epoch+1}.pth'))
        
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
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")