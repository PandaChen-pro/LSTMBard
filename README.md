# LSTM Bard(LSTM自动写诗)
## 使用方法
训练模型：
```shell
python start.py --mode train --data_path ./data/tang.npz --use_wandb --seed 3407  --dropout 0.55 --epochs 80
```
继续训练：
```shell
python start.py --mode train --data_path ./data/tang.npz --use_wandb --seed 3407  --dropout 0.55 --resume ./checkpoints/final_model.pth --epochs 30
```

生成诗句（给定首句）：
```shell
python start.py --mode generate --model_path checkpoints/best_model.pth --initial_text "湖光秋月两相和"
```
生成藏头诗：
```shell
python start.py --mode acrostic --model_path checkpoints/best_model.pth --head_chars "春夏秋冬"
```