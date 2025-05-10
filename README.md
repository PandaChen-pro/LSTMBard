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

模型下载链接：https://drive.google.com/file/d/1G7O29MuobwzGyRKl7hFUtMP1skapYNgQ/view?usp=sharing

生成诗句（给定首句）：
```shell
python start.py --mode generate --model_path checkpoints/best_model.pth --initial_text "湖光秋月两相和"
```
输出：
```shell
湖光秋月两相和。
夜风吹前香花雪，
黄户照茅露下出。
水树生迟如雨雷，
南山蔽日若参差。
```