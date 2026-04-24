import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 必须在 import matplotlib.pyplot 之前调用
import matplotlib.pyplot as plt
from matplotlib import rcParams

from PDSDNet import build_model
from dataset import PipelineDataset, get_transforms
from losses import CombinedLoss
from losses import CombinedLoss2

# 设置中文字体支持（可选）
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


def calculate_iou(pred, target, threshold=0.5):
    """计算 IoU"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个 epoch"""
    model.train()
    # 注意: Backbone 保持 eval 模式
    model.backbone.eval()
    #model.vit_backbone.eval()

    running_loss = 0.0
    running_iou = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, masks, edges in pbar:
        images = images.to(device)
        masks = masks.to(device)
        edges = edges.to(device)

        # 前向传播
        mask_pred, edge_pred = model(images)

        # 计算损失
        loss, loss_dict = criterion(mask_pred, edge_pred, masks, edges)

        # 反向传播
        optimizer.zero_grad()
        # 在 train_one_epoch 函数中
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 强制梯度平滑
        optimizer.step()

        # 统计
        running_loss += loss.item()
        running_iou += calculate_iou(mask_pred, masks)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{loss_dict["dice"]:.4f}',
            'iou': f'{running_iou / (pbar.n + 1):.4f}'
        })

    return running_loss / len(dataloader), running_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks, edges in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            edges = edges.to(device)

            mask_pred, edge_pred = model(images)
            loss, _ = criterion(mask_pred, edge_pred, masks, edges)

            running_loss += loss.item()
            running_iou += calculate_iou(mask_pred, masks)

    return running_loss / len(dataloader), running_iou / len(dataloader)


def plot_training_curves(history, save_path):
    """
    绘制训练曲线（损失和精度在同一张图中）
    左侧 y 轴：损失值
    右侧 y 轴：IoU 精度值
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # ========== 左侧 y 轴：损失 ==========
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color='tab:red')

    # 绘制训练损失
    line1 = ax1.plot(epochs, history['train_loss'],
                     color='#FF6B6B', linewidth=2, marker='o',
                     markersize=4, label='Train Loss', alpha=0.8)

    # 绘制验证损失
    line2 = ax1.plot(epochs, history['val_loss'],
                     color='#FF0000', linewidth=2, marker='s',
                     markersize=4, label='Val Loss', alpha=0.8)

    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # ========== 右侧 y 轴：IoU 精度 ==========
    ax2 = ax1.twinx()
    ax2.set_ylabel('IoU', fontsize=12, fontweight='bold', color='tab:blue')

    # 绘制训练 IoU
    line3 = ax2.plot(epochs, history['train_iou'],
                     color='#4ECDC4', linewidth=2, marker='^',
                     markersize=4, label='Train IoU', alpha=0.8)

    # 绘制验证 IoU
    line4 = ax2.plot(epochs, history['val_iou'],
                     color='#0066CC', linewidth=2, marker='D',
                     markersize=4, label='Val IoU', alpha=0.8)

    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # ========== 标注最佳点 ==========
    best_epoch = np.argmax(history['val_iou']) + 1
    best_iou = max(history['val_iou'])

    ax2.scatter([best_epoch], [best_iou],
                color='gold', s=200, marker='*',
                edgecolors='black', linewidths=2,
                zorder=5, label=f'Best (Epoch {best_epoch})')

    ax2.annotate(f'Best IoU: {best_iou:.4f}\nEpoch: {best_epoch}',
                 xy=(best_epoch, best_iou),
                 xytext=(10, -30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 color='black', lw=1.5),
                 fontsize=10, fontweight='bold')

    # ========== 图例 ==========
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)

    # ========== 标题与布局 ==========
    plt.title('Training and Validation Curves',
              fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存至: {save_path}")
    plt.close()


def main():
    # ========== 配置 ==========

    CONFIG = {
        'data_root': r'E:\ChemistryPark\pipeline dataset\pipeline_dataset',  # 修改为你的数据路径
        'dinov3_path': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',  # 修改为你的 DINOv3 权重路径
        'batch_size': 32,
        'num_epochs': 100,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'num_workers': 16,
        'save_dir': './checkpoints_PDSDNet'
    }


    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========== 数据加载 ==========
    train_dataset = PipelineDataset(
        image_dir=os.path.join(CONFIG['data_root'], 'train/images'),
        mask_dir=os.path.join(CONFIG['data_root'], 'train/labels'),
        transform=get_transforms(is_train=True),
        augment=True
    )

    val_dataset = PipelineDataset(
        image_dir=os.path.join(CONFIG['data_root'], 'test/images'),
        mask_dir=os.path.join(CONFIG['data_root'], 'test/labels'),
        transform=get_transforms(is_train=False),
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")

    # ========== 模型构建 ==========
    model = build_model(CONFIG['dinov3_path'])
    #model = build_model(CONFIG)
    model = model.to(device)

    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")

    # ========== 优化器与损失 ==========
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    #scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    from torch.optim.lr_scheduler import OneCycleLR
    #预热余弦退火
    scheduler = OneCycleLR(optimizer, max_lr=CONFIG['lr'],
                           steps_per_epoch=len(train_loader),
                           epochs=CONFIG['num_epochs'])
    criterion = CombinedLoss(alpha=1.0, beta=1.0, gamma=0.5)

    # ========== 训练历史记录 ==========
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }

    best_iou = 0.0
    best_epoch = 0

    # ========== 训练循环 ==========
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        val_loss, val_iou = validate(model, val_loader, criterion, device)

        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # ========== 保存最佳模型 ==========
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch

            best_model_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'best_iou': best_iou,
                'history': history
            }, best_model_path)

            print(f"✓ 保存最佳模型 (Epoch {epoch}, Val IoU: {val_iou:.4f})")

        print(f"当前最佳: Epoch {best_epoch}, IoU: {best_iou:.4f}\n")

        # ========== 每 10 个 epoch 绘制一次曲线 ==========
        if epoch % 5 == 0 or epoch == CONFIG['num_epochs']:
            plot_path = os.path.join(CONFIG['save_dir'], f'training_curve_epoch_{epoch}.png')
            plot_training_curves(history, plot_path)

    # ========== 保存最后模型 ==========
    last_model_path = os.path.join(CONFIG['save_dir'], 'last_model.pth')
    torch.save({
        'epoch': CONFIG['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_iou': train_iou,
        'val_loss': val_loss,
        'val_iou': val_iou,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'history': history
    }, last_model_path)

    # ========== 绘制最终训练曲线 ==========
    final_plot_path = os.path.join(CONFIG['save_dir'], 'final_training_curve.png')
    plot_training_curves(history, final_plot_path)

    print(f"\n{'=' * 60}")
    print(f"训练完成！")
    print(f"最佳模型: Epoch {best_epoch}, Val IoU: {best_iou:.4f}")
    print(f"最后模型: Epoch {CONFIG['num_epochs']}, Val IoU: {val_iou:.4f}")
    print(f"模型保存路径: {CONFIG['save_dir']}")
    print(f"训练曲线: {final_plot_path}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
