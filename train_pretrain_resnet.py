import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torch.cuda.amp import GradScaler, autocast
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from data.dataset import FineGrainedDataset
from torch.utils.tensorboard import SummaryWriter

# --------------------- 数据增强 ---------------------
def get_simclr_augmentation():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# --------------------- 投影头 ---------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------------------- SimCLR损失 ---------------------
def nt_xent_loss(features, temperature):
    features = nn.functional.normalize(features, dim=1)
    B = features.size(0) // 2
    f1, f2 = torch.split(features, B, dim=0)
    full = torch.cat([f1, f2], dim=0)
    sim_matrix = torch.matmul(full, full.T) / temperature
    mask = torch.eye(2 * B, device=features.device).bool()
    sim_matrix.masked_fill_(mask, -1e4)
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(features.device)
    numerator = torch.exp(sim_matrix[torch.arange(2 * B), pos_idx])
    denominator = torch.exp(sim_matrix).sum(dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss

# --------------------- 特征提取 + 线性评估 ---------------------
@torch.no_grad()
def extract_features_labels(dataloader, model):
    model.eval()
    features, labels = [], []
    for batch in dataloader:
        if len(batch) == 3:
            imgs, _, lbls = batch
        else:
            imgs, lbls = batch
        imgs = imgs.cuda()
        feats = model(imgs)
        feats = nn.functional.normalize(feats, dim=1)
        features.append(feats.cpu())
        labels.append(lbls)
    return torch.cat(features), torch.cat(labels)

def evaluate_linear(backbone, train_loader, val_loader):
    print("\n🔍 Running linear evaluation...")
    feat_train, label_train = extract_features_labels(train_loader, backbone)
    feat_val, label_val = extract_features_labels(val_loader, backbone)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(feat_train.numpy(), label_train.numpy())
    preds = clf.predict(feat_val.numpy())
    acc = accuracy_score(label_val.numpy(), preds)
    print(f"🎯 Linear eval accuracy: {acc:.4f}")
    return acc

# --------------------- t-SNE 可视化 ---------------------
def visualize_tsne(features, labels, epoch, save_path='checkpoints/tsne_epoch_{:03d}.png'):
    tsne = TSNE(n_components=2, perplexity=10, init='pca', random_state=42)
    reduced = tsne.fit_transform(features.numpy())
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels.numpy(), cmap='tab20', s=5, alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(set(labels.numpy()))))
    plt.title(f"t-SNE at Epoch {epoch}")
    plt.savefig(save_path.format(epoch))
    plt.close()
    print(f"🧿 t-SNE plot saved to {save_path.format(epoch)}")

# --------------------- 主训练函数 ---------------------
def main():
    data_dir = "dataset/CUB_200_2011/images"
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 128
    epochs = 200
    lr = 5e-4
    temperature = 0.2
    eval_interval = 10
    best_acc = 0.0

    writer = SummaryWriter(log_dir=save_dir)

    transform_train = get_simclr_augmentation()
    transform_val = get_val_transform()

    train_dataset = FineGrainedDataset(data_dir, mode='train', known_classes=150, two_views=True, transform=transform_train)
    val_dataset = FineGrainedDataset(data_dir, mode='val', known_classes=150, transform=transform_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    backbone = models.resnet101(weights=None)
    backbone.fc = nn.Identity()
    projection_head = ProjectionHead()
    model = nn.Sequential(backbone, projection_head).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs1, imgs2, _ in train_loader:
            imgs1, imgs2 = imgs1.cuda(), imgs2.cuda()
            inputs = torch.cat([imgs1, imgs2], dim=0)
            optimizer.zero_grad()
            with autocast():
                features = model(inputs)
                loss = nt_xent_loss(features, temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch + 1}] Loss: {avg_loss:.4f}")
        writer.add_scalar("Loss/train", avg_loss, epoch + 1)

        if (epoch + 1) % eval_interval == 0:
            acc = evaluate_linear(backbone, train_loader, val_loader)
            writer.add_scalar("Acc/val", acc, epoch + 1)
            feat_val, label_val = extract_features_labels(val_loader, backbone)
            visualize_tsne(feat_val, label_val, epoch + 1)
            if acc > best_acc:
                best_acc = acc
                torch.save(backbone.state_dict(), os.path.join(save_dir, "backbone_best.pth"))
                print(f"✅ Best model updated at Epoch {epoch + 1} with acc={acc:.4f}")

    torch.save(backbone.state_dict(), os.path.join(save_dir, "backbone_final.pth"))
    writer.close()
    print("✅ Pretraining complete.")

if __name__ == "__main__":
    main()
