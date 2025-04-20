# models/finegrained_model.py
import torch
import torch.nn as nn
from torchvision import models

class FineGrainedModel(nn.Module):
    def __init__(self, num_classes=150, backbone_name='resnet50', pretrained_backbone_path=None, attention_dim=1024):
        super().__init__()
        # Load a ResNet backbone
        assert backbone_name == 'resnet50', "Only resnet50 backbone implemented"
        backbone = models.resnet50(pretrained=False)
        if pretrained_backbone_path:
            # Load pretrained weights (from SimCLR)
            backbone.load_state_dict(torch.load(pretrained_backbone_path, map_location='cpu'))
        # We will use layers until layer3 and layer4 separately
        # Keep a copy of layer4 separate, because we will forward layer3 output into it
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # 64 -> 256
        self.layer2 = backbone.layer2  # 256 -> 512
        self.layer3 = backbone.layer3  # 512 -> 1024 (this will be our local feature map)
        self.layer4 = backbone.layer4  # 1024 -> 2048
        self.avgpool = backbone.avgpool  # global average pool (7x7 -> 1x1)
        # Note: backbone.fc is not used (we'll create our own classifier)

        # Semantic-guided channel attention module:
        # We use global feature (2048-D) to produce attention weights for local feature (1024-D).
        self.attention_fc = nn.Linear(2048, attention_dim)  # project global feat to same dim as local channels (1024)
        self.attention_relu = nn.ReLU()
        self.attention_sigmoid = nn.Sigmoid()
        if attention_dim != 1024:
            # If dimension differs, another linear to map to 1024 channels
            self.attention_out = nn.Linear(attention_dim, 1024)
        else:
            self.attention_out = nn.Identity()

        # Classifier for known classes
        self.classifier = nn.Linear(2048 + 1024, num_classes)  # taking concatenated global+local vector

    def forward(self, x):
        # Backbone forward
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        local_map = self.layer3(x)        # feature map of shape [B, 1024, H_l, W_l] (e.g., 14x14)
        x = self.layer4(local_map)        # continue to final layer
        global_map = x                   # [B, 2048, H_g, W_g] (e.g., 7x7)
        # Global feature vector
        glob_feat = self.avgpool(global_map)           # [B, 2048, 1, 1]
        glob_feat = glob_feat.view(glob_feat.size(0), -1)  # [B, 2048]
        # Local feature vector with attention:
        # Compute channel attention from global feature
        att = self.attention_fc(glob_feat)            # [B, attention_dim]
        att = self.attention_relu(att)
        att = self.attention_out(att)                 # [B, 1024] if attention_dim != 1024, otherwise identity
        att_weights = self.attention_sigmoid(att)     # [B, 1024] in range [0,1]
        # Apply channel weights to local feature map
        # Expand att_weights to [B, 1024, H_l, W_l] for multiplication
        att_weights = att_weights.unsqueeze(-1).unsqueeze(-1)  # [B, 1024, 1, 1]
        att_local_map = local_map * att_weights       # weighted local features
        # Pool local map to vector
        local_feat = nn.functional.adaptive_avg_pool2d(att_local_map, (1, 1))
        local_feat = local_feat.view(local_feat.size(0), -1)   # [B, 1024]
        # Concatenate global and local features
        combined_feat = torch.cat([glob_feat, local_feat], dim=1)  # [B, 2048+1024]
        # Class logits for known classes
        logits = self.classifier(combined_feat)       # [B, num_classes]
        # Energy score for open set (negative log sum exp)
        # Using negative sign because lower energy indicates in-distribution
        energy = -torch.logsumexp(logits, dim=1)      # [B]
        return logits, energy
