#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import timm
import torchvision.transforms as transforms
import os
import logging
from config import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGSpatialTemporalEncoder(nn.Module):
   
    def __init__(self, output_dim=512):
        super().__init__()
        
       
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),   # 32x256 -> 16x128
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 16x128 -> 8x64
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 8x64 -> 4x32
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 4x32 -> 2x16
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        
        self.feature_projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)   
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)           # (batch, 256, 1, 1)
        x = x.flatten(1)                  # (batch, 256)
        output = self.feature_projection(x)  # (batch, 512)
        return output

class EEGOptimizedCLIPModel(nn.Module):
    
    def __init__(self, projection_dim=512):
        super().__init__()
        
        
        self.image_encoder = EEGSpatialTemporalEncoder(output_dim=projection_dim)
        
        
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.text_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        
        
        self.load_knowledge_enhanced_encoder()
        
        
        self.text_projection = nn.Sequential(
            nn.Linear(768, projection_dim * 2),
            nn.LayerNorm(projection_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim * 2, projection_dim)
        )
        
        
        self.temperature = nn.Parameter(torch.tensor(0.15), requires_grad=True)
        
        
        self.min_temperature = 0.08  
        self.max_temperature = 0.4   
        
        self._freeze_backbone()
    
    def load_knowledge_enhanced_encoder(self):
        knowledge_model_path = os.path.join(config.model_save_path, 'best_knowledge_model.pth')
        
        if os.path.exists(knowledge_model_path):
           
                checkpoint = torch.load(knowledge_model_path, map_location='cpu')
                
                
                knowledge_encoder_weights = {}
                for key, value in checkpoint['model_state_dict'].items():
                    if key.startswith('encoder.'):
                        new_key = key[8:]
                        knowledge_encoder_weights[new_key] = value
                
                
                missing_keys, unexpected_keys = self.text_encoder.load_state_dict(
                    knowledge_encoder_weights, strict=False
                )
       
    def _freeze_backbone(self):
       
        for name, param in self.text_encoder.named_parameters():
            if 'embeddings' in name or 'encoder.layer.0' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        self.log_trainable_params()
    
    def forward(self, images, texts):
       
        image_features = self.image_encoder(images)
        
        encodings = self.text_tokenizer(
            texts, truncation=True, padding=True, 
            max_length=config.text_max_length, return_tensors='pt'
        )
        device = next(self.parameters()).device
        encodings = {k: v.to(device) for k, v in encodings.items()}
        
        text_outputs = self.text_encoder(**encodings)
        text_features = text_outputs.last_hidden_state[:, 0, :]
        text_proj = self.text_projection(text_features)
        
        return {
            'image_features': image_features,
            'text_features': text_proj
        }
    
    def contrastive_loss(self, outputs):
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        constrained_temperature = self.temperature.clamp(self.min_temperature, self.max_temperature)
        similarity_matrix = torch.matmul(image_features, text_features.T) / constrained_temperature
        
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size, device=similarity_matrix.device)
        
        classification_loss = F.cross_entropy(similarity_matrix, labels)
        
        positive_sims = torch.diag(similarity_matrix)
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        negative_sims = similarity_matrix[negative_mask]
        
        current_separation = positive_sims.mean() - negative_sims.mean()
        
        
        separation_target = config.clip_separation_target
        separation_loss = F.relu(separation_target - current_separation)
        
        total_loss = classification_loss + config.clip_separation_loss_weight * separation_loss 
        
        return total_loss
    
    def get_normalized_features(self, outputs):
        image_features = outputs['image_features']
        text_features = outputs['text_features']
        
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        return image_features, text_features
    
    def compute_similarity_matrix(self, outputs):
        image_features, text_features = self.get_normalized_features(outputs)
        
        similarity_matrix = torch.matmul(image_features, text_features.T)
        
        return similarity_matrix, image_features, text_features
    
    def validate_clip_architecture(self, outputs):
        batch_size = outputs['image_features'].size(0)
        
        image_features, text_features = self.get_normalized_features(outputs)
        
        
        image_norms = torch.norm(image_features, p=2, dim=1)
        text_norms = torch.norm(text_features, p=2, dim=1)
        
        similarity_matrix = torch.matmul(image_features, text_features.T)
        
        min_sim = similarity_matrix.min().item()
        max_sim = similarity_matrix.max().item()
        
        positive_sims = torch.diag(similarity_matrix)
        negative_mask = ~torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix.device)
        negative_sims = similarity_matrix[negative_mask]
        
        separation = positive_sims.mean() - negative_sims.mean()
        
        return {
            'feature_dim': image_features.size(1),
            'image_norm_mean': image_norms.mean().item(),
            'text_norm_mean': text_norms.mean().item(),
            'similarity_range': [min_sim, max_sim],
            'separation': separation.item()
        }
    
    def log_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Parameters for training: {trainable:,} / Total: {total:,} ({trainable/total*100:.1f}%)")

def get_eeg_transforms(is_train=True):

    if is_train:
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.RandomHorizontalFlip(p=0.1), 
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(), 
        ])

if __name__ == "__main__":
    
    model = EEGOptimizedCLIPModel()
    
    batch_size = 4
    test_images = torch.randn(batch_size, 3, 32, 256)  
    test_texts = [      ]
    
    outputs = model(test_images, test_texts)
    
    loss = model.contrastive_loss(outputs)
    print(f"Constractive loss: {loss.item():.4f}")
    
    norm_img_feat, norm_txt_feat = model.get_normalized_features(outputs)
    print("✓ Finished！")