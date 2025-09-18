#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import os
import glob
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from config import config
import logging
from eeg_optimized_model import EEGOptimizedCLIPModel, get_eeg_transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EEGDataset(Dataset):
    
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        
        self.classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_path = os.path.join(data_path, class_name)
            image_files = glob.glob(os.path.join(class_path, "*.png"))
            
            for image_file in image_files:
                self.images.append(image_file)
                self.labels.append(self.class_to_idx[class_name])
        
        
        self.descriptions = {
            # e.g., "ADHD": "A detailed clinical description of Attention-Deficit/Hyperactivity Disorder...",
            # e.g., "Autism": "A detailed clinical description of Autism Spectrum Disorder...",
            # TODO: Populate this dictionary with your actual class descriptions.
        }

        
        logger.info(f"load {len(self.images)} converted EEG images，{len(self.classes)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        class_name = self.classes[label]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        
        if not self.descriptions or class_name not in self.descriptions:
            raise ValueError(f"Description for class '{class_name}' not found. "
                             f"Please populate the self.descriptions dictionary in EEGDataset.")

        text = self.descriptions[class_name]
        
        return {'image': image, 'text': text, 'label': label}

def train_eeg_model():
 
    device = torch.device(config.device)
    
    train_dataset = EEGDataset(config.dataset_path, transform=get_eeg_transforms(is_train=True))
    val_dataset = EEGDataset(config.dataset_path, transform=get_eeg_transforms(is_train=False))
    
    class_texts = [train_dataset.descriptions[c] for c in train_dataset.classes]
   
    
    def compute_class_text_features(current_model, texts, device):
        current_model.eval()
        with torch.no_grad():
            enc = current_model.text_tokenizer(
                texts, truncation=True, padding=True,
                max_length=config.text_max_length, return_tensors='pt'
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            text_out = current_model.text_encoder(**enc)
            text_feat = text_out.last_hidden_state[:, 0, :]
            text_proj = current_model.text_projection(text_feat)
            text_proj = torch.nn.functional.normalize(text_proj, p=2, dim=1)
        current_model.train()
        return text_proj
    
    def get_unique_class_batch_texts(batch_labels, class_texts):
        unique_texts = []
        for label in batch_labels:
            unique_texts.append(class_texts[label.item()])
        return unique_texts
    
    def create_balanced_sampler(labels):
        from torch.utils.data import WeightedRandomSampler
        
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
        return sampler
 
    all_labels = train_dataset.labels
    indices = range(len(train_dataset))
    
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=config.seed,
        stratify=all_labels
    )
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    train_subset_labels = [all_labels[i] for i in train_idx]
    train_sampler = create_balanced_sampler(train_subset_labels)
    
    train_loader = DataLoader(
        train_subset, batch_size=config.clip_batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.clip_batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    model = EEGOptimizedCLIPModel().to(device)
    
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.clip_lr,
        weight_decay=config.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.clip_epochs, eta_min=config.clip_lr * 0.1 
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 25  
    min_delta = 0.0005  
    
    for epoch in range(config.clip_epochs):
        model.train()
        total_loss = 0
        total_matches = 0
        correct_matches = 0
        class_text_features_train = compute_class_text_features(model, class_texts, device)
        train_true_labels, train_pred_labels = [], []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.clip_epochs}"):
            images = batch['image'].to(device)
            texts = get_unique_class_batch_texts(batch['label'], class_texts)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images, texts)
            loss = model.contrastive_loss(outputs)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            with torch.no_grad():
                norm_img_feat, norm_txt_feat = model.get_normalized_features(outputs)
                temp = model.temperature.clamp(model.min_temperature, model.max_temperature)
                similarity_matrix = torch.matmul(norm_img_feat, norm_txt_feat.T) / temp
                predicted_matches = torch.argmax(similarity_matrix, dim=1)
                correct_matches += (predicted_matches == torch.arange(len(labels), device=device)).sum().item()
                total_matches += len(labels)

                class_sim = torch.matmul(norm_img_feat, class_text_features_train.T) / temp
                class_pred = torch.argmax(class_sim, dim=1)
                train_true_labels.extend(labels.detach().cpu().tolist())
                train_pred_labels.extend(class_pred.detach().cpu().tolist())
            
            total_loss += loss.item()
        
        scheduler.step()
        
        model.eval()
        val_loss = 0
        val_similarities = {'positive': [], 'negative': []}
        class_text_features_val = compute_class_text_features(model, class_texts, device)
        val_true_labels, val_pred_labels = [], []
        
        with torch.no_grad():
            
            for batch in val_loader:
                images = batch['image'].to(device)
                texts = get_unique_class_batch_texts(batch['label'], class_texts)
                labels = batch['label'].to(device)
                
                outputs = model(images, texts)
                loss = model.contrastive_loss(outputs)
                val_loss += loss.item()
                
                norm_img_feat, norm_txt_feat = model.get_normalized_features(outputs)
                temp = model.temperature.clamp(model.min_temperature, model.max_temperature)
                similarity_matrix = torch.matmul(norm_img_feat, norm_txt_feat.T) / temp
                
                for i in range(len(labels)):
                    for j in range(len(labels)):
                        if i == j:
                            val_similarities['positive'].append(similarity_matrix[i, j].item())
                        else:
                            val_similarities['negative'].append(similarity_matrix[i, j].item())

                class_sim = torch.matmul(norm_img_feat, class_text_features_val.T) / temp
                class_pred = torch.argmax(class_sim, dim=1)
                val_true_labels.extend(labels.detach().cpu().tolist())
                val_pred_labels.extend(class_pred.detach().cpu().tolist())
        
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_matches / total_matches
        train_class_acc = accuracy_score(train_true_labels, train_pred_labels) if train_true_labels else 0.0
        train_f1_macro = f1_score(train_true_labels, train_pred_labels, average='macro', zero_division=0) if train_true_labels else 0.0
        val_loss = val_loss / len(val_loader)
        val_class_acc = accuracy_score(val_true_labels, val_pred_labels) if val_true_labels else 0.0
        val_f1_macro = f1_score(val_true_labels, val_pred_labels, average='macro', zero_division=0) if val_true_labels else 0.0
        
        avg_positive_sim = np.mean(val_similarities['positive']) if val_similarities['positive'] else 0.0
        avg_negative_sim = np.mean(val_similarities['negative']) if val_similarities['negative'] else 0.0
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train PairAcc: {train_accuracy:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Temperature: {model.temperature.item():.4f}")
        logger.info(f"  Train accuracy : {train_class_acc:.4f}, Train F1(macro): {train_f1_macro:.4f}")
        logger.info(f"  Val accuracy : {val_class_acc:.4f}, Val F1(macro): {val_f1_macro:.4f}")
                
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"  Learning rate: {current_lr:.2e}")
        
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, os.path.join(config.model_save_path, 'best_eeg_optimized_model.pth'))
            logger.info(f"✓ save best models, val_loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"early stopping！")
                break
    
    logger.info("V-L training finished!")

if __name__ == "__main__":
    train_eeg_model()