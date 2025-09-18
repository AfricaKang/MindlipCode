import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import os
import random
from config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedKnowledgeDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=512, disease_categories=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.disease_categories = disease_categories or config.disease_categories

        self.disease_features = self.data
    
        
        self.classification_data = self._generate_optimized_data()
        
        logger.info(f"Generated {len(self.classification_data)} optimized classification samples")
    
    def _create_optimized_features(self):



        return {}
    
    def _generate_optimized_data(self):



        classification_data = []
        
        for disease, features in self.disease_features.items():
            disease_idx = self.disease_categories.index(disease)
            
            for symptom in features.get("symptoms", []):
                text = f"Patients with {disease} show {symptom}."
                classification_data.append({
                    'text': text,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'symptoms'
                })
                
                variant = f"Typical symptoms for diagnosing {disease} include {symptom}."
                classification_data.append({
                    'text': variant,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'symptoms'
                })
            
            for cognitive in features.get("cognitive", []):
                text = f"{disease} affects cognition, characterized by {cognitive}."
                classification_data.append({
                    'text': text,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'cognitive'
                })
                
                variant = f"Cognitive profile of {disease}: {cognitive}."
                classification_data.append({
                    'text': variant,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'cognitive'
                })
            
            if "behavioral" in features:
                for behavioral in features["behavioral"]:
                    text = f"Behavioral characteristics of {disease}: {behavioral}."
                    classification_data.append({
                        'text': text,
                        'label': disease_idx,
                        'disease': disease,
                        'category': 'behavioral'
                    })
                    
                    variant = f"Observed behaviors in {disease}: {behavioral}."
                    classification_data.append({
                        'text': variant,
                        'label': disease_idx,
                        'disease': disease,
                        'category': 'behavioral'
                    })
            
            for treatment in features.get("treatment", []):
                text = f"Treatments for {disease} include {treatment}."
                classification_data.append({
                    'text': text,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'treatment'
                })
                
                variant = f"Clinical therapies for {disease}: {treatment}."
                classification_data.append({
                    'text': variant,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'treatment'
                })
            
            for unique in features.get("unique", []):
                text = f"Distinctive features of {disease}: {unique}."
                classification_data.append({
                    'text': text,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'unique'
                })
                
                variant = f"What differentiates {disease} from others: {unique}."
                classification_data.append({
                    'text': variant,
                    'label': disease_idx,
                    'disease': disease,
                    'category': 'unique'
                })
        
        classification_data.extend(self._generate_disease_comparison_samples())
        
        return classification_data
    
    def _generate_disease_comparison_samples(self):
        comparison_samples = []
        
        for i, disease1 in enumerate(self.disease_categories):
            for j, disease2 in enumerate(self.disease_categories):
                if i != j:  
                    comparison_texts = self._create_disease_comparison(disease1, disease2)
                    
                    for text in comparison_texts:
                        comparison_samples.append({
                            'text': text,
                            'label': i,  
                            'disease': disease1,
                            'category': 'comparison',
                            'compared_with': disease2
                        })
        
        return comparison_samples
    
    def _create_disease_comparison(self, disease1, disease2):

        comparisons = []
        
        # if disease1 == "ADHD" and disease2 == "Autism":
        #     comparisons.extend([
        #         f"Key difference between {disease1} and {disease2}: {disease1} involves inattention with relatively intact social skills, whereas {disease2} presents social difficulties with focused interests.",
        #         f"Differentiation: {disease1} centers on executive dysfunction, while {disease2} centers on social cognition deficits.",
        #         f"{disease1} vs {disease2}: {disease1} shows hyperactivity-impulsivity; {disease2} shows repetitive, stereotyped behaviors."
        #     ])
        
        # elif 


        return comparisons
    
    def __len__(self):
        return len(self.classification_data)
    
    def __getitem__(self, idx):
    
        item = self.classification_data[idx]
        
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'disease': item['disease'],
            'category': item['category']
        }

class OptimizedKnowledgeModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_classes=8):
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        embedding_dim = 768
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim // 2, embedding_dim // 4),
            nn.LayerNorm(embedding_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim // 4, embedding_dim // 8),
            nn.LayerNorm(embedding_dim // 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 8, num_classes)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim // 4, embedding_dim // 2),
            nn.LayerNorm(embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 2, embedding_dim)
        )
        
        self.pos_margin = 0.5   
        self.neg_margin = 0.2  
        self.temperature = 0.1  
        
    def forward(self, input_ids, attention_mask=None, return_features=False):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        features = self.feature_extractor(embeddings)
        
        if return_features:
            projected_features = self.projection(features)
            return projected_features
        
        logits = self.classifier(features)
        
        return logits
    
    def get_disease_embeddings(self, disease_texts):
        self.eval()
        with torch.no_grad():
            encodings = self.tokenizer(
                disease_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors='pt'
            )
            
            embeddings = self.forward(
                encodings['input_ids'], 
                encodings['attention_mask'], 
                return_features=True
            )
            
            return embeddings
    
    def compute_distance_loss(self, projected_features, labels):
        z = F.normalize(projected_features, p=2, dim=1)
        sim = torch.matmul(z, z.T)  
        batch_size = z.size(0)
        
        labels = labels.view(-1, 1)
        same_mask = (labels == labels.T).float()
        diff_mask = 1.0 - same_mask
        eye = torch.eye(batch_size, device=z.device)
        same_mask = same_mask - eye
        diff_mask = diff_mask
        
        pos_violation = F.relu(self.pos_margin - sim) * same_mask
        neg_violation = F.relu(sim - self.neg_margin) * diff_mask
        
        pos_count = same_mask.sum().clamp(min=1.0)
        neg_count = diff_mask.sum().clamp(min=1.0)
        loss = (pos_violation.sum() / pos_count) + (neg_violation.sum() / neg_count)
        return loss

def train_knowledge_model():
    
    with open(config.knowledge_graph_path, 'r', encoding='utf-8') as f:
        knowledge_data = json.load(f)
    
    
    model = OptimizedKnowledgeModel(num_classes=len(config.disease_categories))
    tokenizer = model.tokenizer
    
    dataset = OptimizedKnowledgeDataset(knowledge_data, tokenizer)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.knowledge_learning_batch_size, 
        shuffle=True,
        num_workers=config.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.knowledge_learning_batch_size, 
        shuffle=False,
        num_workers=config.num_workers
    )
    
    device = torch.device(config.device)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.knowledge_learning_lr,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) * 3,
        num_training_steps=len(train_loader) * config.knowledge_learning_epochs
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.knowledge_learning_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.knowledge_learning_epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            
            enc_feats = model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
            feat_mid = model.feature_extractor(enc_feats)
            features = model.projection(feat_mid)
          

            classification_loss = criterion(logits, labels)
            
            distance_loss = model.compute_distance_loss(features, labels)
            
            total_loss_batch = classification_loss + config.knowledge_distance_loss_weight * distance_loss
        
            
            total_loss_batch.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += total_loss_batch.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'cls_loss': classification_loss.item(),
                'dist_loss': distance_loss.item(),
                'total_loss': total_loss_batch.item(),
                'acc': 100 * correct / total
            })
        
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'best_val_acc': val_acc
            }, os.path.join(config.model_save_path, 'best_knowledge_model.pth'))
            
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info("Early stopping!")
                break
    
    logger.info("Saving final model state.")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'final_val_loss': val_loss,
        'final_val_acc': val_acc
    }, os.path.join(config.model_save_path, 'final_knowledge_model.pth'))

    return model, tokenizer

if __name__ == "__main__":
    train_knowledge_model()