import os
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    # Knowledge Graph Learning Configuration
    knowledge_learning_epochs: int = 50
    knowledge_learning_lr: float = 3e-6   
    knowledge_learning_batch_size: int = 2  
    knowledge_embedding_dim: int = 768
    knowledge_margin: float = 0.3
    knowledge_temperature: float = 0.07
    knowledge_distance_loss_weight: float = 0.2
    
    # CLIP Model Configuration
    clip_epochs: int = 100
    clip_lr: float = 5e-5   
    clip_batch_size: int = 64  
    clip_temperature: float = 0.07  
    clip_projection_dim: int = 512
    clip_patience: int = 5

    clip_separation_target: float = 0.5
    clip_separation_loss_weight: float = 0.05
    
    num_workers: int = 2  
    
    # GPU Optimization
    use_amp: bool = True
    
    # Model Training Configuration
    freeze_backbone: bool = False  
    unfreeze_last_layers: int = 0 
    progressive_unfreeze: bool = False  
    
    # Small Dataset Optimization
    dropout_rate: float = 0.1  
    weight_decay: float = 0.0001  
    gradient_accumulation_steps: int = 1  
    
    # EEG Image Configuration 
    image_height: int = 32   
    image_width: int = 256    
    image_channels: int = 3   
    image_patch_size: int = 8 
    image_encoder: str = "coeegen"  
    
    # Text Encoder Configuration
    text_max_length: int = 256  
    
    # Training Configuration
    num_folds: int = 5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Path Configuration
    dataset_path: str = "dataset"
    knowledge_graph_path: str = "dsm5KG.json"
    model_save_path: str = "models"
    results_save_path: str = "results"
    
    # Data Augmentation
    image_augmentation: bool = True
    
    # Disease Categories
    disease_categories: list = None
    
    def __post_init__(self):
        if self.disease_categories is None:
            self.disease_categories = [
            #    "ADHD", "Autism"......
            ]
        
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(self.results_save_path, exist_ok=True)

config = ModelConfig()