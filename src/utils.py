"""
Utility functions for the AI Paper Summarizer & Q&A System
"""

import os
import json
import yaml
import logging
import torch
import PyPDF2
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
import numpy as np
from tqdm import tqdm


def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        log_dir: Directory to save log files
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(
        log_dir, 
        f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def save_config(config: Dict, save_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(config: Dict):
    """
    Create necessary directories from config
    
    Args:
        config: Configuration dictionary
    """
    directories = [
        config['paths']['output_dir'],
        config['paths']['model_save_dir'],
        config['paths']['logs_dir'],
        config['paths']['cache_dir'],
        "./data/raw",
        "./data/processed",
        "./data/datasets"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text content from PDF file
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
            return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def chunk_text(text: str, max_length: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        max_length: Maximum chunk length
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), max_length - overlap):
        chunk = ' '.join(words[i:i + max_length])
        chunks.append(chunk)
        
    return chunks


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters but keep punctuation
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text.strip()


def get_device() -> torch.device:
    """
    Get the best available device (GPU/CPU)
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
        
    return device


def save_json(data: Union[Dict, List], filepath: str):
    """Save data to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Union[Dict, List]:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_metrics(predictions: List[str], references: List[str], 
                   rouge_scorer) -> Dict[str, float]:
    """
    Compute evaluation metrics for summarization
    
    Args:
        predictions: Model predictions
        references: Ground truth references
        rouge_scorer: ROUGE scorer object
        
    Returns:
        Dictionary of metric scores
    """
    scores = {
        'rouge1': [],
        'rouge2': [],
        'rougeL': []
    }
    
    for pred, ref in zip(predictions, references):
        rouge_scores = rouge_scorer.score(ref, pred)
        scores['rouge1'].append(rouge_scores['rouge1'].fmeasure)
        scores['rouge2'].append(rouge_scores['rouge2'].fmeasure)
        scores['rougeL'].append(rouge_scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL'])
    }


def format_time(seconds: float) -> str:
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class ProgressTracker:
    """Track training progress with nice formatting"""
    
    def __init__(self, total_steps: int, description: str = "Training"):
        self.pbar = tqdm(total=total_steps, desc=description)
        self.metrics = {}
        
    def update(self, metrics: Dict[str, float]):
        """Update progress bar with new metrics"""
        self.metrics.update(metrics)
        self.pbar.set_postfix(self.metrics)
        self.pbar.update(1)
        
    def close(self):
        """Close progress bar"""
        self.pbar.close()


def estimate_memory_usage(model) -> str:
    """
    Estimate model memory usage
    
    Args:
        model: PyTorch model
        
    Returns:
        Formatted memory string
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return f"{total_size / 1e6:.2f} MB"


def clear_cuda_cache():
    """Clear CUDA cache to free up memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero"""
    return numerator / denominator if denominator != 0 else default


class ModelCheckpoint:
    """Handle model checkpointing"""
    
    def __init__(self, save_dir: str, metric_name: str = "loss", mode: str = "min"):
        self.save_dir = save_dir
        self.metric_name = metric_name
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        os.makedirs(save_dir, exist_ok=True)
        
    def is_better(self, score: float) -> bool:
        """Check if current score is better than best score"""
        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score
            
    def save(self, model, tokenizer, score: float, epoch: int):
        """Save model if score improved"""
        if self.is_better(score):
            self.best_score = score
            save_path = os.path.join(self.save_dir, "best_model")
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            
            # Save metadata
            metadata = {
                'epoch': epoch,
                'best_score': score,
                'metric_name': self.metric_name
            }
            save_json(metadata, os.path.join(save_path, "metadata.json"))
            
            return True
        return False
