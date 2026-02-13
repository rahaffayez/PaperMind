"""
Model training module for AI Paper Summarizer & Q&A System
Handles fine-tuning of summarization and Q&A models
"""

import os
import torch
from typing import Dict, Optional
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    default_data_collator
)
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from rouge_score import rouge_scorer

from utils import (
    setup_logging, load_config, get_device, 
    create_directories, ModelCheckpoint, set_seed,
    clear_cuda_cache, estimate_memory_usage
)
from data_preparation import tokenize_summarization_dataset, tokenize_qa_dataset


logger = setup_logging()


class SummarizationTrainer:
    """Trainer for summarization model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['model']['summarizer']
        self.train_config = config['training']['summarizer']
        self.lora_config_dict = config['model']['lora']
        self.device = get_device()
        
        set_seed(42)
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading model: {self.model_config['base_model']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model'],
            cache_dir=self.config['paths']['cache_dir']
        )
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_config['base_model'],
            cache_dir=self.config['paths']['cache_dir']
        )
        
        # Configure LoRA for efficient fine-tuning
        lora_config = LoraConfig(
            r=self.lora_config_dict['r'],
            lora_alpha=self.lora_config_dict['lora_alpha'],
            target_modules=["q_proj", "v_proj"],  # For BART/T5 models
            lora_dropout=self.lora_config_dict['lora_dropout'],
            bias=self.lora_config_dict['bias'],
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info(f"Model memory: {estimate_memory_usage(self.model)}")
        
    def load_and_tokenize_data(self):
        """Load and tokenize dataset"""
        logger.info("Loading summarization dataset...")
        
        try:
            # Load prepared dataset
            dataset = load_from_disk("./data/datasets/summarization")
        except:
            logger.error("Dataset not found. Run data_preparation.py first!")
            raise
        
        # Tokenize dataset
        self.tokenized_dataset = tokenize_summarization_dataset(
            dataset,
            self.tokenizer,
            max_input_length=self.config['data']['max_input_length'],
            max_target_length=self.config['data']['max_target_length']
        )
        
        logger.info(f"Training samples: {len(self.tokenized_dataset['train'])}")
        logger.info(f"Validation samples: {len(self.tokenized_dataset['validation'])}")
        
    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(
            predictions, 
            skip_special_tokens=True
        )
        
        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels, 
            skip_special_tokens=True
        )
        
        # Compute ROUGE scores
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = scorer.score(label, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    
    def train(self):
        """Train the summarization model"""
        logger.info("Starting summarization model training...")
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(self.config['paths']['output_dir'], 'summarizer'),
            num_train_epochs=self.train_config['num_epochs'],
            per_device_train_batch_size=self.train_config['batch_size'],
            per_device_eval_batch_size=self.train_config['batch_size'],
            gradient_accumulation_steps=self.train_config['gradient_accumulation_steps'],
            learning_rate=self.train_config['learning_rate'],
            warmup_steps=self.train_config['warmup_steps'],
            weight_decay=self.train_config['weight_decay'],
            max_grad_norm=self.train_config['max_grad_norm'],
            fp16=self.train_config['fp16'] and torch.cuda.is_available(),
            logging_dir=self.config['paths']['logs_dir'],
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL",
            greater_is_better=True,
            predict_with_generate=True,
            generation_max_length=self.model_config['max_length'],
            generation_num_beams=self.model_config['num_beams'],
            report_to="none",
            remove_unused_columns=True,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        try:
            train_result = trainer.train()
            
            # Save final model
            save_path = os.path.join(
                self.config['paths']['model_save_dir'], 
                'summarizer_final'
            )
            trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Model saved to {save_path}")
            logger.info(f"Training metrics: {train_result.metrics}")
            
            # Evaluate
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            clear_cuda_cache()


class QATrainer:
    """Trainer for Q&A model"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['model']['qa']
        self.train_config = config['training']['qa']
        self.device = get_device()
        
        set_seed(42)
        
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        logger.info(f"Loading Q&A model: {self.model_config['base_model']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['base_model'],
            cache_dir=self.config['paths']['cache_dir']
        )
        
        # Load model
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            self.model_config['base_model'],
            cache_dir=self.config['paths']['cache_dir']
        )
        
        logger.info(f"Model memory: {estimate_memory_usage(self.model)}")
        
    def load_and_tokenize_data(self):
        """Load and tokenize Q&A dataset"""
        logger.info("Loading Q&A dataset...")
        
        try:
            # Load prepared dataset
            dataset = load_from_disk("./data/datasets/qa")
        except:
            logger.error("Q&A dataset not found. Run data_preparation.py first!")
            raise
        
        # Tokenize dataset
        self.tokenized_dataset = tokenize_qa_dataset(
            dataset,
            self.tokenizer,
            max_length=self.model_config['max_length'],
            doc_stride=self.model_config['doc_stride']
        )
        
        logger.info(f"Training samples: {len(self.tokenized_dataset['train'])}")
        logger.info(f"Validation samples: {len(self.tokenized_dataset['validation'])}")
    
    def compute_metrics(self, eval_pred):
        """Compute F1 and EM metrics for Q&A"""
        predictions, labels = eval_pred
        start_preds, end_preds = predictions
        start_labels, end_labels = labels
        
        # Simple exact match calculation
        start_match = (start_preds.argmax(-1) == start_labels).astype(float)
        end_match = (end_preds.argmax(-1) == end_labels).astype(float)
        exact_match = (start_match * end_match).mean()
        
        return {
            'exact_match': exact_match,
            'f1': exact_match  # Simplified - proper F1 would need token overlap
        }
    
    def train(self):
        """Train the Q&A model"""
        logger.info("Starting Q&A model training...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(self.config['paths']['output_dir'], 'qa'),
            num_train_epochs=self.train_config['num_epochs'],
            per_device_train_batch_size=self.train_config['batch_size'],
            per_device_eval_batch_size=self.train_config['batch_size'],
            gradient_accumulation_steps=self.train_config['gradient_accumulation_steps'],
            learning_rate=self.train_config['learning_rate'],
            warmup_steps=self.train_config['warmup_steps'],
            weight_decay=self.train_config['weight_decay'],
            max_grad_norm=self.train_config['max_grad_norm'],
            fp16=self.train_config['fp16'] and torch.cuda.is_available(),
            logging_dir=self.config['paths']['logs_dir'],
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="exact_match",
            greater_is_better=True,
            report_to="none",
            remove_unused_columns=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset['train'],
            eval_dataset=self.tokenized_dataset['validation'],
            tokenizer=self.tokenizer,
            data_collator=default_data_collator,
            compute_metrics=self.compute_metrics
        )
        
        # Train
        try:
            train_result = trainer.train()
            
            # Save final model
            save_path = os.path.join(
                self.config['paths']['model_save_dir'], 
                'qa_final'
            )
            trainer.save_model(save_path)
            self.tokenizer.save_pretrained(save_path)
            
            logger.info(f"Q&A model saved to {save_path}")
            logger.info(f"Training metrics: {train_result.metrics}")
            
            # Evaluate
            eval_results = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_results}")
            
            return trainer
            
        except Exception as e:
            logger.error(f"Q&A training failed: {e}")
            raise
        finally:
            clear_cuda_cache()


def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    create_directories(config)
    
    # Train summarization model
    logger.info("="*50)
    logger.info("TRAINING SUMMARIZATION MODEL")
    logger.info("="*50)
    
    summ_trainer = SummarizationTrainer(config)
    summ_trainer.setup_model_and_tokenizer()
    summ_trainer.load_and_tokenize_data()
    summ_trainer.train()
    
    # Clear memory
    del summ_trainer
    clear_cuda_cache()
    
    # Train Q&A model
    logger.info("\n" + "="*50)
    logger.info("TRAINING Q&A MODEL")
    logger.info("="*50)
    
    qa_trainer = QATrainer(config)
    qa_trainer.setup_model_and_tokenizer()
    qa_trainer.load_and_tokenize_data()
    qa_trainer.train()
    
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
