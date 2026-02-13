"""
Data preparation module for AI Paper Summarizer & Q&A System
Handles data loading, preprocessing, and dataset creation
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import logging
from tqdm import tqdm

from utils import (
    setup_logging, load_config, clean_text, 
    chunk_text, save_json, load_json
)


logger = setup_logging()


class ArxivDatasetProcessor:
    """Process arXiv dataset for summarization and Q&A tasks"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.data_config = config['data']
        self.paths = config['paths']
        
    def load_arxiv_dataset(self) -> pd.DataFrame:
        """
        Load arXiv dataset from Kaggle
        
        Returns:
            DataFrame with paper data
        """
        logger.info("Loading arXiv dataset...")
        
        try:
            # Try to load from Kaggle input directory
            arxiv_path = os.path.join(self.paths['data_dir'], 'arxiv')
            
            if os.path.exists(arxiv_path):
                # Load from local Kaggle dataset
                json_path = os.path.join(arxiv_path, 'arxiv-metadata-oai-snapshot.json')
                
                papers = []
                with open(json_path, 'r') as f:
                    for i, line in enumerate(tqdm(f, desc="Loading papers")):
                        if i >= self.data_config['max_samples']:
                            break
                        papers.append(json.loads(line))
                
                df = pd.DataFrame(papers)
                
            else:
                # Fallback: Load from HuggingFace
                logger.info("Loading from HuggingFace datasets...")
                dataset = load_dataset(
                    "scientific_papers", 
                    "arxiv",
                    split=f"train[:{self.data_config['max_samples']}]"
                )
                df = pd.DataFrame(dataset)
                
            logger.info(f"Loaded {len(df)} papers")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            # Create a small sample dataset for testing
            logger.info("Creating sample dataset for testing...")
            return self._create_sample_dataset()
    
    def _create_sample_dataset(self) -> pd.DataFrame:
        """Create a small sample dataset for testing"""
        sample_data = {
            'title': [
                'Attention Is All You Need',
                'BERT: Pre-training of Deep Bidirectional Transformers',
                'GPT-3: Language Models are Few-Shot Learners'
            ],
            'abstract': [
                'We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train.',
                'We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.',
                'Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples.'
            ],
            'article': [
                'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.' * 5,
                'Language model pre-training has been shown to be effective for improving many natural language processing tasks. These include sentence-level tasks such as natural language inference and paraphrasing, which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition and question answering, where models are required to produce fine-grained output at the token level.' * 5,
                'Language models have recently been shown to be very effective at a wide range of NLP tasks when trained on very large datasets. GPT-3 is an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting.' * 5
            ]
        }
        
        return pd.DataFrame(sample_data)
    
    def prepare_summarization_data(self, df: pd.DataFrame) -> DatasetDict:
        """
        Prepare data for summarization task
        
        Args:
            df: Input DataFrame
            
        Returns:
            DatasetDict with train and validation splits
        """
        logger.info("Preparing summarization dataset...")
        
        # Clean and prepare data
        summaries = []
        articles = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
            try:
                # Use abstract as summary, article/full text as input
                if 'abstract' in row and 'article' in row:
                    summary = clean_text(str(row['abstract']))
                    article = clean_text(str(row['article']))
                elif 'abstract' in row:
                    # If no article, create one from title + abstract
                    summary = clean_text(str(row['abstract']))
                    article = clean_text(str(row.get('title', '')) + ' ' + summary)
                else:
                    continue
                
                # Filter out too short or too long texts
                if len(summary.split()) < 20 or len(article.split()) < 50:
                    continue
                if len(article.split()) > self.data_config['max_input_length']:
                    article = ' '.join(article.split()[:self.data_config['max_input_length']])
                
                summaries.append(summary)
                articles.append(article)
                
            except Exception as e:
                logger.warning(f"Error processing row {idx}: {e}")
                continue
        
        # Create dataset
        data_dict = {
            'article': articles,
            'summary': summaries
        }
        
        dataset = Dataset.from_dict(data_dict)
        
        # Split into train and validation
        split_dataset = dataset.train_test_split(
            test_size=1 - self.data_config['train_split'],
            seed=42
        )
        
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        logger.info(f"Train samples: {len(dataset_dict['train'])}")
        logger.info(f"Validation samples: {len(dataset_dict['validation'])}")
        
        return dataset_dict
    
    def prepare_qa_data(self, df: pd.DataFrame) -> DatasetDict:
        """
        Prepare data for Q&A task
        Generate questions from papers
        
        Args:
            df: Input DataFrame
            
        Returns:
            DatasetDict with train and validation splits
        """
        logger.info("Preparing Q&A dataset...")
        
        qa_pairs = []
        
        # Generate Q&A pairs from papers
        question_templates = [
            ("What is the main contribution of this paper?", "contribution"),
            ("What method does this paper propose?", "method"),
            ("What are the key findings?", "findings"),
            ("What is the research question?", "question"),
            ("What datasets are used in this study?", "datasets"),
        ]
        
        for idx, row in tqdm(df.iterrows(), total=min(len(df), 100), desc="Generating Q&A pairs"):
            try:
                # Use abstract or article as context
                context = clean_text(str(row.get('abstract', row.get('article', ''))))
                
                if len(context.split()) < 30:
                    continue
                
                # For each paper, create Q&A pairs
                for question, qa_type in question_templates[:3]:  # Limit templates
                    # Use portions of abstract as answers
                    sentences = context.split('.')
                    if len(sentences) > 2:
                        answer = sentences[1].strip()  # Use second sentence
                        
                        qa_pairs.append({
                            'question': question,
                            'context': context,
                            'answer': answer,
                            'answer_start': context.find(answer)
                        })
                        
            except Exception as e:
                logger.warning(f"Error generating Q&A for row {idx}: {e}")
                continue
        
        # Convert to dataset format
        dataset = Dataset.from_dict({
            'question': [qa['question'] for qa in qa_pairs],
            'context': [qa['context'] for qa in qa_pairs],
            'answers': [
                {'text': [qa['answer']], 'answer_start': [qa['answer_start']]}
                for qa in qa_pairs
            ]
        })
        
        # Split dataset
        split_dataset = dataset.train_test_split(
            test_size=1 - self.data_config['train_split'],
            seed=42
        )
        
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        logger.info(f"Train Q&A pairs: {len(dataset_dict['train'])}")
        logger.info(f"Validation Q&A pairs: {len(dataset_dict['validation'])}")
        
        return dataset_dict


def tokenize_summarization_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_input_length: int = 1024,
    max_target_length: int = 256
) -> DatasetDict:
    """
    Tokenize dataset for summarization
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        max_input_length: Maximum input sequence length
        max_target_length: Maximum target sequence length
        
    Returns:
        Tokenized dataset
    """
    logger.info("Tokenizing summarization dataset...")
    
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples['article'],
            max_length=max_input_length,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples['summary'],
            max_length=max_target_length,
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing"
    )
    
    return tokenized_dataset


def tokenize_qa_dataset(
    dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    max_length: int = 384,
    doc_stride: int = 128
) -> DatasetDict:
    """
    Tokenize dataset for question answering
    
    Args:
        dataset: Input dataset
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        doc_stride: Document stride for splitting
        
    Returns:
        Tokenized dataset
    """
    logger.info("Tokenizing Q&A dataset...")
    
    def tokenize_function(examples):
        # Tokenize questions and contexts
        tokenized = tokenizer(
            examples['question'],
            examples['context'],
            max_length=max_length,
            truncation='only_second',
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding='max_length'
        )
        
        # Get answer positions
        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")
        
        start_positions = []
        end_positions = []
        
        for i, offsets in enumerate(offset_mapping):
            sample_idx = sample_mapping[i]
            answer = examples['answers'][sample_idx]
            
            start_char = answer['answer_start'][0]
            end_char = start_char + len(answer['text'][0])
            
            # Find token positions
            token_start = 0
            token_end = 0
            
            for idx, (offset_start, offset_end) in enumerate(offsets):
                if offset_start <= start_char < offset_end:
                    token_start = idx
                if offset_start < end_char <= offset_end:
                    token_end = idx
                    break
            
            start_positions.append(token_start)
            end_positions.append(token_end)
        
        tokenized['start_positions'] = start_positions
        tokenized['end_positions'] = end_positions
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names,
        desc="Tokenizing Q&A"
    )
    
    return tokenized_dataset


def main():
    """Main function to prepare datasets"""
    # Load configuration
    config = load_config()
    
    # Initialize processor
    processor = ArxivDatasetProcessor(config)
    
    # Load raw data
    df = processor.load_arxiv_dataset()
    
    # Prepare summarization dataset
    summ_dataset = processor.prepare_summarization_data(df)
    summ_dataset.save_to_disk("./data/datasets/summarization")
    logger.info("Saved summarization dataset")
    
    # Prepare Q&A dataset
    qa_dataset = processor.prepare_qa_data(df)
    qa_dataset.save_to_disk("./data/datasets/qa")
    logger.info("Saved Q&A dataset")
    
    logger.info("Data preparation complete!")


if __name__ == "__main__":
    main()
