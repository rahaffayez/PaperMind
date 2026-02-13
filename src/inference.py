"""
Inference module for AI Paper Summarizer & Q&A System
Handles model loading and inference for both summarization and Q&A
"""

import os
import torch
from typing import Dict, List, Optional, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForQuestionAnswering,
    pipeline
)
from peft import PeftModel
import logging
import time

from utils import (
    setup_logging, load_config, get_device,
    clean_text, chunk_text
)


logger = setup_logging()


class PaperSummarizer:
    """Summarization model for research papers"""
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize summarizer
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.model_config = self.config['model']['summarizer']
        self.device = get_device()
        
        logger.info(f"Loading summarization model from {model_path}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def summarize(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> str:
        """
        Generate summary for input text
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary
        """
        # Clean input text
        text = clean_text(text)
        
        # Use config defaults if not specified
        max_length = max_length or self.model_config['max_length']
        min_length = min_length or self.model_config['min_length']
        num_beams = num_beams or self.model_config['num_beams']
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            max_length=self.config['data']['max_input_length'],
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=2.0,
                early_stopping=True
            )
        
        # Decode summary
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return summary
    
    def summarize_long_text(
        self,
        text: str,
        chunk_size: int = 1000,
        overlap: int = 100
    ) -> str:
        """
        Summarize long text by chunking
        
        Args:
            text: Long input text
            chunk_size: Size of text chunks
            overlap: Overlap between chunks
            
        Returns:
            Combined summary
        """
        # Split into chunks
        chunks = chunk_text(text, max_length=chunk_size, overlap=overlap)
        
        logger.info(f"Processing {len(chunks)} chunks...")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary = self.summarize(chunk)
            chunk_summaries.append(summary)
        
        # Combine summaries
        combined_text = " ".join(chunk_summaries)
        
        # Generate final summary from combined chunks
        if len(chunk_summaries) > 1:
            final_summary = self.summarize(combined_text)
        else:
            final_summary = chunk_summaries[0]
        
        return final_summary


class PaperQA:
    """Question Answering model for research papers"""
    
    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize Q&A model
        
        Args:
            model_path: Path to trained model
            config: Configuration dictionary
        """
        self.config = config or load_config()
        self.model_config = self.config['model']['qa']
        self.device = get_device()
        
        logger.info(f"Loading Q&A model from {model_path}")
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Create pipeline for easier inference
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info("Q&A model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Q&A model: {e}")
            raise
    
    def answer_question(
        self,
        question: str,
        context: str,
        max_answer_length: int = 100
    ) -> Dict[str, any]:
        """
        Answer question based on context
        
        Args:
            question: Question to answer
            context: Context containing the answer
            max_answer_length: Maximum answer length
            
        Returns:
            Dictionary with answer, score, start, and end positions
        """
        # Clean inputs
        question = clean_text(question)
        context = clean_text(context)
        
        # Get answer using pipeline
        result = self.qa_pipeline(
            question=question,
            context=context,
            max_answer_len=max_answer_length,
            handle_impossible_answer=True
        )
        
        return {
            'answer': result['answer'],
            'score': result['score'],
            'start': result['start'],
            'end': result['end']
        }
    
    def answer_multiple_questions(
        self,
        questions: List[str],
        context: str
    ) -> List[Dict[str, any]]:
        """
        Answer multiple questions about the same context
        
        Args:
            questions: List of questions
            context: Context text
            
        Returns:
            List of answer dictionaries
        """
        answers = []
        for question in questions:
            answer = self.answer_question(question, context)
            answers.append({
                'question': question,
                **answer
            })
        
        return answers


class PaperQASystem:
    """Combined system for paper summarization and Q&A"""
    
    def __init__(
        self,
        summarizer_path: str = "./models/summarizer_final",
        qa_path: str = "./models/qa_final",
        config: Optional[Dict] = None
    ):
        """
        Initialize complete system
        
        Args:
            summarizer_path: Path to summarization model
            qa_path: Path to Q&A model
            config: Configuration dictionary
        """
        self.config = config or load_config()
        
        # Initialize models
        logger.info("Initializing Paper Q&A System...")
        
        try:
            self.summarizer = PaperSummarizer(summarizer_path, config)
            self.qa = PaperQA(qa_path, config)
            logger.info("System initialized successfully")
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def process_paper(
        self,
        paper_text: str,
        questions: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Process a research paper: summarize and answer questions
        
        Args:
            paper_text: Full text of the paper
            questions: Optional list of questions to answer
            
        Returns:
            Dictionary with summary and Q&A results
        """
        start_time = time.time()
        
        # Generate summary
        logger.info("Generating summary...")
        summary = self.summarizer.summarize_long_text(paper_text)
        
        # Answer questions if provided
        qa_results = []
        if questions:
            logger.info(f"Answering {len(questions)} questions...")
            qa_results = self.qa.answer_multiple_questions(questions, paper_text)
        
        processing_time = time.time() - start_time
        
        return {
            'summary': summary,
            'qa_results': qa_results,
            'processing_time': processing_time
        }
    
    def interactive_qa(self, paper_text: str):
        """
        Interactive Q&A session for a paper
        
        Args:
            paper_text: Full text of the paper
        """
        print("\n" + "="*50)
        print("Interactive Paper Q&A System")
        print("="*50)
        print("Type 'summary' to get paper summary")
        print("Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() == 'quit':
                print("Goodbye!")
                break
            
            if question.lower() == 'summary':
                print("\nGenerating summary...")
                summary = self.summarizer.summarize_long_text(paper_text)
                print(f"\nSummary:\n{summary}\n")
                continue
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Answer question
            print("\nFinding answer...")
            result = self.qa.answer_question(question, paper_text)
            
            print(f"\nAnswer: {result['answer']}")
            print(f"Confidence: {result['score']:.2%}\n")


def test_models(paper_sample: str):
    """
    Test both models with sample text
    
    Args:
        paper_sample: Sample paper text
    """
    logger.info("Testing models...")
    
    # Initialize system
    system = PaperQASystem()
    
    # Test questions
    test_questions = [
        "What is the main contribution?",
        "What methods are proposed?",
        "What are the key findings?",
        "What datasets are used?"
    ]
    
    # Process paper
    results = system.process_paper(paper_sample, test_questions)
    
    # Display results
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(results['summary'])
    
    print("\n" + "="*50)
    print("Q&A RESULTS")
    print("="*50)
    
    for qa in results['qa_results']:
        print(f"\nQ: {qa['question']}")
        print(f"A: {qa['answer']}")
        print(f"Confidence: {qa['score']:.2%}")
    
    print(f"\nProcessing time: {results['processing_time']:.2f}s")


def main():
    """Main function for testing inference"""
    # Sample paper text for testing
    sample_paper = """
    Attention Is All You Need
    
    Abstract: We propose a new simple network architecture, the Transformer, 
    based solely on attention mechanisms, dispensing with recurrence and 
    convolutions entirely. Experiments on two machine translation tasks show 
    these models to be superior in quality while being more parallelizable 
    and requiring significantly less time to train.
    
    Introduction: The dominant sequence transduction models are based on 
    complex recurrent or convolutional neural networks that include an 
    encoder and a decoder. The best performing models also connect the 
    encoder and decoder through an attention mechanism. We propose a new 
    simple network architecture, the Transformer, based solely on attention 
    mechanisms.
    
    The Transformer follows the overall architecture using stacked self-attention 
    and point-wise, fully connected layers for both the encoder and decoder. 
    In this work we propose the Transformer, a model architecture eschewing 
    recurrence and instead relying entirely on an attention mechanism to draw 
    global dependencies between input and output.
    """
    
    # Test models
    test_models(sample_paper)


if __name__ == "__main__":
    main()
