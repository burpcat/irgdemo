import logging
from config import Config
from utils import setup_logging, ensure_directories, calculate_metrics, prepare_model_input, save_results
from dataset import MIMICSDataset  # Import the new dataset class
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # Fixed import
from tqdm import tqdm
from typing import List, Dict, Any

class ClarifyingQuestionSystem:
    def __init__(self, config: Config):
        self.config = config
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model and tokenizer"""
        logging.info(f"Loading model: {self.config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_NAME)
        
        self.model = self.model.to(self.config.DEVICE)
        logging.info(f"Using device: {self.config.DEVICE}")
    
    # def generate_question(self, query: str, snippets: List[str]) -> str:
    #     """Generate a clarifying question"""
    #     inputs = prepare_model_input(self.tokenizer, query, snippets, self.config)
        
    #     with torch.no_grad():
    #         outputs = self.model.generate(
    #             input_ids=inputs['input_ids'],
    #             attention_mask=inputs['attention_mask'],
    #             max_length=self.config.MAX_QUESTION_LENGTH,
    #             num_beams=4,
    #             no_repeat_ngram_size=3
    #         )
        
    #     question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #     return question

    def generate_question(self, query: str, snippets: List[str]) -> str:
        """Generate a clarifying question"""
        # Prepare input text that includes query and available options
        input_text = f"Query: {query}\nOptions: {' | '.join(snippets)}\n"
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.config.DEVICE)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config.MAX_QUESTION_LENGTH,
                num_beams=4,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up the generated question
        if "Query:" in question:
            question = question.split("Query:")[0]
        if "Options:" in question:
            question = question.split("Options:")[0]
            
        return question.strip()

def run_demo():
    # Setup
    setup_logging()
    ensure_directories()
    logging.info("Starting Clarifying Questions Demo")
    
    # Initialize configurations and system
    config = Config()
    
    # Use the new dataset loader
    dataset = MIMICSDataset(
        data_dir=config.DATA_DIR,
        split="train",  # or "test" for evaluation
        retriever_type="bing",  # or "bm25" or "contriever"
        max_documents=config.NUM_DOCUMENTS,
        tokenizer_name=config.MODEL_NAME
    )
    
    system = ClarifyingQuestionSystem(config)
    
    logging.info(f"Loaded dataset with {len(dataset)} examples")
    
    # Store results
    results = []
    
    # Run examples
    logging.info("Generating clarifying questions:")
    for example in tqdm(dataset):
        query = example['query']
        snippets = example['evidence_documents']
        ground_truth = example['question']
        
        # Generate question
        generated_question = system.generate_question(query, snippets)
        
        # Calculate metrics
        metrics = calculate_metrics(generated_question, ground_truth)
        
        # Store results
        results.append({
            'query': query,
            'generated_question': generated_question,
            'ground_truth': ground_truth,
            'metrics': metrics
        })
        
        # Display results
        print("\n" + "="*50)
        print(f"Query: {query}")
        print(f"Generated Question: {generated_question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Metrics: {metrics}")
        print("="*50)
    
    # Save results
    save_results(results, 'generation_results.json')
    logging.info("Results saved to output/generation_results.json")

if __name__ == "__main__":
    run_demo()