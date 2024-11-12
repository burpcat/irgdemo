import logging
from config import Config
from utils import (
    setup_logging, 
    ensure_directories, 
    load_sample_data, 
    calculate_metrics,
    prepare_model_input,
    save_results
)
import torch
# Update this line
# from transformers import AutoTokenizer, AutoModelForSeq2SeqGenerationWithLMHead
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Any

class SimpleMIMICSDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.data = load_sample_data()
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

class ClarifyingQuestionSystem:
    def __init__(self, config: Config):
        self.config = config
        self.setup_model()
        
    def setup_model(self):
        """Initialize the model and tokenizer"""
        logging.info(f"Loading model: {self.config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
        # Update this line
        # self.model = AutoModelForSeq2SeqGenerationWithLMHead.from_pretrained(self.config.MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.MODEL_NAME)
        
        self.model = self.model.to(self.config.DEVICE)
        logging.info(f"Using device: {self.config.DEVICE}")
    
    def generate_question(self, query: str, snippets: List[str]) -> str:
        """Generate a clarifying question"""
        inputs = prepare_model_input(self.tokenizer, query, snippets, self.config)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=self.config.MAX_QUESTION_LENGTH,
                num_beams=4,
                no_repeat_ngram_size=3
            )
        
        question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return question

def run_demo():
    # Setup
    setup_logging()
    ensure_directories()
    logging.info("Starting Clarifying Questions Demo")
    
    # Initialize configurations and system
    config = Config()
    dataset = SimpleMIMICSDataset(config.DATA_DIR)
    system = ClarifyingQuestionSystem(config)
    
    logging.info(f"Loaded dataset with {len(dataset)} examples")
    
    # Store results
    results = []
    
    # Run examples
    logging.info("Generating clarifying questions:")
    for example in tqdm(dataset):
        query = example['query']
        snippets = example['snippets']
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