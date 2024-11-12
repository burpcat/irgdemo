import os
import json
import logging
from pathlib import Path
import torch
from typing import List, Dict, Any
from config import Config  # Add this import

def setup_logging():
    """Configure logging settings"""
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('clarifying_questions.log')
        ]
    )

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['./data', './models', './output']
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def save_results(results: Dict[str, Any], filename: str):
    """Save results to output directory"""
    output_path = Path("./output") / filename
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def load_sample_data() -> List[Dict[str, Any]]:
    """Load sample data for testing"""
    return [
        {
            "query": "windows",
            "question": "Are you looking for Microsoft Windows or home windows?",
            "facets": ["microsoft windows", "home windows"],
            "snippets": [
                "Windows is a popular operating system by Microsoft",
                "Home windows come in various styles including double-hung and casement"
            ]
        },
        {
            "query": "python",
            "question": "Are you interested in the programming language or the snake?",
            "facets": ["python programming", "python snake"],
            "snippets": [
                "Python is a popular programming language",
                "Pythons are large non-venomous snakes"
            ]
        },
        {
            "query": "jaguar",
            "question": "Are you looking for information about the animal or the car brand?",
            "facets": ["jaguar animal", "jaguar cars"],
            "snippets": [
                "Jaguar is a luxury vehicle brand owned by Tata Motors",
                "The jaguar is a large cat species native to the Americas"
            ]
        }
    ]

def calculate_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    return {
        "length": len(generated.split()),
        "matches_reference": generated.lower() == reference.lower()
    }

def prepare_model_input(tokenizer, query: str, snippets: List[str], config: Config) -> Dict[str, torch.Tensor]:
    """Prepare input for the model"""
    # Format snippets into a single context string
    context = " [SEP] ".join([s.strip() for s in snippets if s.strip()])
    
    # Create input text with special tokens
    input_text = f"Query: {query.strip()} Context: {context}"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=config.MAX_LENGTH,
        **config.TOKENIZER_KWARGS
    )
    
    # Move to correct device
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
    return inputs