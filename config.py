import torch

class Config:
    # Paths
    DATA_DIR = "./data"
    MODEL_DIR = "./models"
    OUTPUT_DIR = "./output"
    
    # Model parameters
    MODEL_NAME = "facebook/bart-base"
    MAX_LENGTH = 512
    MAX_QUESTION_LENGTH = 64
    
    # Training parameters
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    
    # Retrieval parameters
    NUM_DOCUMENTS = 10
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Tokenizer settings
    TOKENIZER_KWARGS = {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt"
    }