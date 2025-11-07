import urllib.request
import re
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# =============================
# Data Loading and Preprocessing
# =============================

def download_text_data(url, file_path):
    """Download text data from URL and save locally."""
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
        out_file.write(response.read())

def load_text_data(file_path):
    """Load text data from local file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# =============================
# Tokenization
# =============================

def preprocess_text(raw_text):
    """
    Preprocess text using regex tokenization.
    
    Args:
        raw_text (str): Raw text to preprocess
        
    Returns:
        tuple: (preprocessed_tokens, token_ids)
    """
    # Split text into tokens using regex
    preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed_text = [token.strip() for token in preprocessed_text if token.strip()]
    
    # Use tiktoken for BPE tokenization
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(raw_text)
    
    return preprocessed_text, token_ids

# =============================
# Custom Dataset for Language Model Training
# =============================

class GPTDatasetV1(Dataset):
    """
    Custom dataset for language model training using sliding window approach.
    """
    
    def __init__(self, text, tokenizer, max_length, stride):
        """
        Initialize dataset with text data and tokenization parameters.
        
        Args:
            text (str): Raw text to be processed
            tokenizer: Tokenizer object (e.g., tiktoken encoder)
            max_length (int): Sequence length for input/target chunks
            stride (int): Sliding window step size
        """
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        # Create sliding window chunks
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i+max_length]
            target_chunk = token_ids[i+1:i+max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# =============================
# DataLoader Creation
# =============================

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, 
                        shuffle=True, drop_last=True, num_workers=0):
    """
    Create DataLoader for language model training.
    
    Args:
        txt (str): Raw text data
        batch_size (int): Number of samples per batch
        max_length (int): Maximum sequence length
        stride (int): Sliding window step size
        shuffle (bool): Whether to shuffle data
        drop_last (bool): Whether to drop incomplete batches
        num_workers (int): Number of worker processes
        
    Returns:
        DataLoader: Configured PyTorch DataLoader
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=num_workers
    )
    return dataloader



# =============================
# Tokenizer Utility
# =============================

def get_tokenizer(name="gpt2"):
    """
    Get tiktoken tokenizer by name.
    
    Args:
        name (str): Tokenizer name ("gpt2", etc.)
        
    Returns:
        tiktoken.Encoding: Tokenizer object
    """
    return tiktoken.get_encoding(name)

def encode_with_special_tokens(text, tokenizer, allowed_special=None):
    """
    Encode text with special token handling.
    
    Args:
        text (str): Text to encode
        tokenizer: tiktoken tokenizer object
        allowed_special (set): Set of allowed special tokens
        
    Returns:
        list: Token IDs
    """
    if allowed_special:
        return tokenizer.encode(text, allowed_special=allowed_special)
    else:
        return tokenizer.encode(text)

def decode_tokens(token_ids, tokenizer):
    """
    Decode token IDs back to text.
    
    Args:
        token_ids (list): List of token IDs
        tokenizer: tiktoken tokenizer object
        
    Returns:
        str: Decoded text
    """
    return tokenizer.decode(token_ids) 