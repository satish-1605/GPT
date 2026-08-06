
from src.datasets.dataloader import create_dataloader
from sklearn.model_selection import train_test_split
from src.datasets.dataset import GPTDataset

from src.utils.config import GPTConfig
from src.tokenizer.tokenizer import BPETokenizer
from src.datasets.preprocess import DatasetPreprocessor
from src.datasets.download import ds

def get_train_val_loaders(config):
    config = GPTConfig()
   
    tokenizer = BPETokenizer.from_pretrained(config.load_dir)

    preprocessor = DatasetPreprocessor(tokenizer)

    processed_stories = preprocessor.preprocess_corpus(
        dataset=ds['train'], 
        max_stories=config.max_stories)

    train_stories, val_stories = train_test_split(processed_stories, test_size=0.2, random_state=42)

    train_dataset = GPTDataset(
        tokenized_stories=train_stories,
        block_size= config.max_seq_len)

    train_loader = create_dataloader(dataset=train_dataset,
                                batch_size=config.batch_size,
                                shuffle=True)

    val_dataset = GPTDataset(
            tokenized_stories=val_stories,
            block_size= config.max_seq_len)
    
    val_loader = create_dataloader(dataset=val_dataset,
                                batch_size=config.batch_size,
                                shuffle=False)
    
    return train_loader, val_loader