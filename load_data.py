from datasets import load_dataset, concatenate_datasets



def load_penn_bank_dataset():
    
    ds = load_dataset("ptb_text_only")["test"]
    
    ds = ds.shuffle(seed = 777)
    
    return ds 