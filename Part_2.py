from load_model import get_gpt2
from load_data import load_penn_bank_dataset


from utils import print_memory_usage, convert_to_hrs_min_sec
from tqdm  import tqdm 


import torch 
import torch.nn as nn 

import math 
import time 

def calculate_perplexity(dataset, model):
    # Model to evaluation mode
    model.eval()

    total_count = 0.0
    total_loss = 0.0 
    
    # Iterate over the dataset
    progress_bar = tqdm(dataset, desc = f"Progress", leave = False)
    
    start = time.time() 
    for sent in progress_bar: 
        # Text
        text = sent["sentence"]
        
        # Now tokenize  
        tokens = tokenizer(text, truncation = False, return_tensors = "pt")["input_ids"].to(device) 
        
        if tokens.size(1) <= 1:
            continue 
        
        # Predict the next tokens
        with torch.no_grad():
            # Prediction
            predicted_logits = model(
                input_ids = tokens,
                labels = tokens
            )
            
            loss = predicted_logits.loss 
            
            total_loss += loss.item() 
            total_count += 1
            
            progress_bar.set_postfix(loss = total_loss/total_count) 
    
    end = time.time()
    
    print(f"---------------------------------------------------------------")
    print(f"Testing Summary:")
    print(f"\tTotal Time Taken: {convert_to_hrs_min_sec(end - start)} for whole iteration") 
    print(f"\tAverage Perplexity : {math.exp(total_loss/total_count) : .4f}")
    print(f"---------------------------------------------------------------")


device = "cuda" if torch.cuda.is_available() else "cpu"

penn_bank_dataset = load_penn_bank_dataset()





print_memory_usage("Before Loading 8 bit Bits&Bytes...") 
##################################### LOAD THE MODEL ####################################################
dic = get_gpt2(device, bits_bytes = True, bits = 8) 
model = dic["model"]
tokenizer = dic["tokenizer"]

torch.save(model.state_dict(), "gpt2_bitsbytes_8.pt") 

print(f"Memory Used By 8 bit Bits&Bytes: {dic['size_mb'] : .2f} MB")

print_memory_usage("After Loading  8 bit Bits&Bytes...")

calculate_perplexity(penn_bank_dataset, model) 
#########################################################################################################







print_memory_usage("Before Loading 4 bit Bits&Bytes...") 
##################################### LOAD THE MODEL ####################################################
dic = get_gpt2(device, bits_bytes = True, bits = 4) 
model = dic["model"]
tokenizer = dic["tokenizer"]

torch.save(model.state_dict(), "gpt2_bitsbytes_4.pt") 

print(f"Memory Used By 4 bit Bits&Bytes: {dic['size_mb'] : .2f} MB")

print_memory_usage("After Loading  4 bit Bits&Bytes...")

calculate_perplexity(penn_bank_dataset, model) 
#########################################################################################################


print_memory_usage("Before Loading NF4...") 
##################################### LOAD THE MODEL ####################################################
dic = get_gpt2(device, nf4 = True)  
model = dic["model"]
tokenizer = dic["tokenizer"]

torch.save(model.state_dict(), "gpt2_nf4.pt") 

print(f"Memory Used By NF4: {dic['size_mb'] : .2f} MB")

print_memory_usage("After Loading NF4...")

calculate_perplexity(penn_bank_dataset, model) 
#########################################################################################################