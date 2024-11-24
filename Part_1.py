from load_model import get_gpt2
from load_data import load_penn_bank_dataset


from utils import print_memory_usage, quantize_model, dequantize_model, convert_to_hrs_min_sec
from tqdm  import tqdm 


import torch 
import torch.nn as nn 
import math 
import time 



device = "cuda" if torch.cuda.is_available() else "cpu"

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

#########################################################################################################
penn_bank_dataset = load_penn_bank_dataset()
#########################################################################################################


print_memory_usage("Before Loading Model...") 
##################################### LOAD THE MODEL ####################################################
dic = get_gpt2(device) 

model = dic["model"]
tokenizer = dic["tokenizer"]

print(f"Memory Used By Model : {dic['size_mb'] : .2f} MB")

torch.save(model.state_dict(), "gpt2_original_model.pt")

print_memory_usage("After Loading Model...")

calculate_perplexity(penn_bank_dataset, model) 
#########################################################################################################



############################### QUANTIZE THE MODEL ######################################################
dic = get_gpt2(device) 
model = dic["model"]
tokenizer = dic["tokenizer"]
model = quantize_model(model) 

print(f"Memory Used By Model : {model.get_memory_footprint() / 1024**2 : .2f} MB")

torch.save(model.state_dict(), "gpt2_full_int8_quantize.pt") 

print_memory_usage("After Full Quantization...")

calculate_perplexity(penn_bank_dataset, dequantize_model(model)) 
#########################################################################################################

############################### QUANTIZE THE MLP MODEL ##################################################
dic = get_gpt2(device) 
model = dic["model"]
tokenizer = dic["tokenizer"]
model = quantize_model(model, mlp = True) 

print(f"Memory Used By Model : {model.get_memory_footprint() / 1024**2 : .2f} MB")

torch.save(model.state_dict(), "gpt2_selective_mlp_int8_quantize.pt") 

print_memory_usage("After MLP Quantization...")

calculate_perplexity(penn_bank_dataset, dequantize_model(model, mlp = True)) 
#########################################################################################################




############################### QUANTIZE THE MLP MODEL ##################################################
dic = get_gpt2(device) 
model = dic["model"]
tokenizer = dic["tokenizer"]
model = quantize_model(model, attn = True)  

print(f"Memory Used By Model : {model.get_memory_footprint() / 1024**2 : .2f} MB")

torch.save(model.state_dict(), "gpt2_selective_attn_int8_quantize.pt") 

print_memory_usage("After Attention Quantization...")

calculate_perplexity(penn_bank_dataset, dequantize_model(model, attn = True)) 
#########################################################################################################
