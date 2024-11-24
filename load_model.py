from transformers import GPT2LMHeadModel, GPT2Tokenizer, BitsAndBytesConfig

import torch 


def get_gpt2(device, bits_bytes = False, bits = 8, nf4 = False):
    
    if bits_bytes == False and nf4 == False: 
        gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device) 
    
    elif bits_bytes == True:
        if bits == 8:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            gpt2 = GPT2LMHeadModel.from_pretrained(
                "gpt2", 
                quantization_config=quantization_config)
        elif bits == 4:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

            gpt2 = GPT2LMHeadModel.from_pretrained(
                "gpt2", 
                quantization_config=quantization_config)
        else: 
            raise ValueError("Invalid Bits Size")


    elif nf4 == True:
        nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)
        
        gpt2 = GPT2LMHeadModel.from_pretrained(
                "gpt2", 
                quantization_config = nf4_config)
        
    
    
    size_in_mb = gpt2.get_memory_footprint()/ 1024**2

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_tokenizer.padding_side = 'left'
    
    return {
        "model" : gpt2, 
        "tokenizer" : gpt2_tokenizer,
        "size_mb" : size_in_mb
    }