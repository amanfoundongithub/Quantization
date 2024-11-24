import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from tqdm import tqdm 
import time 
import math 

from transformers import BartForConditionalGeneration, BartTokenizer
from load_data import load_penn_bank_dataset



class W8A16LinearLayer(nn.Module):
  def __init__(self, input_features, output_features, bias=True, dtype=torch.float32):
    super().__init__()

    self.register_buffer("int8_weights", torch.randint(-128,127, (output_features, input_features), dtype=torch.int8))
    self.register_buffer("scales", torch.randn((output_features), dtype= dtype))

    if bias:
      self.register_buffer("bias", torch.randn((1, output_features), dtype = dtype))
    else:
      self.bias = None

  def forward(self, inputs):
    converted_weights = self.int8_weights.to(inputs.dtype)
    output = F.linear(inputs, converted_weights) * self.scales

    if self.bias is not None:
      output = output + self.bias

    return output

  def quantize(self, weights):
    w_fp32 = weights.clone().to(torch.float32)

    scales = w_fp32.abs().max(dim=-1).values/127
    scales = scales.to(weights.dtype)

    int8_weights = torch.round(weights/scales.unsqueeze(1)).to(torch.int8)

    self.int8_weights  = int8_weights
    self.scales = scales
    

def replace_linear_layer_with_W8A16Linear_layer_and_quantization(module, target , exclude_list):
  for name, child in module.named_children():
    if isinstance(child, nn.Linear) and not any([x == name for x in exclude_list]):
      old_bias = child.bias
      old_weights = child.weight

      new_module = target(child.in_features, child.out_features, 
                                   old_bias is not None, child.weight.dtype)

      setattr(module, name, new_module)
      getattr(module, name).quantize(old_weights)

      if old_bias is not None:
        getattr(module, name).bias = old_bias

    # else:
    #   replace_linear_layer_with_W8A16Linear_layer(child, target, exclude_list)
    else: 
      replace_linear_layer_with_W8A16Linear_layer_and_quantization(child, target, exclude_list) 


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
        tokens = tokenizer(text, truncation = False, return_tensors = "pt")["input_ids"].to("cuda") 
        
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
    print(f"\tAverage Perplexity : {math.exp(total_loss/total_count) : .4f}")
    print(f"---------------------------------------------------------------")

    
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to("cuda") 
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

print(model.get_memory_footprint()) 


penn_bank_dataset = load_penn_bank_dataset()

calculate_perplexity(penn_bank_dataset, model)   

replace_linear_layer_with_W8A16Linear_layer_and_quantization(model, 
                                      W8A16LinearLayer, [])


print(model.get_memory_footprint()) 


calculate_perplexity(penn_bank_dataset, model)   