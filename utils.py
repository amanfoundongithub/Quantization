import torch 

from transformers import GPT2LMHeadModel


INT8_MAX = 127 

def get_memory_usage():
    return {
        "allocated" : torch.cuda.memory_allocated()/ 1024**2,
        "reserved" : torch.cuda.memory_allocated()/ 1024**2 
    }


def print_memory_usage(title : str = None):
    memory = get_memory_usage()
    print(f"---------------------------------------------------------------")
    print(f"{title}")
    print(f"Memory Usage Summary:")
    print(f'\tAllocated : {memory["allocated"] : .2f} MB')
    print(f'\tReserved : {memory["reserved"] : .2f} MB') 
    print(f"---------------------------------------------------------------")


def quantize_model(model : GPT2LMHeadModel, mlp = False, attn = False):
    
    for name, param in model.named_parameters():
        
        if mlp: 
            if "weight" in name and "mlp" in name: 
                # Scaling
                scale = (INT8_MAX)/param.abs().max() 
            
                scale = scale.item() 
                # Reduced weights 
                reduced_weights = (param * scale).round().to(torch.int8) 
            
                # Store the scale and reduced weights
                param.requires_grad = False 
                param.data = reduced_weights
                param.scale = scale 
            
        elif attn: 
            if "weight" in name and "c_attn" in name: 
                # Scaling
                scale = (INT8_MAX)/param.abs().max() 
            
                scale = scale.item() 
                # Reduced weights 
                reduced_weights = (param * scale).round().to(torch.int8) 
            
                # Store the scale and reduced weights
                param.requires_grad = False 
                param.data = reduced_weights
                param.scale = scale 
        else: 
            if "weight" in name: 
                # Scaling
                scale = (INT8_MAX)/param.abs().max() 
            
                scale = scale.item() 
                # Reduced weights 
                reduced_weights = (param * scale).round().to(torch.int8) 
            
                # Store the scale and reduced weights
                param.requires_grad = False 
                param.data = reduced_weights
                param.scale = scale
                
                
    return model

def dequantize_model(model : GPT2LMHeadModel, mlp = False, attn = False):
    
    for name, param in model.named_parameters():
        if mlp: 
            if "weight" in name and "mlp" in name: 
                # Scaling 
                scale = param.scale 
            
                # Dereduced weights
                dereduced_weights = (param/scale).float() 
            
                # Store the scale and reduced weight
                param.data = dereduced_weights
                param.scale = scale 
                param.requires_grad = True
        elif attn: 
            if "weight" in name and "c_attn" in name: 
                # Scaling 
                scale = param.scale 
            
                # Dereduced weights
                dereduced_weights = (param/scale).float() 
            
                # Store the scale and reduced weight
                param.data = dereduced_weights
                param.scale = scale 
                param.requires_grad = True
        else: 
            if "weight" in name: 
                # Scaling 
                scale = param.scale 
            
                # Dereduced weights
                dereduced_weights = (param/scale).float() 
            
                # Store the scale and reduced weight
                param.data = dereduced_weights
                param.scale = scale 
                param.requires_grad = True
            
            
    return model



def convert_to_hrs_min_sec(duration):
    
    hours = duration//3600
    duration = duration - 3600*hours 
    
    minutes = duration//60
    duration = duration - 60*minutes 
    
    seconds = duration 
    
    return f"{hours : .2f} hrs {minutes : .2f} min and {seconds : .2f} seconds"     