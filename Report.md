Report
---

## Results of Quantization

| **Model**           | **Memory Used (MB)** | **Latency per sentence (sec)** |    **Average Perplexity** |
|-----------------|-------------|---------------|------------|
| Original Model        | 486.70      | 0.0083        | **261.66**     |
| int8 (Full Model)      | 130.97      | **0.0081**        | 309.76     |
| int8 (Attention Only)  | 405.70      | **0.0081**        | 272.28     |
| 8-bit  (Bits & Bytes)       | 168.35      | 0.0474        | 267.02     |
| 4-bit  (Bits & Bytes)           | **127.85**      | 0.0220        | 284.91     |
| NF4 Quantization           | **127.85**      | 0.0262        | 276.55     |


---

## Analysis


### 1. Whole-Model vs. Selective Quantization
Whole-model quantization applies quantization uniformly across all layers, whereas selective quantization targets specific layers (attention layer). Whole-model quantization tends to reduce memory usage and latency significantly but can lead to higher perplexity, indicating a loss in model accuracy. In the experiment:

- **Whole 8-bit quantization** reduced memory usage from 486.7 MB to 130.97 MB but significantly increased perplexity from 261.66 to 309.76, indicating a significant decrease in accuracy while reducing memory by 75%.
- **Selective int8 quantization** (only quantizing attention layers) had a lower perplexity of 272.28 compared to full int8 quantization (309.76) while consuming more memory (405.70 MB vs. 130.97 MB). 

Selective quantization thus provides a compromise, preserving model accuracy by maintaining the precision of critical layers while reducing memory usage to an extent. This approach is effective for tasks where preserving accuracy is crucial but some memory savings are still desired. 

### 2. NF4 Quantization: Nonlinear Quantization Scales
NF4 quantization uses a nonlinear, nonuniform quantization scale. It uses a power-of-two scale to represent values more efficiently in the range critical for model weights and activations. Unlike linear quantization, which divides the value range uniformly, NF4 scales more flexibly and assigns greater precision to lower magnitude values, where neural network weights and activations often cluster.

This nonlinear approach helps to maintain model accuracy with fewer bits, where small weights can have a significant impact. In this experiment, the **NF4 quantized model** achieved a perplexity of 276.55 with a memory usage of 127.85 MB, which is close to that of the 4-bit model (perplexity 284.91) but with improved accuracy.

### 3. Impact of Linear vs. Nonlinear Quantization on Model Accuracy and Efficiency
Linear quantization (like 4-bit and 8-bit quantization) maps values in a fixed, uniform scale, which is simple to implement but may cause greater rounding errors for models with diverse parameter distributions. Nonlinear quantization, such as NF4, is adaptive and helps preserve the accuracy of critical parameters by representing smaller values with higher precision. This results in a better balance between model efficiency and accuracy.

In this case, the **NF4 model** outperformed the 4-bit linear model in terms of perplexity (276.55 vs. 284.91) while using the same amount of memory (127.85 MB). This suggests that NF4 quantization, due to its adaptive precision, retains accuracy better than strictly linear quantization at low bit rates.

### 4. Impact of Quantization on Perplexity, Speed, and Memory Usage
Quantization introduces trade-offs in perplexity, speed (latency), and memory usage. The results in this experiment show the following trends:

- **Perplexity**: As quantization reduces precision (moving from FP32 to 8-bit, 4-bit, and NF4), perplexity generally increases. The FP32 model achieved the lowest perplexity (261.66), while the Fully Quantized int8 model exhibited the highest perplexity (309.76). NF4 provided lower perplexity compared to full quantization at similar memory savings.
  
- **Speed (Latency)**: Full 8-bit and NF4 quantization models showed increased latency due to additional computational overhead from their complex scaling schemes. The FP32 model maintained a lower latency of 0.0083 seconds, while the 8-bit and NF4 models had latencies of 0.0474 and 0.0262 seconds, respectively.

- **Memory Usage**: Quantization significantly reduced memory requirements. The 8-bit and 4-bit quantized models required only 168.35 MB and 127.85 MB, respectively, compared to 486.7 MB for the FP32 model with similar model performance. 

### 5. Summary of Findings
The results highlight that quantization effectively reduces memory usage and, to some extent, latency, but at the cost of increased perplexity. 

- **Selective quantization** provides a balance between memory savings and accuracy, retaining lower perplexity than full quantization for tasks sensitive to accuracy loss.
- **NF4 nonlinear quantization** maintains better accuracy compared to linear quantization at similar bit rates, making it a suitable choice that benefits from adaptive precision.
- **Memory savings** increase with lower precision levels, though very low precision (e.g., full 4-bit quantization) results in significant accuracy loss.
- **Latency** varies depending on the model type, with some quantization methods introducing additional computational overhead, particularly in the NF4 and 8-bit quantized models.

In conclusion, quantization offers valuable trade-offs for deploying models on memory-constrained devices. NF4 and 8-bit quantization, in particular, achieve good memory efficiency with limited impact on model accuracy, presenting viable options for quantization.
