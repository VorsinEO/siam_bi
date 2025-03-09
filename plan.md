# plan
todo:
* code for transform preds with function inverse
* code for quality measure
* prepare thee scaler and save_load robust/quantile for data
* prepare for target three scalers and save_load 

## exps 1
Log1p for timeseries
RobustScaler for targets

## exps 2
RobustScaler for timeseries
RobustScaler for targets

## exp3
delety 2-5% outliers from targets


## exp4 
data smoothing https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html



For 45,000 samples with sequences of length ~1000, the transformer encoder needs to be carefully designed to balance capacity and generalization. Here’s a good starting configuration to avoid overfitting while ensuring stable training:  

---

### **1. Transformer Encoder Architecture**
| Parameter                | Recommended Value |
|-------------------------|-----------------|
| **Embedding Dim (d_model)** | 64 or 128 |
| **Number of Layers (N)** | 2–4 |
| **Number of Attention Heads** | 4–8 |
| **Feedforward Dimension (d_ff)** | 256–512 |
| **Dropout Rate** | 0.1–0.2 |
| **Max Sequence Length** | 1024 (slightly more than 1000 to allow some flexibility) |
| **Activation Function** | GELU |
| **Normalization** | LayerNorm |

#### **Justification:**
- **d_model = 64 or 128** → Keeps the parameter count moderate while still allowing the model to capture patterns in the sequences.
- **2–4 layers** → Helps avoid excessive depth, which could lead to overfitting.
- **4–8 attention heads** → Allows the model to capture multiple aspects of the time series behavior without excessive memory overhead.
- **d_ff = 256–512** → Ensures enough capacity in the feedforward network without making it too large.
- **Dropout = 0.1–0.2** → Helps prevent overfitting, especially given that transformers tend to overfit on small datasets.
- **LayerNorm** → Stabilizes training and prevents exploding/vanishing gradients.

---

### **2. Regularization Techniques**
- **Weight Decay (~1e-4 to 1e-5)** → Encourages the model to prefer smaller weights.
- **Gradient Clipping (e.g., max_norm = 1.0)** → Prevents exploding gradients.
- **Early Stopping** → Monitor validation loss and stop training when overfitting starts.

---

### **3. Training Strategy**
- **Batch Size:** 64–128 (Start with 64, increase if memory allows)
- **Optimizer:** AdamW (better generalization than plain Adam)
- **Learning Rate:** 1e-4 with a cosine decay or linear warmup for the first few epochs
- **Epochs:** 20–50 (Monitor validation loss and stop early)

---

### **4. Augmentations to Improve Generalization**
1. **Random Noise Injection** → Add small Gaussian noise to input features during training.
2. **Time Warping** → Slightly distort the time axis to encourage robustness.
3. **CutMix/Mixup for Sequences** → Mix portions of different sequences to improve generalization.
4. **Random Masking** → Randomly mask some inputs (like in BERT) to encourage robustness.

---

### **5. Alternative Architectures to Test**
- **Temporal Convolutional Network (TCN)** → If transformers struggle, TCNs can be a strong alternative.
- **Hybrid Transformer-LSTM** → First process time series with a CNN or LSTM, then apply a transformer for long-range dependencies.

---

### **Final Thoughts**
This setup provides a **good balance of expressiveness and regularization** to start without overfitting. After initial tests, you can fine-tune hyperparameters based on validation loss trends. 🚀




Results

CV1_vit 64 with weights

TS_10 10_10 0.6563 with weights

TS3 12 0.631 with weights ts_3f_12_631

CV2_conv Результат: 0.6343 with weights 

TS 3f 13 Результат: 0.6537  with weights