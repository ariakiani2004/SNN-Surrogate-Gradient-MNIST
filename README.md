# 🧠 Spiking Neural Network with Surrogate Gradient on MNIST

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

A compact and educational implementation of a **Spiking Neural Network (SNN)** for handwritten digit classification on **MNIST**, trained with **Backpropagation Through Time (BPTT)** and a **sigmoid-based surrogate gradient**.

The implementation is intentionally explicit: the spiking neuron dynamics, surrogate spike function, temporal forward pass, cross-entropy loss, and **manual SGD weight updates** are all written directly in PyTorch without `torch.optim`, `snntorch`, or `torchvision`.

---

## ✨ Highlights

- Fully connected SNN architecture: **784 → 48 → 24 → 10**
- Two spiking hidden layers
- Non-spiking linear readout layer
- Hard binary spikes in the forward pass
- Sigmoid surrogate gradient in the backward pass
- Backpropagation Through Time (**BPTT**)
- Manual stochastic gradient descent (**SGD**)
- Bernoulli / rate-based spike encoding
- Cross-entropy classification loss
- Local MNIST CSV loading
- CPU-only execution
- Single-thread configuration for improved Jupyter stability
- No plotting libraries required
- No `snntorch`, `torchvision`, or `pandas` dependency

---

## 📁 Repository Structure

```text
SNN-Surrogate-Gradient-MNIST/
│
├── SNN_SG_CODE.ipynb
├── README.md
├── requirements.txt
├── main_train.csv
└── main_test.csv
```

The notebook expects the two MNIST CSV files to be located in the same directory as `SNN_SG_CODE.ipynb`.

---

## 🧠 Network Architecture

The implemented network is:

```text
MNIST image (28 × 28)
        │
        ▼
   784 inputs
        │
       W1
        │
        ▼
48 spiking neurons
        │
       W2
        │
        ▼
24 spiking neurons
        │
      Wout
        │
        ▼
10 class logits
        │
        ▼
 predicted digit
```

Compactly:

```text
784 → 48 → 24 → 10
```

The trainable synaptic weight matrices are:

```text
W1   : 784 × 48
W2   : 48 × 24
Wout : 24 × 10
```

Weights are initialized using Xavier uniform initialization.

---

## ⚡ Spiking Neuron Dynamics

The hidden layers maintain a synaptic current `I` and membrane potential `U`.

For the first hidden layer:

```text
I₁[n+1] = α I₁[n] + S_in[n] W₁
U₁[n+1] = β U₁[n] + I₁[n+1]
```

For the second hidden layer:

```text
I₂[n+1] = α I₂[n] + S₁[n] W₂
U₂[n+1] = β U₂[n] + I₂[n+1]
```

A neuron emits a spike when its membrane potential crosses the threshold:

```text
S = H(U - θ)
```

where `H` is the Heaviside step function.

After a spike, the membrane is reset by subtraction:

```text
U ← U - θS
```

In the implementation, the spike used for reset is detached from the computational graph.

---

## 🔥 Surrogate Gradient

A hard threshold produces true binary spikes but is not suitable for ordinary gradient-based learning because its derivative is zero almost everywhere.

The notebook therefore uses:

```python
def surrogate_spike(x, slope=SG_SLOPE):
    hard = (x >= 0).to(x.dtype)
    soft = torch.sigmoid(slope * x)
    return soft + (hard - soft).detach()
```

This creates two behaviors:

### Forward pass

The actual output remains a hard binary spike:

```text
S ∈ {0, 1}
```

### Backward pass

The gradient is taken from the smooth sigmoid approximation:

```text
σ(kx) = 1 / (1 + exp(-kx))
```

with derivative:

```text
dσ/dx = k σ(kx) [1 - σ(kx)]
```

The surrogate slope used by the notebook is:

```text
k = 8.0
```

This is the core idea behind **surrogate-gradient learning** in SNNs.

---

## ⏱️ Temporal Processing

Each input image is simulated for:

```text
T = 8 time steps
```

At every time step, normalized pixel intensities are converted to binary spikes using Bernoulli sampling:

```python
S_in = torch.bernoulli(x)
```

Therefore, a brighter pixel has a higher probability of producing a spike.

The temporal processing pipeline is:

```text
MNIST image
    ↓
normalize pixels to [0, 1]
    ↓
Bernoulli spike encoding
    ↓
spiking hidden layer 1
    ↓
spiking hidden layer 2
    ↓
linear readout
    ↓
accumulate logits over T steps
    ↓
average logits
    ↓
digit prediction
```

---

## 🔄 Backpropagation Through Time

The membrane potential and synaptic current preserve state across time steps.

Therefore, when:

```python
loss.backward()
```

is executed, PyTorch propagates gradients through the complete temporal computation graph.

The learning process is therefore:

```text
Forward SNN simulation
        ↓
Cross-Entropy Loss
        ↓
Backpropagation Through Time
        ↓
Surrogate Gradient at spike functions
        ↓
Weight gradients
        ↓
Manual SGD
```

---

## 🎯 Loss Function

The model uses PyTorch cross-entropy loss:

```python
loss = F.cross_entropy(logits, labels)
```

For a 10-class classification problem, cross entropy encourages the score corresponding to the correct digit to become larger than the other class scores.

The predicted class is:

```python
prediction = logits.argmax(dim=1)
```

---

## 📉 Manual SGD

This project intentionally does **not** use `torch.optim`.

After `loss.backward()`, the gradients are stored in:

```text
W1.grad
W2.grad
Wout.grad
```

The weights are then updated manually using:

```text
W_new = W_old - η ∂L/∂W
```

Implemented as:

```python
with torch.no_grad():
    W1   -= LEARNING_RATE * W1.grad
    W2   -= LEARNING_RATE * W2.grad
    Wout -= LEARNING_RATE * Wout.grad
```

The learning rate is:

```text
η = 0.05
```

Before processing the next batch, previous gradients are cleared:

```python
for W in weights:
    W.grad = None
```

---

## ⚙️ Default Experiment Configuration

| Parameter | Value |
|---|---:|
| Training samples | 60,000 |
| Test samples | 10,000 |
| Batch size | 100 |
| Epochs | 2 |
| Learning rate | 0.05 |
| Input neurons | 784 |
| Hidden layer 1 | 48 |
| Hidden layer 2 | 24 |
| Output classes | 10 |
| Time steps | 8 |
| Synaptic decay `α` | 0.80 |
| Membrane decay `β` | 0.95 |
| Threshold `θ` | 1.0 |
| Surrogate slope | 8.0 |
| Random seed | 42 |
| Device | CPU |
| CPU threads | 1 |
| Optimizer | Manual SGD |
| Loss function | Cross Entropy |

---

## 📊 Example Results

The saved notebook execution produced the following results:

| Epoch | Train Loss | Train Accuracy | Test Loss | Test Accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.8232 | 84.46% | 0.3961 | 91.87% |
| 2 | 0.3400 | 92.05% | 0.2836 | 92.93% |

Final evaluation stored in the notebook:

```text
Final test loss     : 0.2845
Final test accuracy : 92.85%
```

For the first 100 examples of one test batch:

```text
Correct predictions : 94
Wrong predictions   : 6
Sample accuracy     : 94.00%
```

> **Note:** Exact results may vary slightly between runs because Bernoulli spike generation is stochastic. The notebook fixes the random seed to improve reproducibility, but software/hardware differences can still affect the result.

---

## 📂 MNIST CSV Format

The notebook does not download MNIST automatically.

It expects:

```text
main_train.csv
main_test.csv
```

Each row must contain:

```text
label,pixel1,pixel2,...,pixel784
```

For example:

```text
7,0,0,0,12,...,255,...,0
```

The first value is the digit label:

```text
0 ... 9
```

The remaining 784 values are the grayscale pixels of a `28 × 28` image.

Pixel values must be in:

```text
0 ... 255
```

The notebook validates the row length and input range when loading the CSV files.

A header line is tolerated and skipped automatically if it does not contain 785 numerical values.

---

## 💾 Memory-Efficient Dataset Loading

MNIST images are initially stored as unsigned 8-bit integers:

```text
torch.uint8
```

This keeps RAM usage low.

The saved execution reports approximately:

```text
Training images : 44.86 MB
Test images     : 7.48 MB
```

Only the current mini-batch is converted to `float32`:

```python
x = images.to(device=device, dtype=torch.float32)
x = x.flatten(1).div(255.0)
```

---

## 🖥️ CPU and Jupyter Stability

The notebook is intentionally configured for CPU execution:

```python
device = torch.device("cpu")
```

It also limits major numerical libraries to one thread:

```python
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
```

The `DataLoader` also uses:

```python
num_workers=0
pin_memory=False
```

These settings are intended to reduce thread oversubscription and improve stability in local Jupyter environments where multi-threaded numerical libraries may cause excessive resource usage or kernel crashes.

They are execution settings and do not change the SNN learning rule.

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/ariakiani2004/SNN-Surrogate-Gradient-MNIST.git
cd SNN-Surrogate-Gradient-MNIST
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Notebook

Make sure the repository contains:

```text
SNN_SG_CODE.ipynb
main_train.csv
main_test.csv
```

Then run:

```bash
jupyter notebook
```

Open:

```text
SNN_SG_CODE.ipynb
```

and execute the cells from top to bottom.

You can also use JupyterLab if it is installed:

```bash
jupyter lab
```

---

## 🧪 Complete Learning Pipeline

```text
MNIST image
      │
      ▼
Pixel normalization
      │
      ▼
Bernoulli spike encoding
      │
      ▼
W1: 784 × 48
      │
      ▼
Spiking hidden layer 1
      │
      ▼
W2: 48 × 24
      │
      ▼
Spiking hidden layer 2
      │
      ▼
Wout: 24 × 10
      │
      ▼
Average class logits
      │
      ▼
Cross-Entropy Loss
      │
      ▼
BPTT
      │
      ▼
Sigmoid surrogate gradient
      │
      ▼
Manual SGD
      │
      ▼
Updated synaptic weights
```

---

## 🔬 What Is Learned?

The trainable parameters are only the synaptic weight matrices:

```text
W1
W2
Wout
```

The following quantities are fixed hyperparameters in the current notebook:

```text
α
β
θ
T
SG_SLOPE
```

Thus, learning modifies the synaptic connections while the neuron and temporal constants remain fixed.

---

## 📚 Concepts Demonstrated

This repository can be used to study:

- Spiking Neural Networks
- Discrete-time spiking neuron dynamics
- Membrane potential and synaptic current
- Rate-based neural encoding
- Bernoulli spike generation
- Hard threshold spike functions
- Surrogate gradients
- Sigmoid surrogate derivatives
- Backpropagation Through Time
- Cross-entropy loss
- Manual gradient descent
- Synaptic weight learning
- MNIST classification
- PyTorch autograd

---

## 📖 Background

Surrogate-gradient methods make gradient-based training possible in spiking neural networks by preserving discrete spike generation in the forward pass while replacing the non-differentiable spike derivative with a smooth approximation during backpropagation.

A useful theoretical reference is:

> Neftci, E. O., Mostafa, H., & Zenke, F. (2019).  
> **Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks.**  
> IEEE Signal Processing Magazine, 36(6), 51–63.  
> DOI: 10.1109/MSP.2019.2931595

---

## ⚠️ Scope and Limitations

This repository is primarily an educational implementation designed to make the SNN learning mechanism easy to inspect.

Important limitations include:

- Evaluation is limited to MNIST.
- The architecture is fully connected rather than convolutional.
- Input spikes use simple Bernoulli/rate coding.
- The output layer is non-spiking.
- Hyperparameters are manually fixed.
- Training is CPU-only.
- The implementation prioritizes clarity and stability over maximum speed.
- The reported accuracy should not be interpreted as a state-of-the-art MNIST benchmark.

---

## 👨‍💻 Author

**Aria Kiani**

GitHub: [@ariakiani2004](https://github.com/ariakiani2004)

---

## 🤝 Contributing

Contributions, experiments, corrections, and improvements are welcome.

Possible extensions include:

- convolutional SNN layers,
- alternative surrogate-gradient functions,
- learnable neuron parameters,
- GPU support,
- additional datasets,
- comparison with STDP,
- alternative spike encoders,
- confusion-matrix evaluation,
- energy/spike-count analysis.

---

## ⭐ Support

If you find this repository useful for studying **Spiking Neural Networks**, **Surrogate Gradient Learning**, or **Neuromorphic Computing**, consider giving it a ⭐ on GitHub.

---

## License

No license file is included automatically in this repository.

If you want the project to be open source, add a `LICENSE` file (for example, the MIT License) before publishing.
