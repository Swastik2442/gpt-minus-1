from math import sqrt
import numpy as np

def ReLU(mat: np.ndarray):
    return np.maximum(0, mat)
def softmax(mat: np.ndarray):
    exp_mat = np.exp(mat - np.max(mat, axis=-1, keepdims=True))
    return exp_mat / np.sum(exp_mat, axis=-1, keepdims=True)
def layerNorm(mat: np.ndarray, gamma: np.ndarray, beta: np.ndarray):
    mean = np.mean(mat, axis=-1, keepdims=True)
    stddev = np.std(mat, axis=-1, keepdims=True)
    xi = (mat - mean) / (stddev + 1e-6)
    return gamma * xi + beta

n_vocab = 37000
n_ctx = 512
d_model = 768
d_k = 64
h = d_model // d_k

# Input
X = np.random.randint(0, n_vocab-1, (n_ctx,))
y = np.random.randint(0, n_vocab-1, (n_ctx,))

# Text + Pos. Embedding
Wt = np.random.rand(n_vocab, d_model)
Wp = np.random.rand(n_ctx, d_model)

tp = Wt[X] + Wp
print(tp.shape)

# Multi-Head Attention
Wq = np.random.rand(d_model, d_model)
Wk = np.random.rand(d_model, d_model)
Wv = np.random.rand(d_model, d_model)

q = tp @ Wq
k = tp @ Wk
v = tp @ Wv

sa = softmax((q @ k.T) / sqrt(d_k)) @ v

# Add & LayerNorm
Wg1 = np.ones((d_model,))
Wb1 = np.zeros((d_model,))
aln1 = layerNorm(tp + sa, Wg1, Wb1)

# FFNN
Wf1 = np.random.rand(d_model, d_model * 4)
Wf2 = np.random.rand(d_model * 4, d_model)

# Feed Forward Network
ffn = (ReLU(aln1 @ Wf1)) @ Wf2

# Add & LayerNorm
Wg2 = np.ones((d_model,))
Wb2 = np.zeros((d_model,))
aln2 = layerNorm(aln1 + ffn, Wg2, Wb2)

# No Linear or Softmax in GPT

# Loss (Language Modelling)
# Cross-Entropy Loss, cannot comprehend it rn

print(f"{X.shape=}\n{y.shape=}\n{tp.shape=}\n{q.shape=}\n{k.shape=}\n{v.shape=}\n{sa.shape=}\n{Wg1.shape=}\n{Wb1.shape=}\n{aln1.shape=}\n{Wf1.shape=}\n{Wf2.shape=}\n{ffn.shape=}\n{Wg2.shape=}\n{Wb2.shape=}\n{aln2.shape=}")
