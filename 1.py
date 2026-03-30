import numpy as np
import matplotlib.pyplot as plt

tokens = ["I", "love", "deep", "learning"]
np.random.seed(0)
d_model = 4
X = np.random.rand(len(tokens), d_model)

W_q = np.random.rand(d_model, d_model)
W_k = np.random.rand(d_model, d_model)
W_v = np.random.rand(d_model, d_model)

Q = X @ W_q
K = X @ W_k
V = X @ W_v

dk = d_model

scores = Q @ K.T / np.sqrt(dk)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
  
attention_weights = softmax(scores)
context_vector = attention_weights @ V
print("Attention Weights:\n", attention_weights)
print("\nContext Vector:\n", context_vector)

plt.figure(figsize=(6,5))
plt.imshow(attention_weights, cmap='viridis')
plt.colorbar()
plt.xticks(range(len(tokens)), tokens)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.title("Self-Attention Weights Visualization")

for i in range(len(tokens)):
    for j in range(len(tokens)):
        plt.text(j, i,
                 f"{attention_weights[i,j]:.2f}",
                 ha="center",
                 va="center",
                 color="white",
                 fontsize=11,
                 fontweight="bold")

plt.tight_layout()
plt.show()
