# **Neural Network Classifier for Iris Dataset using JAX**



```python
# JAX Installation
!pip3 install --upgrade jax jaxlib
```

    Defaulting to user installation because normal site-packages is not writeable
    Looking in links: /usr/share/pip-wheels
    Requirement already satisfied: jax in ./.local/lib/python3.10/site-packages (0.6.1)
    Requirement already satisfied: jaxlib in ./.local/lib/python3.10/site-packages (0.6.1)
    Requirement already satisfied: ml_dtypes>=0.5.0 in ./.local/lib/python3.10/site-packages (from jax) (0.5.1)
    Requirement already satisfied: numpy>=1.25 in /opt/conda/envs/anaconda-2024.02-py310/lib/python3.10/site-packages (from jax) (1.26.4)
    Requirement already satisfied: opt_einsum in ./.local/lib/python3.10/site-packages (from jax) (3.4.0)
    Requirement already satisfied: scipy>=1.11.1 in /opt/conda/envs/anaconda-2024.02-py310/lib/python3.10/site-packages (from jax) (1.12.0)



```python
# Double checking installation (optional)
!pip freeze | grep jax 
```

    jax==0.6.1
    jaxlib==0.6.1



```python
#Import libraries
import jax
import jax.numpy as jnp
import time

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
```

## ***Model Setup***


```python
#Initialize Model Parameters

# X variables represents the features (e.g. flower features)
# y variable represents the target classes (e.g. flower type)
data = load_iris()
X = data.data
y = data.target.reshape(-1, 1) # prepares labels for OneHotEncoder

# OneHotEncoder turns numeric class labels (like 0, 1, 2) into binary vectors (like [1, 0, 0] and [0, 1, 0])
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalizes features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test)
```

## ***Main Execution***


```python
# Initializing Weights and Biases
# input --> hidden layer 1 --> hidden layer 2 --> output
def init_params(input_dim, hidden_dim1, hidden_dim2, output_dim, random_key):
  random_keys = jax.random.split(random_key, 3)

  W1 = jax.random.normal(random_keys[0], (input_dim, hidden_dim1)) # weight matrix connecting input and hidden layers
  b1 = jnp.zeros((hidden_dim1,))
  W2 = jax.random.normal(random_keys[1], (hidden_dim1, hidden_dim2)) 
  b2 = jnp.zeros((hidden_dim2,))
  W3 = jax.random.normal(random_keys[2], (hidden_dim2, output_dim)) 
  b3 = jnp.zeros((output_dim,))

  return W1, b1, W2, b2, W3, b3
```


```python
# Forward Step
def forward(params, X):
  W1, b1, W2, b2, W3, b3 = params
  h1 = jax.nn.relu(jnp.dot(X, W1) + b1) # dot product of two matricies, add biases, and apply activation function (i.e ReLU)
  h2 = jax.nn.relu(jnp.dot(h1, W2) + b2)
  logits = jnp.dot(h2, W3) + b3
  return logits
```


```python
# Loss Function
def loss_fn(params, x, y, l2_reg=0.0001): # regularization penalizes large weight (we want to keep weights small)
  logits = forward(params, x)
  probs = jax.nn.softmax(logits) # softmax activation turns logits into probs that sum to 1 -- interpreted as model's confidence for each class
  l2_loss = l2_reg * sum([jnp.sum(w ** 2) for w in params[::2]]) # squaring every weight and then summing them
  return -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-8), axis=1)) + l2_loss # cross entropy + regularization loss

```


```python
# Training Step
@jax.jit 
def train_step(params, x, y, lr):
  grads = jax.grad(loss_fn)(params, x, y)
  return tuple(param - lr * grad for param, grad in zip(params, grads))
```


```python
# Accuracy Evaluation
def accuracy(params, x, y):
  preds = jnp.argmax(forward(params, x), axis=1)
  targets = jnp.argmax(y, axis=1)
  return jnp.mean(preds == targets)
```


```python
# Data Loader
def data_loader(X, y, batch_size):
  for i in range(0,len(X), batch_size):
    yield X[i:i+batch_size], y[i:i+batch_size]
```

## ***Evaluation***


```python
# Parameters and Evaluation 
random_key = jax.random.key(int(time.time()))
input_dim = X_train.shape[1]
hidden_dim1 = 16 
hidden_dim2 = 8
output_dim = y_train.shape[1]
learning_rate = 0.005
batch_size = 8
epochs = 200

params = init_params(input_dim, hidden_dim1, hidden_dim2, output_dim, random_key)

for epoch in range(epochs):
  for X_batch, y_batch in data_loader(X_train, y_train, batch_size):
    params = train_step(params, X_batch, y_batch, learning_rate)

  if epoch % 10 == 0:
      train_acc = accuracy(params, X_train, y_train)
      test_acc = accuracy(params, X_test, y_test)

      print(f'Epoch {epoch}: Train Acc ({train_acc: .4f}), Test Acc ({test_acc: .4f})')

print(f'Final Test Acc: {accuracy(params, X_test, y_test): .4f}')
```

    Epoch 0: Train Acc ( 0.3083), Test Acc ( 0.4333)
    Epoch 10: Train Acc ( 0.4917), Test Acc ( 0.5667)
    Epoch 20: Train Acc ( 0.5500), Test Acc ( 0.5667)
    Epoch 30: Train Acc ( 0.6500), Test Acc ( 0.6333)
    Epoch 40: Train Acc ( 0.6750), Test Acc ( 0.6333)
    Epoch 50: Train Acc ( 0.7167), Test Acc ( 0.6000)
    Epoch 60: Train Acc ( 0.7167), Test Acc ( 0.6000)
    Epoch 70: Train Acc ( 0.8083), Test Acc ( 0.7000)
    Epoch 80: Train Acc ( 0.8250), Test Acc ( 0.7000)
    Epoch 90: Train Acc ( 0.8417), Test Acc ( 0.7000)
    Epoch 100: Train Acc ( 0.8417), Test Acc ( 0.7333)
    Epoch 110: Train Acc ( 0.8583), Test Acc ( 0.7667)
    Epoch 120: Train Acc ( 0.8750), Test Acc ( 0.7667)
    Epoch 130: Train Acc ( 0.8833), Test Acc ( 0.7667)
    Epoch 140: Train Acc ( 0.8667), Test Acc ( 0.7667)
    Epoch 150: Train Acc ( 0.8833), Test Acc ( 0.7667)
    Epoch 160: Train Acc ( 0.8833), Test Acc ( 0.8333)
    Epoch 170: Train Acc ( 0.8833), Test Acc ( 0.8667)
    Epoch 180: Train Acc ( 0.9000), Test Acc ( 0.9333)
    Epoch 190: Train Acc ( 0.9333), Test Acc ( 0.9333)
    Final Test Acc:  0.9333

