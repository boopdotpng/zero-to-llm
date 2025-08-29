from tinygrad import Tensor

def make_blocks(token_ids, block_size: int = 256, stride: int = 64):
  X, y = [], []
  for i in range(0, len(token_ids) - block_size - 1, stride):
    x = token_ids[i : i + block_size]
    Y = token_ids[i + 1 : i + block_size + 1]
    X.append(x)
    y.append(Y)
  num_samples = len(X)
  train_size = int(num_samples * 0.9)
  X_train, Y_train = X[:train_size], y[:train_size]
  X_test,  Y_test  = X[train_size:], y[train_size:]
  return Tensor(X_train), Tensor(Y_train), Tensor(X_test), Tensor(Y_test)
