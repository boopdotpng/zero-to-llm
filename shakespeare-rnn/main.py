from tinygrad import Tensor 
import tinygrad.nn as nn
from tinygrad.helpers import GlobalCounters, getenv, trange
from typing import List, Tuple
from pathlib import Path
from utils import make_blocks

class CharTokenizer:
  def __init__(self, path: Path):
    self.unique = sorted(set(path.open().read()))
    self.vocab = {ch:i for i, ch in enumerate(self.unique)}
    self.r_vocab = {i: ch for i, ch in enumerate(self.unique)}
  def encode(self, s: str): return [self.vocab[x] for x in s]
  def decode(self, ids): return "".join(self.r_vocab[i] for i in ids)
  def vocab_size(self): return len(self.unique)

dataset = Path(__file__).parent.parent / "datasets" / "shakespeare.txt"
tokenizer = CharTokenizer(dataset)

class LSTMCell:
  def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0):
    self.hidden_size = hidden_size
    self.dropout = dropout
    # using nn.Linear guarantees parameters are discoverable by nn.state.get_parameters
    self.x2g = nn.Linear(input_size, 4*hidden_size, bias=True)
    self.h2g = nn.Linear(hidden_size, 4*hidden_size, bias=True)

  def __call__(self, x: Tensor, h: Tensor, c: Tensor):
    # x: (B, I), h,c: (B, H)
    gates = self.x2g(x) + self.h2g(h)           # (B, 4H)
    i, f, g, o = gates.chunk(4, 1)              # split along channel dim
    i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
    c_new = f * c + i * g
    h_new = o * c_new.tanh()
    if self.dropout and self.dropout > 0:
      h_new = h_new.dropout(self.dropout)
    return h_new, c_new

class LSTM:
  def __init__(self, vocab_size: int, embed_size: int, hidden_size: int, layers: int = 1, dropout: float = 0.0):
    # These are all attached to self, so the optimizer can find them
    self.embed = nn.Embedding(vocab_size, embed_size)
    self.cells = [LSTMCell(embed_size if l==0 else hidden_size, hidden_size, dropout=dropout) for l in range(layers)]
    self.proj = nn.Linear(hidden_size, vocab_size, bias=True)
    self.hidden_size = hidden_size
    self.layers = layers
    self.dropout = dropout

  def __call__(self, X: Tensor, hc=None):
    # X: (B, T) int tokens
    B, T = X.shape
    # Embedding: (B, T, E)
    E = self.embed(X)

    # init states if not provided; you usually don't want grads on initial h0/c0
    if hc is None:
      hs = [Tensor.zeros(B, self.hidden_size) for _ in range(self.layers)]
      cs = [Tensor.zeros(B, self.hidden_size) for _ in range(self.layers)]
    else:
      hs, cs = hc

    logits_t = []
    for t in range(T):
      inp = E[:, t, :]                         # (B, E) or (B, H) as we go deeper
      for l, cell in enumerate(self.cells):
        hs[l], cs[l] = cell(inp, hs[l], cs[l]) # (B, H), (B, H)
        inp = hs[l]
      logits_t.append(self.proj(inp))          # append (B, V)

    # (B, T, V)
    logits = Tensor.stack(logits_t, dim=1)
    return logits, (hs, cs)


if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = make_blocks(token_ids=tokenizer.encode(dataset.open().read()))

  # X_train, Y_train: (N, T) int tokens
  model = LSTM(vocab_size=tokenizer.vocab_size(), embed_size=256, hidden_size=128, layers=2, dropout=0.1)
  opt = nn.optim.AdamW(nn.state.get_parameters(model), lr=3e-4)

  @Tensor.train()
  def train_step():
    opt.zero_grad()
    # batch of sequences
    idx = Tensor.randint(128, high=X_train.shape[0])
    Xb, Yb = X_train[idx], Y_train[idx]              # (B, T)

    logits, _ = model(Xb)                            # (B, T, V)
    B, T, V = logits.shape
    loss = logits.reshape(B*T, V).sparse_categorical_crossentropy(Yb.reshape(B*T)).mean()
    loss.backward()
    opt.step()
    return loss

  def val_ppl():
    logits, _ = model(X_test)
    B, T, V = logits.shape
    val_loss = logits.reshape(B*T, V).sparse_categorical_crossentropy(Y_test.reshape(B*T)).mean()
    return val_loss.exp()

  test_ppl = float('nan')
  for i in (t := trange(getenv("STEPS", 1000))):
    GlobalCounters.reset()
    loss = train_step()
    t.set_description(f"loss: {loss.item():6.2f} val_ppl: {test_ppl:.2f}")
    if i % 50 == 49:
      test_ppl = val_ppl().item()
