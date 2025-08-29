from tinygrad import Tensor, TinyJit
import tinygrad.nn as nn
from tinygrad.helpers import GlobalCounters, getenv, trange 
from typing import List
from pathlib import Path
from utils import make_blocks
import math

class CharTokenizer:
  def __init__(self, path: Path):
    self.unique = sorted(set(path.open().read()))
    self.vocab = {self.unique[i]: i for i in range(len(self.unique))}
    self.r_vocab = {i: self.unique[i] for i in range(len(self.unique))}
  def encode(self, s: str) -> List[int]: return list(map(lambda x: self.vocab[x], list(s)))
  def decode(self, ids: List[int]) -> str:
    return "".join(self.r_vocab[i] for i in ids)
  def vocab_size(self) -> int: return len(self.unique)

dataset = Path(__file__).parent.parent / "datasets" / "shakespeare.txt"
tokenizer = CharTokenizer(dataset)

class LSTMCell:
  def __init__(self, input_size: int, hidden_size: int, bias:bool=False):
    stdv = 1.0 / math.sqrt(hidden_size)
    self.weight_ih = Tensor.uniform(hidden_size*4, input_size, low=-stdv, high=stdv)
    self.weight_hh = Tensor.uniform(hidden_size*4, hidden_size, low=-stdv, high=stdv)
    self.bias_ih: Tensor|None = Tensor.zeros(hidden_size*4) if bias else None
    self.bias_hh: Tensor|None = Tensor.zeros(hidden_size*4) if bias else None
    self.H = hidden_size
  def __call__(self, x: Tensor, hc: tuple[Tensor, Tensor]|None = None) -> tuple[Tensor, Tensor]:
    h_prev, c_prev = hc if hc is not None else (Tensor.zeros(x.size(0), self.H), Tensor.zeros(x.size(0), self.H))
    gates = x.linear(self.weight_ih.T, self.bias_ih) + h_prev.linear(self.weight_hh.T, self.bias_hh)
    i,f,g,o = gates.chunk(4, dim=1)
    i,f,g,o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()

    # these are element-wise, the gates are 0..1 
    c = f * c_prev + i * g 
    h = o * c.tanh()
    # return hidden state and cell state for this timestep
    return (h, c)

# loops over LSTMCell and contains entire network
class LSTM:
  def __init__(self, n_layers:int=4, emb_dim:int=128, hidden:int=128, n_ctx:int=256):
    self.n_ctx = n_ctx
    self.n_layers = n_layers
    self.hidden = hidden
    self.emb = nn.Embedding(tokenizer.vocab_size(), emb_dim)
    self.lstms = [LSTMCell(input_size=(emb_dim if i==0 else hidden), hidden_size=hidden, bias=True) for i in range(n_layers)]
    self.lin = nn.Linear(hidden, tokenizer.vocab_size()) # .shape = (65, 128)
  def __call__(self, x: Tensor, targets: Tensor) -> Tensor:
    x = self.emb(x) 
    
    # run through all lstm cells per timestep, build one hidden state and cell state per LSTM cell 
    logits_ts = [] # logits per time step for loss calculation
    h = [Tensor.zeros(x.size(0), self.hidden) for _ in range(self.n_layers)] 
    c = [Tensor.zeros(x.size(0), self.hidden) for _ in range(self.n_layers)] 

    for t in range(self.n_ctx): # run n_ctx generation attempts (same as T) 
      inp = x[:, t, :] # inp.shape = (B, C) -- select all embeddings @ batch for index t
      for i, cell in enumerate(self.lstms): # run through each lstm 
        h[i], c[i] = cell(inp, (h[i], c[i]))
        inp = h[i]
      logits_ts.append(self.lin(inp))
    
    logits = Tensor.stack(*logits_ts, dim=1)

    return logits.reshape(x.size(0)*x.size(1), tokenizer.vocab_size()).cross_entropy(targets.reshape(x.size(0)*x.size(1)))

if __name__ == "__main__":
  X_train, Y_train, X_test, Y_test = make_blocks(token_ids=tokenizer.encode(dataset.open().read())) 

  model = LSTM()

  opt = nn.optim.AdamW(nn.state.get_parameters(model))

  @TinyJit
  @Tensor.train(True)
  def train_step() -> Tensor:
    opt.zero_grad()
    samples = Tensor.randint(128, high=X_train.shape[0])
    loss = model(X_train[samples], Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())

  @TinyJit
  @Tensor.train(False)
  def get_val_ppl() -> Tensor:
    val_loss = model(X_test, Y_test)
    return val_loss.exp()

  test_ppl = float('nan')
  for i in (t := trange(getenv("STEPS", 1000))):
    GlobalCounters.reset()
    loss = train_step()
    t.set_description(f"loss: {loss.item():6.2f} val_ppl: {test_ppl:.2f}")
    if i % 50 == 49: 
      test_ppl = get_val_ppl().item()
