from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
import tinygrad.nn as nn
from tinygrad.helpers import GlobalCounters, getenv, trange
from typing import Tuple, List
from pathlib import Path
from utils import make_blocks

# ---------------- Tokenizer ----------------
class CharTokenizer:
    def __init__(self, path: Path):
        txt = path.read_text()
        self.unique = sorted(set(txt))
        self.vocab = {ch: i for i, ch in enumerate(self.unique)}
        self.r_vocab = {i: ch for i, ch in enumerate(self.unique)}
    def encode(self, s: str): return [self.vocab[x] for x in s]
    def decode(self, ids): return "".join(self.r_vocab[i] for i in ids)
    def vocab_size(self): return len(self.unique)

dataset = Path(__file__).parent.parent / "datasets" / "shakespeare.txt"
tokenizer = CharTokenizer(dataset)

# -------------- LSTM (per-step JIT like rnnt) --------------
class LSTMCell:
    def __init__(self, input_size:int, hidden_size:int, dropout:float):
        self.hidden_size = hidden_size
        self.dropout = dropout
        # One linear for all 4 gates on concatenated (x,h)
        self.xh2g = nn.Linear(input_size + hidden_size, 4*hidden_size, bias=True)

    def __call__(self, x: Tensor, hc_flat: Tensor) -> Tensor:
        # hc_flat shape (2B, H): first B rows = h, second B rows = c
        B = x.shape[0]
        h = hc_flat[:B]
        c = hc_flat[B:]
        gates = self.xh2g(x.cat(h, dim=1))
        i, f, g, o = gates.chunk(4, 1)
        i, f, g, o = i.sigmoid(), f.sigmoid(), g.tanh(), o.sigmoid()
        c_new = f * c + i * g
        h_new = o * c_new.tanh()
        if self.dropout and Tensor.training: h_new = h_new.dropout(self.dropout)
        return h_new.cat(c_new, dim=0)

class LSTM:
    def __init__(self, vocab_size:int, embed_size:int, hidden_size:int, layers:int=1, dropout:float=0.0):
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Dropout on all but last layer (mirroring rnnt style)
        self.cells = [LSTMCell(embed_size if l == 0 else hidden_size,
                               hidden_size,
                               dropout if l != layers-1 else 0.0)
                      for l in range(layers)]
        self.proj = nn.Linear(hidden_size, vocab_size, bias=True)
        self.hidden_size = hidden_size
        self.layers = layers

        def _step(x_t: Tensor, hc: Tensor):
            # hc shape (L, 2B, H)
            new_layers: List[Tensor] = []
            cur = x_t
            for l, cell in enumerate(self.cells):
                hc_l = hc[l]
                hc_l_new = cell(cur, hc_l)        # (2B,H)
                new_layers.append(hc_l_new)
                cur = hc_l_new[:x_t.shape[0]]     # next layer input = new h
            # ensure stacked hidden state has stable contiguous strides for JIT cache
            return Tensor.stack(*new_layers).contiguous()
        self._step = _step  # store JITed function

    def __call__(self, X: Tensor, hc: Tensor | None = None) -> Tuple[Tensor, Tensor]:
        # X: (B,T)
        B, T = X.shape
        E = self.embed(X)              # (B,T,E)
        E = E.transpose(0,1).contiguous()  # (T,B,E)
        if hc is None:
            # NOTE: ensure a fresh realized contiguous buffer so JIT sees identical strides every invocation.
            # Without this, some Tensor.zeros implementations may return a broadcasted (stride=0) layout causing JIT mismatch.
            hc = Tensor.zeros(self.layers, 2*B, self.hidden_size).contiguous()
        logits = None
        for t in range(T):
            # +1-1 trick as in rnnt to avoid certain fusions
            # ensure hc fed to JIT step also retains standard contiguous layout
            hc = self._step(E[t] + 1 - 1, hc.contiguous())
            h_last = hc[-1, :B]             # (B,H)
            logit_t = self.proj(h_last)     # (B,V)
            if logits is None:
                logits = logit_t.unsqueeze(1)
            else:
                # Realize after each concat to bound graph size
                logits = logits.cat(logit_t.unsqueeze(1), dim=1)
        return logits, hc  # logits (B,T,V)

# ---------------- Main Training Script ----------------
if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = make_blocks(token_ids=tokenizer.encode(dataset.read_text()))
    # Shapes: (N, T)

    model = LSTM(vocab_size=tokenizer.vocab_size(), embed_size=256, hidden_size=128, layers=2, dropout=0.1)
    params = nn.state.get_parameters(model)
    opt = nn.optim.AdamW(params, lr=3e-4)

    BATCH = getenv("BATCH", 128)

    @Tensor.train()
    def fwd_bwd():
        opt.zero_grad()
        idx = Tensor.randint(BATCH, high=X_train.shape[0])
        Xb, Yb = X_train[idx], Y_train[idx]         # (B,T)
        logits, _ = model(Xb)                       # (B,T,V)
        B, T, V = logits.shape
        loss = logits.reshape(B*T, V).sparse_categorical_crossentropy(Yb.reshape(B*T)).mean().backward()
        return loss.realize(*opt.schedule_step())

    @TinyJit
    @Tensor.train(False)
    def val_ppl():
        logits, _ = model(X_test)
        B, T, V = logits.shape
        vloss = logits.reshape(B*T, V).sparse_categorical_crossentropy(Y_test.reshape(B*T)).mean()
        return vloss.exp()

    test_ppl = float('nan')
    for i in (t := trange(getenv("STEPS", 1000))):
        GlobalCounters.reset()
        loss = fwd_bwd()
        t.set_description(f"loss: {loss.item():6.2f} val_ppl: {test_ppl:.2f}")
        if i % 50 == 49:
            test_ppl = val_ppl().item()
