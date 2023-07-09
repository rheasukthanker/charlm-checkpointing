from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union
import matplotlib.pyplot as plt

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, block_size: int, embed_size: int, head_size: int):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        embed_size: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        """
        super().__init__()

        self.block_size = block_size  # equivalent to T
        self.embed_size = embed_size  # equivalent to C
        self.head_size = head_size

        self.key = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.query = nn.Linear(self.embed_size, self.head_size, bias=False)
        self.value = nn.Linear(self.embed_size, self.head_size, bias=False)

        self.register_buffer(
            'tril', torch.tril(torch.ones(self.block_size, self.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape  # B: batch size; T: block size; C: embedding size
        k = self.key(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        q = self.query(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1)  # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)

        # performing `scaled` attention
        wei *= self.head_size ** -(1 / 2)  # scaling by `1/sqrt(head size)`

        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C) @ (C, head_size)  -> (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(
        self, block_size: int, embed_size: int, head_size: int, num_heads: int
    ):
        """
        Arguments
        ---------
        block_size: int
            The sentence length allowed
        embed_size: int
            The size (dimension) of the token embeddings
        head_size: int
            The size (dimension) of the output of an attention head
        num_heads: int
            The number of single attention heads that together form
            one multi-headed attention layer
        """
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, embed_size, head_size) for _ in range(num_heads)]
        )
        # linear FC layer
        self.proj = nn.Linear(head_size * num_heads, embed_size)

    def forward(self, x):
        # simply stack multiple heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # B: batch size; T: block size; C: embedding size; H: head_size * num_heads
        out = self.proj(out)  # (B, T, H) @ (H, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(
            self,
            embed_size: int,
            wide_factor: int = 4,
            activation: str = "relu",
            dropout: float = 0.0
        ):
        super().__init__()
        self.activation = torch.nn.ReLU if activation == "relu" else torch.nn.GELU
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(embed_size, wide_factor * embed_size),
            self.activation(),
            nn.Linear(wide_factor * embed_size, embed_size),
            nn.Dropout(self.dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(
        self,
        block_size: int,
        embed_size: int,
        num_heads: int,
        wide_factor: int = 4,
        activation: str = "relu",  # could also be "gelu"
        dropout: float = 0.0,
        prenormalize: bool = False
    ):
        super().__init__()
        # setting head_size to be a factor of other dimensions
        head_size = embed_size // num_heads
        # the multi-headed self-attention (msa)
        self.msa = MultiHeadAttention(block_size, embed_size, head_size, num_heads)
        self.ffwd = FeedForward(embed_size, wide_factor, activation, dropout)

        self.prenormalize = prenormalize
        if prenormalize:
            self.pre_ln = nn.LayerNorm(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        if self.prenormalize:
            # normalizes inputs before passing it through the attention block
            x = x + self.msa( self.pre_ln(x) )
        else:
            x = x + self.msa(x)
        # norm after attention
        x = self.ln1(x)
        # feed-forward
        x = x + self.ffwd(x)
        # norm after feed-forward
        x = self.ln2(x)
        return x
    
class CharLM(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            n_layers: int,
            block_size: int,
            embed_size: int,
            num_heads: int,
            wide_factor: int = 4,
            activation: str = "relu",  # could also be "gelu"
            dropout: float = 0.0,
            prenormalize: bool = False,
            device: str = None,
            **kwargs
    ):
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = "gpu" if torch.cuda.is_available() else "cpu"
        # each token directly reads off the logits for the next
        # token from a lookup table
        # Note attention does not have any notion of colocation
        # of characters/words and this is important for lms
        self.token_embedding_table = nn.Embedding(vocab_size, embed_size)  # , device=self.device)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)  # , device=self.device)
        self.blocks = nn.Sequential(
            *[Block(
                block_size=block_size,
                embed_size=embed_size,
                num_heads=num_heads,
                wide_factor=wide_factor,
                activation=activation,
                dropout=dropout,
                prenormalize=prenormalize,
            ) for _ in range(n_layers)]  # stacks the layers of Transformer blocks
        )
        self.ln_f = nn.LayerNorm(embed_size)  #, device=self.device)  # final layer norm (has bias)
        self.lm_head = nn.Linear(embed_size, vocab_size)  #, device=self.device)


    def forward(self, idx, targets=None):
        # B: batch_size, T: block_size, C: embedding_size
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        # fixing positional inputs and learning an embedding over it
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        # adding the positional embeddings across the token embedding batch
        x = tok_emb + pos_emb  # (B,T,C)
        # forward pass through the Transformer layers
        x = self.blocks(x)  # (B,T,C)
        # final layernorm
        x = self.ln_f(x)  # (B,T,C)
        # projecting to the vocabulary
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, block_size, verbose: bool = False):
        # B: batch_size, T: block_size, C:
        # idx is (B, T) array of indices in the current context

        self.eval()

        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, vocab_size)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        self.train()
        return idx
    
def plot_loss(losses):
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__main__":
    # testing the model
    model = CharLM(
        vocab_size=100,
        n_layers=2,
        block_size=10,
        embed_size=16,
        num_heads=2,
        wide_factor=4,
        activation="relu",
        dropout=0.0,
        prenormalize=False,
        device="cpu",
    )
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(500):
        idx = torch.randint(0, 100, (64, 10))
        logits, loss = model(idx, targets=idx)
        print(logits.shape, loss)
        optimizer.zero_grad()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    #save model
    state_dict = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
    torch.save(state_dict, "model.pt")
    #load model
    state_dict = torch.load("model.pt")
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    epoch = state_dict["epoch"]
    # Check loss now
    idx = torch.randint(0, 100, (64, 10))
    logits, loss = model(idx, targets=idx)
    print("Loss after loading model")
    print(logits.shape, loss)
    # Start training again
    for epoch in range(epoch, 1000):
        idx = torch.randint(0, 100, (64, 10))
        logits, loss = model(idx, targets=idx)
        print(logits.shape, loss)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    plot_loss(losses)
    #Save plot
    plt.savefig("loss.png")