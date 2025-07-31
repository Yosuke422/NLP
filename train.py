import torch
from torch.utils.data import Dataset, DataLoader
from model import TextGenModel
from utils import train_bpe_tokenizer, load_tokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=32):
        self.tokens = tokenizer.encode(text)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx:idx+self.seq_len]
        y = self.tokens[idx+1:idx+self.seq_len+1]
        return torch.tensor(x), torch.tensor(y)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_bpe_tokenizer("data/corpus.txt")
    tokenizer = load_tokenizer()
    text = open("data/corpus.txt", encoding='utf-8').read()

    dataset = TextDataset(text, tokenizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = TextGenModel(tokenizer.vocab_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(5):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out, _ = model(x)
            loss = loss_fn(out.view(-1, tokenizer.vocab_size()), y.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), "model.pt")

if __name__ == "__main__":
    train()
