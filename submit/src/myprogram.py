import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from datasets import load_dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Hyperparameters
EMB_SIZE = 128
NHEAD = 4
NUM_LAYERS = 3
HIDDEN_DIM = 512
SEQ_LEN = 128
BATCH_SIZE = 64
EPOCHS = 5
LR = 2e-4
SAMPLE_FRACTION = 0.33  # Use 33% of dataset each epoch

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

# Enforce CUDA
assert DEVICE == 'cuda', "❌ CUDA not available — training will be slow. Aborting."
print(f"✅ Using device: {DEVICE}")


class CharDataset(Dataset):
    def __init__(self, text, seq_len, char2idx):
        self.text = text
        self.seq_len = seq_len
        self.char2idx = char2idx

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.text[idx:idx + self.seq_len]
        target_char = self.text[idx + self.seq_len]
        input_tensor = torch.tensor([self.char2idx[c] for c in input_seq], dtype=torch.long)
        target_tensor = torch.tensor(self.char2idx[target_char], dtype=torch.long)
        return input_tensor, target_tensor


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, emb_size, nhead, num_layers, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc_out = nn.Linear(emb_size, vocab_size)

    def forward(self, src):
        embedded = self.embedding(src)
        output = self.transformer_encoder(embedded)
        logits = self.fc_out(output[:, -1, :])
        return logits


class MyModel:
    model = None
    char2idx = None
    idx2char = None
    vocab = None

    @classmethod
    def load_training_data(cls):
        dataset = load_dataset("wikipedia", "20220301.simple", split='train[:1%]', trust_remote_code=True)
        text_data = " ".join(dataset['text']).replace('\n', ' ')
        text_data = text_data[:len(text_data) // 1]  # Optional reduction
        cls.vocab = sorted(list(set(text_data)))
        cls.char2idx = {ch: i for i, ch in enumerate(cls.vocab)}
        cls.idx2char = {i: ch for ch, i in cls.char2idx.items()}
        return text_data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line.rstrip('\n')
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write(f"{p}\n")

    def run_train(self, data, work_dir):
        dataset = CharDataset(data, SEQ_LEN, self.char2idx)
        half_dataset_len = int(SAMPLE_FRACTION * len(dataset))

        self.model = CharTransformer(len(self.char2idx), EMB_SIZE, NHEAD, NUM_LAYERS, HIDDEN_DIM).to(DEVICE)
        print(f"Model is on device: {next(self.model.parameters()).device}")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LR)

        self.model.train()
        for epoch in range(EPOCHS):
            # Re-sample a new random half of the dataset each epoch
            sampler = RandomSampler(dataset, replacement=False, num_samples=half_dataset_len)
            data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

            total_loss = 0
            for batch_idx, (input_seq, target) in enumerate(data_loader):
                input_seq, target = input_seq.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = self.model(input_seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(data_loader)}], "
                          f"Loss: {total_loss / (batch_idx + 1):.4f}")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'char2idx': self.char2idx,
            'idx2char': self.idx2char
        }, os.path.join(work_dir, 'model.checkpoint'))

    def run_pred(self, data):
        preds = []
        self.model.eval()
        for context in data:
            context = context[-SEQ_LEN:]
            context_encoded = torch.tensor([[self.char2idx.get(c, 0) for c in context]], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                logits = self.model(context_encoded)
                probabilities = torch.softmax(logits, dim=-1)
                top_probs, top_idxs = probabilities.topk(3)
                top_chars = [self.idx2char[idx.item()] for idx in top_idxs[0]]
                preds.append("".join(top_chars))
        return preds

    def save(self, work_dir):
        pass  # Already saved in run_train

    @classmethod
    def load(cls, work_dir):
        checkpoint = torch.load(os.path.join(work_dir, 'model.checkpoint'), map_location=DEVICE)
        model = MyModel()
        model.char2idx = checkpoint['char2idx']
        model.idx2char = checkpoint['idx2char']
        model.vocab = sorted(model.char2idx.keys())
        model.model = CharTransformer(len(model.char2idx), EMB_SIZE, NHEAD, NUM_LAYERS, HIDDEN_DIM).to(DEVICE)
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.eval()
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')

