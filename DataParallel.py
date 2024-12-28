import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from torch.utils.data import Dataset

from transformers import AutoTokenizer

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.utils import set_seed

MENG_PORT="3007"

class LanguageModelDataset(Dataset):
    def __init__(self, split):
        # Load WikiText-2 dataset
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        if split == "train":
            raw_dataset = dataset["train"].select(range(100))
        elif split == "test":
            raw_dataset = dataset["test"].select(range(10))
        else:
            raw_dataset = dataset["validation"].select(range(20))

        # tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        # def tokenize_function(examples):
        #     # padding to max model input length
        #     return tokenizer(examples["text"], padding="max_length", truncation=True)

        # self.tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

        tokenizer = BPETokenizer()

        self.max_sentence_length = 0
        for sample in raw_dataset:
            sent = sample['text']
            tokens = tokenizer(sent).view(-1)
            self.max_sentence_length = max(len(tokens), self.max_sentence_length)

        self.truncation = 1024
        self.pad_token = 0

        self.data = []
        for sample in raw_dataset:
            sent = sample['text']
            tokens = tokenizer(sent).view(-1)
            paddings = self.pad_token * torch.ones(self.max_sentence_length - len(tokens), dtype=torch.long)
            padded_tokens = torch.cat([tokens, paddings])
            # #print(padded_tokens)
            self.data.append(padded_tokens)

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        x = tokens[:-1]
        y = tokens[1:]
        return (x, y)
    
    def get_vocab_size(self):
        return 50257
    
    def get_block_size(self):
        # max model input length for GPT-2
        return self.max_sentence_length

def train_loop(num_epochs, batch_size):
    train_dataset = LanguageModelDataset(split="train")
    print(len(train_dataset))
    print(train_dataset.get_block_size())

    # val_dataset = LanguageModelDataset(split="val")
    # print(len(val_dataset))
    # print(val_dataset.get_block_size())

    # test_dataset = LanguageModelDataset(split="test")
    # print(len(test_dataset))
    # print(test_dataset.get_block_size())

    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    gpt_model = GPT(model_config)

    model = torch.nn.DataParallel(gpt_model)
    model = model.to("cuda")
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(1, num_epochs + 1):    
        for x, y in train_loader:
            batch_sz = len(x)
            print(f"Train batch of size {batch_sz}")

            x, y = x.to("cuda"), y.to("cuda")
            print(f"Data to GPU")

            optimizer.zero_grad()
            print(f"Zero grad")
            
            logits, loss = model(x, y)
            print(f"Forward pass done")
            print(loss)
            
            loss = loss.mean()
            print(f"Average loss across GPUs")
            loss.backward()
            print(f"Backward pass done")

            optimizer.step()
            print(f"Optimizer done")

        print(f"Epoch {epoch} done, Loss: {loss.item()}")

if __name__ == "__main__":
    WORLD_SIZE = torch.cuda.device_count()
    print(f"We have {WORLD_SIZE} GPUs")

    train_loop(1, 16)