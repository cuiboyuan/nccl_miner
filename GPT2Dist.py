import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import load_dataset
from torch.utils.data import Dataset

from transformers import AutoTokenizer

import torch.multiprocessing as mp

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.utils import set_seed 

MENG_PORT="3002"

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
    
def train(model, rank, world_sz, loader, optimizer, ep):
    model.train()
    
    distributed_loss = torch.zeros(2).to(rank)

    for x, y in loader:
        batch_sz = len(x)
        #print(f"Rank {rank} train batch of size {batch_sz}")
        

        x, y = x.to(rank), y.to(rank)
        #print(f"Rank {rank} data to GPU")
        
        optimizer.zero_grad()
        #print(f"Rank {rank} zero grad")
        
        logits, loss = model(x, y)
        #print(f"Rank {rank} forward pass done")
        
        loss.backward()
        #print(f"Rank {rank} backward pass done")
        
        optimizer.step()
        #print(f"Rank {rank} Optimizer done")
        

        distributed_loss[0] += loss.item()
        distributed_loss[1] += batch_sz
        
        #print(f"Rank {rank} \t Current Loss: {distributed_loss[0]/distributed_loss[1]}")
        
    
    # dist.all_reduce(distributed_loss, op=dist.ReduceOp.SUM)
    # if rank == 0:
    #     dist.send(distributed_loss, dst=1)
    #     #print(f"Train Epoch: {ep} \t Loss: {distributed_loss[0]/distributed_loss[1]}")
        
    # else:
    #     dist.recv(distributed_loss, src=0)
    #     #print(f"Train Epoch: {ep} \t Loss: {distributed_loss[0]/distributed_loss[1]}")
        


def validate(model, rank, world_sz, loader):
    model.eval()

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MENG_PORT

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def fsdp_main(rank, world_size, train_args):
    print(f"Rank {rank} starts")
    
    batch_size, num_epochs, save_model = train_args['batch_size'], train_args['num_epochs'], train_args['save_model']
    
    setup(rank, world_size)

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                              batch_size=batch_size,
                              sampler=val_sampler)
    
    # torch.cuda.set_device(rank)

    # torch.cuda.empty_cache()

    # create a GPT instance

    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    gpt_model = GPT(model_config)

    model = gpt_model.to(rank)
    model = DDP(model, device_ids=[rank])
    # model = FSDP(model)
    optimizer = optim.Adam(model.parameters())
    #print(f"Rank {rank} Start training")
    
    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train(model=model, rank=rank, world_sz=world_size, loader=train_loader, optimizer=optimizer, ep=epoch)
        #print(f"Rank {rank} epoch {epoch} done")
        
        # validate(model=model, rank=rank, world_sz=world_size, loader=val_loader)

    if save_model:
        dist.barrier()
        if rank == 0:
            torch.save(model.state_dict, "/workspace/mingpt.pt")

    cleanup()
    print(f"Rank {rank} ends")
    


if __name__ == "__main__":
    train_dataset = LanguageModelDataset(split="train")
    print(len(train_dataset))
    print(train_dataset.get_block_size())

    val_dataset = LanguageModelDataset(split="val")
    print(len(val_dataset))
    print(val_dataset.get_block_size())

    test_dataset = LanguageModelDataset(split="test")
    print(len(test_dataset))
    print(test_dataset.get_block_size())

    WORLD_SIZE = torch.cuda.device_count()
    print(f"We have {WORLD_SIZE} GPUs")

    train_args = {
        'batch_size': 2,
        'num_epochs': 10,
        'save_model': False,
    }
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"

    processes = []
    # mp.set_start_method("spawn")
    for rank in range(WORLD_SIZE):
        p = mp.Process(target=fsdp_main, args=(rank, WORLD_SIZE, train_args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()