import torch.utils.data

from utils_generating import ResonseDataset,SPECIAL_TOKENS
from transformers import BertTokenizer
import argparse
import os
from tqdm import tqdm,trange
from transformers import GPT2LMHeadModel,AdamW,get_linear_schedule_with_warmup

def train(train_data,model):

    global_step = 0
    train_sampler = torch.utils.data.RandomSampler(train_data)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,sampler=train_sampler,batch_size=args.train_batch_size,collate_fn=train_data.collate_fn)
    train_iterator = trange(
        0, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    t_total = len(train_loader) // args.gradient_accumulation_steps * args.num_train_epochs
    optimizer = AdamW(model.parameters(),lr=args.learning_rate,eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=t_total
    )
    model.zero_grad()

    for _ in train_iterator:
        local_steps,tr_loss = 0,0
        epoch_iterator = tqdm(train_loader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, token_type_ids, lm_labels = batch
            model_outputs = model(input_ids=input_ids, token_type_ids=None, labels=lm_labels)
            loss = model_outputs[0]
            lm_logits = model_outputs[1]
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                local_steps += 1
                epoch_iterator.set_postfix(Loss=tr_loss / local_steps)
        # result = evaluate()

    print(len(train_data))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default=os.path.join(os.path.abspath("."),"data"),help="folder path which contains several tasks")
    parser.add_argument("--task_name",type=str,default="base")
    parser.add_argument("--vocab_path", type=str, default="./vocab/vocab.txt")
    parser.add_argument('--max_len', type=int, default=512, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument("--history_max_tokens", type=int, default=224,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--resp_max_tokens", type=int, default=400,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--knowledge_max_tokens", type=int, default=350,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--num_train_epochs", type=int, default=4,
                        help="Maximum length in tokens for history, will override that value in config.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    args = parser.parse_args()
    args.model_name_or_path = r"E:\MyPython\Pre-train-Model\uergpt2-chinese-cluecorpussmall"
    args.device = r"cpu"
    args.gradient_accumulation_steps = 1
    args.learning_rate = 6.25e-5
    args.adam_epsilon = 1e-8
    args.warmup_steps = 0
    args.max_grad_norm = 1.0


    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    train_data = ResonseDataset(args,tokenizer)

    train(train_data,model)




