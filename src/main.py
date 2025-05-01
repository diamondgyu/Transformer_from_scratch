'''
Trainer Function
    1) Load Data
    2) Initialize Model
    3) Train

Hyperparameters from the Paper(Attention is all you need):

Model Architecture
    Number of encoder layers: 6
    Number of decoder layers: 6
    Number of attention heads: 8
    Embedding size: 512
    Feed-forward network (FFNN) size: 2048
    Dropout rate: 0.1
Optimizer
    Optimizer: Adam
    Learning rate: 1e-4
    Beta1: 0.9
    Beta2: 0.98
    Epsilon: 1e-9
Training
    Batch size: 4096 tokens (approximately 256 sequences of length 16)
    Sequence length: 16 (for training) and 50 (for evaluation)
    Training steps: 100,000
    Warmup steps: 10,000
Other
    Label smoothing: 0.1
    Weight decay: 0.01
'''

import torch
from tqdm import tqdm

# WandB is used for logging
# and tracking the training process
# and results
import wandb
from model import TransformerModel
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from test import generate, calculate_bleu_score
import logging
from downloader import download_data
import pathlib

path = pathlib.Path(__file__).parent.parent

language = 'de'

bos_token = {'bos_token': '<s>'}
tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.add_special_tokens(bos_token)
device_name = 'cuda:2'

with open(path / 'logs/sample_generation.log', 'w') as f:
    pass

logger_main = logging.getLogger(__name__)
logging.basicConfig(filename=path / 'logs/sample_generation.log', level=logging.DEBUG)
logger_main.info("logging started")

    

def train_model(model, train, valid, device, vocab_size, run_name, length_data, tokenizer_max_len, epochs=5):

    count = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-5)
    
    warmup_steps = 4000
    def lr_lambda(current_step:int):
        current_step += 1
        scale = 1024 ** -0.5
        warmup = current_step * (warmup_steps ** -1.5)
        decay = current_step ** -0.5
        wandb.log({'lr': scale * min(warmup, decay)}, step=count)
        return scale * min(warmup, decay)
    
    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    criterion_valid = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    valid_loss = 0
    best_loss = 100000
    

    scaler = torch.amp.GradScaler(device_name)

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        
        batches = tqdm(train, dynamic_ncols=False, mininterval=200, miniters=200)
        for batch in batches:
            count += 1
            model.train()

            src = batch['input_ids'].to(device, non_blocking=True).contiguous()
            tgt = batch['labels'].to(device, non_blocking=True).contiguous()

            tgt_input = torch.zeros_like(tgt).to(device)
            tgt_input[:, 1:] = tgt[:,:-1]
            tgt_input[:, 0] = tokenizer.bos_token_id
            
            with torch.amp.autocast(device_type=device_name):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))


            wandb.log({"Train Loss": loss}, step=count)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if count % 1000 == 1:

                model.eval()
                
                with torch.no_grad():
                    output_tf = tokenizer.decode(torch.argmax(model(src, tgt_input), dim=-1)[0], skip_special_tokens=True)
                    result = generate(model, src[0].unsqueeze(0), tokenizer, max_len=tokenizer_max_len)
                    bleu_score = calculate_bleu_score(result, batch['translation']['en'][0])
                    wandb.log({"BLEU Score": bleu_score}, step=count)
                    result = tokenizer.batch_decode(result, skip_special_tokens=True)[0]
                    logger_main.info('')
                    logger_main.info('--------------------------------------------------------')
                    logger_main.info('')
                    logger_main.info('Epoch {}, Batch {}'.format(epoch+1, count%len(train)))
                    logger_main.info('Original Text:')
                    logger_main.info(batch['translation'][language][0])
                    logger_main.info('')
                    logger_main.info('Ideal Output:')
                    logger_main.info(batch['translation']['en'][0])
                    logger_main.info('')
                    logger_main.info('Output(teacher forced):')
                    logger_main.info(output_tf)
                    logger_main.info('')
                    logger_main.info('Generated Text: ')
                    logger_main.info(result)

                # Due to the dropout layer and some reasons, the validation loss can be smaller than the training loss
                # To show this, turn off the model.eval() and calculate the validation loss with dropout

                total_valid_loss = 0
                with torch.no_grad():
                    for valid_batch in valid:
                        src = valid_batch['input_ids'].to(device, non_blocking=True)
                        tgt = valid_batch['labels'].to(device, non_blocking=True)
                        tgt_input = torch.full_like(tgt, 0)
                        tgt_input[:, 1:] = tgt[:, :-1]
                        tgt_input[:, 0] = tokenizer.bos_token_id
                        
                        with torch.amp.autocast(device_type=device_name):
                            output = model(src, tgt_input)
                            total_valid_loss += criterion_valid(
                                output.view(-1, vocab_size),
                                tgt.view(-1)
                            ).detach()
                            
                valid_loss = total_valid_loss / len(valid)
                wandb.log({"Valid Loss": valid_loss}, step=count)

                # save model if improved
                if valid_loss < best_loss or count%len(batches) == 0:
                    torch.save(model.state_dict(), \
                        path/'models/{}/model-checkpoint-epoch{}-iter{}-{}.mdl'.format(run_name, epoch+1, count-1, length_data))
                    best_loss = valid_loss
            
            batches.set_description(f"Loss: {loss:.6f} / Valid loss: {valid_loss:.6f}")


def test_model(model, test, device, vocab_size):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        total_loss = 0
        count = 0
        for batch in tqdm(test, desc="Testing"):
            count += 1
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            tgt_input = torch.full_like(tgt, 0)
            tgt_input[:, 1:] = tgt[:, :-1]
            tgt_input[:, 0] = tokenizer.bos_token_id

            output = model(src, tgt_input)
            loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))

            total_loss += loss.item()
            if count == 100:
                break

        avg_loss = total_loss / len(test)
        print(f"Test Loss: {avg_loss:.4f}")
        wandb.log({"Test Loss": avg_loss})


def test():

    # Hyperparameters
    # num_heads 16, hidden_layers 4096, train_steps 300k for the same result with the paper
    # Training 300K -> dataset 4.5M (wmt14 en-de)
    embed_dim = 1024
    num_heads = 16  
    dropout_rate = 0.1
    hidden_layer_dim = 1024*3
    tokenizer_max_len = 128
    vocab_size = 32129
    stacks = 6
    batch_size = 64
    train_on_short_sentences = False
    len_train = batch_size*50*1000
    epochs = 5
    run_name = '1024_2048_50k*64'
    
    # Download dataset
    train, valid, test = download_data(batch_size=batch_size, tokenizer_max_len=tokenizer_max_len, len_train=len_train, short_sentences=train_on_short_sentences)

    # Initialize wandb
    wandb.init(project='transformer_from_scratch', name=run_name)

    # Create model
    model = TransformerModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        hidden_layer_dim=hidden_layer_dim,
        max_len=tokenizer_max_len,
        vocab_size=vocab_size,
        stacks=stacks,
    )

    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(device)

    # Load model
    # model.load_state_dict(torch.load(path/'models/{}/model-checkpoint-epoch2-iter172000-short.mdl'.format(run_name), map_location=device, weights_only=True))

    # Train
    train_model(model, train, valid, device, vocab_size, epochs=epochs, run_name=run_name, tokenizer_max_len=tokenizer_max_len, length_data='short' if train_on_short_sentences else 'long')
    
    # Test
    test_model(model, test, device, vocab_size)

if __name__ == "__main__":
    test()  