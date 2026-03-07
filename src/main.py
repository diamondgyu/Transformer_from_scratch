'''
Trainer Function
    1) Load Data
    2) Initialize Model
    3) Train

Hyperparameters from the Paper (Attention is all you need):

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
    Batch size: 48
    Sequence length: 16 (for training) and 50 (for evaluation)
    Training steps: 100,000
    Warmup steps: 10,000
Other
    Label smoothing: 0.1
    Weight decay: 0.01
'''

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from tqdm import tqdm

import wandb
from model import TransformerModel
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import LambdaLR
from test import generate, calculate_bleu_score
import logging
from process_korean_data import download_data
import pathlib

path = pathlib.Path(__file__).parent.parent


language = 'ko'

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

with open(path / 'logs/sample_generation.log', 'w') as f:
    pass

logger_main = logging.getLogger(__name__)
logging.basicConfig(filename=path / 'logs/sample_generation.log', level=logging.DEBUG, encoding='utf-8')
logger_main.info("logging started")


def train_model(model, train, valid, device, vocab_size, run_name, tokenizer_max_len, epochs=5):

    count = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-5)

    warmup_steps = 8000
    def lr_lambda(current_step: int):
        current_step += 1
        scale = 0.4 * 512 ** -0.5
        warmup = current_step * (warmup_steps ** -1.5)
        decay = current_step ** -0.5
        wandb.log({'lr': scale * min(warmup, decay)}, step=count)
        return scale * min(warmup, decay)

    scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)
    criterion_valid = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    valid_loss = 0
    best_loss = 100000

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')

        batches = tqdm(train, dynamic_ncols=False)
        for batch in batches:
            count += 1
            model.train()

            src = batch['input_ids'].to(device, non_blocking=True).contiguous()
            tgt = batch['labels'].to(device, non_blocking=True).contiguous()

            # Build tgt_input on GPU directly – no redundant zeros_like + .to(device)
            bos_column = torch.full((tgt.size(0), 1), tokenizer.bos_token_id,
                                    device=device, dtype=tgt.dtype)
            tgt_input = torch.cat([bos_column, tgt[:, :-1]], dim=1)

            use_amp = torch.cuda.is_available()
            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp, dtype=torch.bfloat16):
                output = model(src, tgt_input)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))

            wandb.log({"Train Loss": loss}, step=count)

            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
            
            

            if count % 1000 == 1:

                model.eval()

                with torch.no_grad():
                    output_tf = tokenizer.decode(
                        torch.argmax(model(src, tgt_input), dim=-1)[0],
                        skip_special_tokens=True,
                    )
                    result = generate(model, src[0].unsqueeze(0), tokenizer, max_len=tokenizer_max_len)
                    result = tokenizer.batch_decode(result, skip_special_tokens=True)[0]
                    bleu_score = calculate_bleu_score(result, batch['translation']['en'][0])
                    wandb.log({"BLEU Score": bleu_score}, step=count)

                    logger_main.info('')
                    logger_main.info('--------------------------------------------------------')
                    logger_main.info('')
                    logger_main.info('Epoch {}, Batch {}'.format(epoch + 1, count % len(train)))
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

                # Sampled validation (fixed N batches to avoid stalling training)
                n_valid_batches = 50
                total_valid_loss = 0
                with torch.no_grad():
                    for i, valid_batch in enumerate(valid):
                        if i >= n_valid_batches:
                            break
                        v_src = valid_batch['input_ids'].to(device, non_blocking=True)
                        v_tgt = valid_batch['labels'].to(device, non_blocking=True)
                        bos_col = torch.full((v_tgt.size(0), 1), tokenizer.bos_token_id,
                                             device=device, dtype=v_tgt.dtype)
                        v_tgt_input = torch.cat([bos_col, v_tgt[:, :-1]], dim=1)

                        use_amp = torch.cuda.is_available()
                        with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp, dtype=torch.bfloat16):
                            v_output = model(v_src, v_tgt_input)
                            total_valid_loss += criterion_valid(
                                v_output.view(-1, vocab_size),
                                v_tgt.view(-1),
                            ).item()

                valid_loss = total_valid_loss / n_valid_batches
                wandb.log({"Valid Loss": valid_loss}, step=count)

                # Save model in bf16 if improved (or at end of epoch)
                if valid_loss < best_loss or count % len(batches) == 0:
                    save_path = path / 'models/{}/model-checkpoint-epoch{}-iter{}-{}.mdl'.format(
                        run_name, epoch + 1, count - 1, 'all',
                    )
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    TransformerModel.save_bf16(model, save_path)
                    best_loss = valid_loss

            batches.set_description(f"Loss: {loss:.6f} / Valid loss: {valid_loss:.6f}")


def test_model(model, test, device, vocab_size):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    with torch.no_grad():
        total_loss = 0
        count = 0
        for batch in tqdm(test, desc="Testing"):
            count += 1
            src = batch['input_ids'].to(device)
            tgt = batch['labels'].to(device)
            bos_column = torch.full((tgt.size(0), 1), tokenizer.bos_token_id,
                                    device=device, dtype=tgt.dtype)
            tgt_input = torch.cat([bos_column, tgt[:, :-1]], dim=1)

            use_amp = torch.cuda.is_available()
            with torch.amp.autocast(device_type='cuda' if use_amp else 'cpu', enabled=use_amp, dtype=torch.bfloat16):
                output = model(src, tgt_input)
            loss = criterion(output.view(-1, vocab_size), tgt.reshape(-1))

            total_loss += loss.item()
            if count == 100:
                break

        avg_loss = total_loss / count
        print(f"Test Loss: {avg_loss:.4f}")
        wandb.log({"Test Loss": avg_loss})


def test():

    # Hyperparameters
    # num_heads 16, hidden_layers 4096, train_steps 300k for the same result with the paper
    # Training 300K -> dataset 4.5M (wmt14 en-de)
    embed_dim = 512
    num_heads = 16
    dropout_rate = 0.1
    hidden_layer_dim = 2048
    tokenizer_max_len = 128
    vocab_size = len(tokenizer)
    stacks = 8
    batch_size = 48
    len_train = -1
    epochs = 5
    run_name = 'transformer_ko_en_base'

    # Download dataset
    train, valid, test = download_data(
        batch_size=batch_size, tokenizer_max_len=tokenizer_max_len,
        len_train=len_train,
    )

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
        pad_token_id=tokenizer.pad_token_id,
    )

    device = torch.device(device_name if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(device)

    # To load an existing checkpoint
    # TransformerModel.load_bf16(model, 'models/.../checkpoint.mdl', device=device)
    
    # Train
    train_model(
        model, train, valid, device, vocab_size,
        epochs=epochs, run_name=run_name, tokenizer_max_len=tokenizer_max_len,
    )

    # Test
    test_model(model, test, device, vocab_size)

if __name__ == "__main__":
    test()  