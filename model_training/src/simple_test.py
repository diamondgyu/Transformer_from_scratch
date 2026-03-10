
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoTokenizer
from model import TransformerModel
import sacrebleu
import pathlib

def calculate_bleu_score(generated_sentence, label):
    """Sentence-level BLEU via sacrebleu (returns 0-1 scale)."""
    score = sacrebleu.sentence_bleu(generated_sentence, [label])
    return score.score / 100.0


def calculate_corpus_bleu(generated_sentences, references):
    """Corpus-level BLEU via sacrebleu (returns 0-100 scale)."""
    score = sacrebleu.corpus_bleu(generated_sentences, [references])
    return score.score


def cut_string_between_bos_eos(string, bos='[CLS]', eos='[SEP]'):
    bos_index = string.find(bos)
    eos_index = string.find(eos)
    if bos_index == -1:
        bos_index = -len(bos)
    if eos_index == -1:
        eos_index = len(string)
    return string[bos_index + len(bos):eos_index].strip()


# Example Usage
if __name__ == "__main__":

    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token
    
    path = pathlib.Path(__file__).parent.parent.resolve()

    # This is KR->EN translation.
    local_model_path = path / 'models' / 'model-pretrained.pt'
    if not local_model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {local_model_path}")

    embed_dim = 512
    num_heads = 16
    dropout_rate = 0.1
    hidden_layer_dim = 1362
    tokenizer_max_len = 128
    vocab_size = len(tokenizer)
    stacks = 8

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
        tokenizer=tokenizer
    )

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Load checkpoint (strict=False to handle new register_buffer keys in old checkpoints)
    model.to(device)
    TransformerModel.load_bf16(model, local_model_path, device=device) # type: ignore

    file = open(path / 'output_text_pairs.txt', 'w')
    bleu_scores = []

    # Interactive mode
    while True:
        input_text1 = input('Enter Korean sentence (type \'quit\' to quit): ')
        if input_text1 == 'quit':
            break
        
        # reference_text = input('Enter Reference English sentence (optional, press Enter to skip): ')

        test_input = tokenizer(
            [input_text1], padding='max_length', truncation=True,
            return_tensors="pt", max_length=tokenizer_max_len,
        ).to(device=device)

        output_sentences = model.beam_generate(test_input['input_ids'],
                                               max_len=tokenizer_max_len)
        output_sentences = tokenizer.batch_decode(output_sentences, skip_special_tokens=False)
        translation = cut_string_between_bos_eos(output_sentences[0])

        print('Translation:', translation)
        
        print()

    '''  ------------------------------------------------------------------------------------------------  '''

    # Batch evaluation with corpus-level BLEU (sacrebleu)
    # _, valid, test_data = download_data(
    #     batch_size=32, tokenizer_max_len=128, len_train=1,
    #     test_ratio=0.99,
    # )

    # all_generated = []
    # all_references = []

    # for batch in tqdm(test_data):
    #     input_sentences = batch['input_ids'].to(device, non_blocking=True).contiguous()
    #     output_sentences = model.beam_generate(input_sentences, max_len=tokenizer_max_len)

    #     output_texts = tokenizer.batch_decode(output_sentences, skip_special_tokens=False)

    #     for i in range(len(output_texts)):
    #         output = cut_string_between_bos_eos(output_texts[i])
    #         reference = batch['translation']['en'][i]

    #         all_generated.append(output)
    #         all_references.append(reference)

    #         sent_bleu = calculate_bleu_score(output, reference)

    #         print('Optimal Sentence:', file=file)
    #         print(reference, file=file)
    #         print('Generated Sentence:', file=file)
    #         print(output, file=file)
    #         print('BLEU score:', round(sent_bleu * 100, 2), file=file)
    #         print('', file=file)
    #         bleu_scores.append(sent_bleu)

    # # Corpus-level BLEU - the proper metric (sacrebleu)
    # corpus_bleu = calculate_corpus_bleu(all_generated, all_references)
    # print(f'Corpus BLEU Score (sacrebleu): {corpus_bleu:.2f}')
    # print(f'Average Sentence BLEU: {sum(bleu_scores) / len(bleu_scores) * 100:.2f}')
    # print(f'Max Sentence BLEU: {max(bleu_scores) * 100:.2f}')
    # print(f'Min Sentence BLEU: {min(bleu_scores) * 100:.2f}')
    # file.flush()
    # file.close()
