import torch
from transformers import AutoTokenizer
from model import TransformerModel
from tqdm import tqdm
from process_korean_data import download_data
import sacrebleu
import pathlib

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# bos_token = {'bos_token': '<s>'}
# tokenizer.add_special_tokens(bos_token)


def generate(model, input_text, tokenizer, max_len):
    """
    Generate translations using greedy decoding with KV cache.
    Encodes the source ONCE, then decodes token-by-token.

    input_text: str or (batch_size, seq_len) tensor
    Returns: (batch_size, max_len) tensor of token ids
    """
    device = next(model.parameters()).device
    model.eval()

    if isinstance(input_text, str):
        encoded = tokenizer(input_text, padding='max_length',
                            truncation=True, max_length=max_len)
        input_text = torch.as_tensor(
            encoded['input_ids'], dtype=torch.long, device=device,
        ).unsqueeze(0)

    return model.beam_generate(
        src=input_text,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_len=max_len,
    )


def calculate_bleu_score(generated_sentence, label):
    """Sentence-level BLEU via sacrebleu (returns 0-1 scale)."""
    score = sacrebleu.sentence_bleu(generated_sentence, [label])
    return score.score / 100.0


def calculate_corpus_bleu(generated_sentences, references):
    """Corpus-level BLEU via sacrebleu (returns 0-100 scale)."""
    score = sacrebleu.corpus_bleu(generated_sentences, [references])
    return score.score


def cut_string_between_bos_eos(string, bos='<s>', eos='</s>'):
    bos_index = string.find(bos)
    eos_index = string.find(eos)
    if bos_index == -1:
        bos_index = -len(bos)
    if eos_index == -1:
        eos_index = len(string)
    return string[bos_index + len(bos):eos_index].strip()


# Example Usage
if __name__ == "__main__":
    # This is EN->DE translation.
    local_model_path = "models/1024_2048_50k*64/model-checkpoint-epoch5-iter249000-long.mdl"

    embed_dim = 1024
    num_heads = 16
    dropout_rate = 0.1
    hidden_layer_dim = 1024 * 3
    tokenizer_max_len = 128
    vocab_size = 32129
    stacks = 6

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

    path = pathlib.Path(__file__).parent.parent.resolve()
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    # Load checkpoint (strict=False to handle new register_buffer keys in old checkpoints)
    model.load_state_dict(
        torch.load(path / local_model_path, weights_only=True),
        strict=False,
    )
    model = model.to(device)

    file = open(path / 'output_text_pairs.txt', 'w')
    bleu_scores = []

    # Interactive mode
    while True:
        input_text1 = input('Enter german sentence: ')
        if input_text1 == 'quit':
            break

        test_input = tokenizer(
            [input_text1], padding='max_length', truncation=True,
            return_tensors="pt", max_length=tokenizer_max_len,
        ).to(device=device)

        output_sentences = generate(model, test_input['input_ids'], tokenizer,
                                    max_len=tokenizer_max_len)
        output_sentences = tokenizer.batch_decode(output_sentences, skip_special_tokens=False)
        translation = cut_string_between_bos_eos(output_sentences[0])

        print('Translation:', translation)
        bleu_scores.append(calculate_bleu_score(translation, input_text1))
        print()

    '''  ------------------------------------------------------------------------------------------------  '''

    # Batch evaluation with corpus-level BLEU (sacrebleu)
    _, valid, test_data = download_data(
        batch_size=32, tokenizer_max_len=128, len_train=1,
        test_ratio=0.99,
    )

    all_generated = []
    all_references = []

    for batch in tqdm(test_data):
        input_sentences = batch['input_ids'].to(device, non_blocking=True).contiguous()
        output_sentences = generate(model, input_sentences, tokenizer, max_len=tokenizer_max_len)

        output_texts = tokenizer.batch_decode(output_sentences, skip_special_tokens=False)

        for i in range(len(output_texts)):
            output = cut_string_between_bos_eos(output_texts[i])
            reference = batch['translation']['en'][i]

            all_generated.append(output)
            all_references.append(reference)

            sent_bleu = calculate_bleu_score(output, reference)

            print('Optimal Sentence:', file=file)
            print(reference, file=file)
            print('Generated Sentence:', file=file)
            print(output, file=file)
            print('BLEU score:', round(sent_bleu * 100, 2), file=file)
            print('', file=file)
            bleu_scores.append(sent_bleu)

    # Corpus-level BLEU - the proper metric (sacrebleu)
    corpus_bleu = calculate_corpus_bleu(all_generated, all_references)
    print(f'Corpus BLEU Score (sacrebleu): {corpus_bleu:.2f}')
    print(f'Average Sentence BLEU: {sum(bleu_scores) / len(bleu_scores) * 100:.2f}')
    print(f'Max Sentence BLEU: {max(bleu_scores) * 100:.2f}')
    print(f'Min Sentence BLEU: {min(bleu_scores) * 100:.2f}')
    file.flush()
    file.close()
