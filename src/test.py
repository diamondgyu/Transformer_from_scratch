import torch
from transformers import AutoTokenizer
from model import TransformerModel
from tqdm import tqdm
from downloader import download_data
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time
import pathlib
device_name = 'cuda:1'

bos_token = {'bos_token': '<s>'}
tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.add_special_tokens(bos_token)

'''
input_text is a string or a tensor
if it is a tensor, its shape is (batch_size, seq_len)
'''
def generate(model, input_text, tokenizer, max_len):
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    if type(input_text) == str:
        input_text = tokenizer(input_text, padding='max_length', truncation=True, max_length=max_len)['input_ids']
        input_text = torch.as_tensor(input_text, dtype=torch.long, device=model.device)
        input_text = input_text.unsqueeze(0)

    # Initialize sentence with BOS token
    sentences = torch.zeros((input_text.shape[0], max_len), dtype=torch.long, device=device).contiguous()
    sentences[:, 0] = tokenizer.bos_token_id
    
    with torch.no_grad():
        
        for index_working in range(0, max_len-1):
            # Get model outputs (logits)
            logits = model(input_text, sentences)
            
            # Get probabilities for the current position
            probs = torch.nn.functional.softmax(logits[:, index_working], dim=-1)
            top_probs, top_indices = torch.topk(probs, 5, dim=-1)

            # Select the top token
            next_token = top_indices[:, 0]

            # if every row of sentences contains the EOS token, break
            if torch.all( torch.any(sentences == tokenizer.eos_token_id, dim=1) ):
                break
            
            # Update sentence tensor
            sentences[:, index_working+1] = next_token
    
    return sentences


def calculate_bleu_score(generated_sentence, label):
    # Tokenize the generated sentence and label
    generated_tokens = word_tokenize(generated_sentence)
    label_tokens = word_tokenize(label)
    
    # Calculate BLEU score
    bleu_score = sentence_bleu([label_tokens], generated_tokens, smoothing_function=SmoothingFunction().method4)
    
    return bleu_score
    
def cut_string_between_bos_eos(string, bos='<s>', eos='</s>'):
    bos_index = string.find(bos)
    eos_index = string.find(eos)
    return string[bos_index+4:eos_index]


# Example Usage
if __name__ == "__main__":
    # This is EN->DE translation.
    # 177000 is the current SOTA
    local_model_path = "models/1024_2048_50k*64/model-checkpoint-epoch5-iter249000-long.mdl"  # Path to your saved model directory
    # result_optimal1 = "In the years before covid-19 struck, rents were high but not growing fast: the cost of leasing a home rose by about 2% a year, according to official data."
    # input_text1     = 'In den Jahren vor dem Ausbruch von Covid-19 waren die Mieten zwar hoch, stiegen jedoch nicht schnell: Offiziellen Angaben zufolge stiegen die Kosten für die Miete einer Wohnung jährlich um etwa 2 %.'

    # result_optimal2 = "In the latest dramatic showdown between the White House and the judiciary, a federal judge temporarily blocked the administration’s ability to use the centuries-old Alien Enemies Act on Saturday evening, and verbally ordered any planes in the air carrying some of those migrants to turn back to the US."
    # input_text2     = 'Im jüngsten dramatischen Showdown zwischen dem Weißen Haus und der Justiz blockierte ein Bundesrichter am Samstagabend vorübergehend die Möglichkeit der Regierung, den jahrhundertealten Alien Enemies Act anzuwenden, und wies alle Flugzeuge mit Migranten an Bord mündlich an, in die USA zurückzukehren.'

    # optimal_results = [result_optimal1, result_optimal2]
    # Load model and tokenizer from local directory
    embed_dim = 1024
    num_heads = 16
    dropout_rate = 0.1
    hidden_layer_dim = 1024*3
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
        stacks=stacks
    )

    path = pathlib.Path(__file__).parent.parent.resolve()
    file = open(path/'output_text_pairs.txt', 'w')
    model.load_state_dict(torch.load(path/local_model_path, weights_only=True))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    bos_token = {'bos_token': '<s>'}
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    tokenizer.add_special_tokens(bos_token)
    model = model.to(device)

    bleu_scores = []

    while True:
        input_text1 = input('Enter german sentence: ')
        if input_text1 == 'quit':
            break

        test = tokenizer([input_text1], padding='max_length', truncation=True, return_tensors="pt", max_length=tokenizer_max_len).to(device=device)

        output_sentences = generate(model, test['input_ids'], tokenizer, max_len=tokenizer_max_len)

        output_sentences = tokenizer.batch_decode(output_sentences, skip_special_tokens=False)
        print('Translation: ', cut_string_between_bos_eos(output_sentences[0]))

        bleu_scores.append(calculate_bleu_score(cut_string_between_bos_eos(output_sentences[0]), input_text1))
        print()
        # for i in range(len(output_sentences)):
        #     output = cut_string_between_bos_eos(output_sentences[i])
        #     print()
        #     print('Generated Translation: ', output)
        #     print()
        #     input_2 = input('Enter the correct translation: ')
        #     print('BLEU Score: ', round(calculate_bleu_score(input_2, output)*100, 2))
        #     print()
        
        
    '''  ------------------------------------------------------------------------------------------------  '''

    # Will not use train set
    _, valid, test = download_data(batch_size=32, tokenizer_max_len=128, len_train=1, short_sentences=False, test_ratio=0.99)

    for batch in tqdm(test):
        input_sentences = batch['input_ids'].to(device, non_blocking=True).contiguous()
        output_sentences = generate(model, input_sentences, tokenizer, max_len=tokenizer_max_len)

        input_sentences = tokenizer.batch_decode(input_sentences, skip_special_tokens=False)
        output_sentences = tokenizer.batch_decode(output_sentences, skip_special_tokens=False)

        for i in range(len(input_sentences)):
            input = cut_string_between_bos_eos(input_sentences[i])
            output = cut_string_between_bos_eos(output_sentences[i])
            bleu_score = calculate_bleu_score(output, batch['translation']['en'][i])

            # print(input)
            print('Optimal Sentence:', file=file)
            print(batch['translation']['en'][i], file=file)
            print('Generated Sentence:', file=file)
            print(output, file=file)
            print('BLEU score: ', round(bleu_score*100, 2), file=file)
            print('', file=file)
            bleu_scores.append(bleu_score)

    print('Average BLEU Score: ', round(sum(bleu_scores)/len(bleu_scores)*100, 2))
    print('Max BLEU Score: ', round(max(bleu_scores)*100, 2))
    print('Min BLEU Score: ', round(min(bleu_scores)*100, 2))
    file.flush()
    file.close()