import pandas as pd
import json
from pathlib import Path
from datasets import Dataset as HFDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch

path = Path(__file__).parent.parent
language = 'ko' # Input is Korean

tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token


def convert_xlsx_to_json():
    data_dir = path / 'data'
    xlsx_files = list(data_dir.glob('*.xlsx'))
    
    all_data = []
    
    for file in xlsx_files:
        print(f"Processing {file.name}...")
        df = pd.read_excel(file)
        if '원문' in df.columns and '번역문' in df.columns:
            for _, row in df.iterrows():
                all_data.append({
                    'translation': {
                        'ko': str(row['원문']),
                        'en': str(row['번역문'])
                    }
                })
        else:
            print(f"Warning: Columns '원문' or '번역문' not found in {file.name}. Found columns: {df.columns.tolist()}")

    output_file = data_dir / 'korean_english_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(all_data)} pairs to {output_file}")
    return output_file

def process_korean_data(batch_size, tokenizer_max_len, len_train, test_ratio=0.5):
    data_dir = path / 'data'
    json_path = data_dir / 'korean_english_data.json'
    ratio_key = str(test_ratio).replace('.', 'p')
    
    # If json doesn't exist, create it from xlsx
    if not json_path.exists():
        convert_xlsx_to_json()

    # Unified save paths (use all sentences)
    train_save_path = data_dir / f'ko-en-train-all-{tokenizer_max_len}'
    test_save_path = data_dir / f'ko-en-test-all-{tokenizer_max_len}'
    valid_sorted_save_path = data_dir / f'ko-en-valid-sorted-all-{tokenizer_max_len}-r{ratio_key}'
    test_sorted_save_path = data_dir / f'ko-en-test-sorted-all-{tokenizer_max_len}-r{ratio_key}'

    try:
        train = HFDataset.load_from_disk(train_save_path)
        test = HFDataset.load_from_disk(test_save_path)
        print("Loaded existing dataset from disk.")
    except:
        print("Dataset not found on disk. Creating from JSON...")
        # Load from JSON
        # datasets.load_dataset("json", data_files=str(json_path)) is an alternative
        # but here we load and filter to match downloader.py logic
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        
        full_dataset = HFDataset.from_list(data)

        dataset = full_dataset.shuffle(seed=42)
        
        # Split into train and the rest (which will be further split into valid and test)
        # Note: downloader.py uses WMT19's natural train/validation split.
        # Here we split the loaded dataset.
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42) # reserve 10% for valid+test
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

        def preprocess_function(examples):
            source = [example[language] for example in examples['translation']]
            target = [example['en'] for example in examples['translation']]
            target_with_eos = [t + tokenizer.eos_token for t in target]
            
            source_encodings = tokenizer(source, padding='max_length', truncation=True, max_length=tokenizer_max_len)
            target_encodings = tokenizer(target_with_eos, padding='max_length', truncation=True, max_length=tokenizer_max_len, add_special_tokens=False)
            
            return {
                'input_ids': source_encodings['input_ids'],
                'labels': target_encodings['input_ids']
            }

        train = train_dataset.map(preprocess_function, batched=True, load_from_cache_file=False)
        test = test_dataset.map(preprocess_function, batched=True, load_from_cache_file=False)

        train.save_to_disk(train_save_path)
        test.save_to_disk(test_save_path)
        print("Dataset saved to disk.")

    # Use len_train as requested
    if len_train != -1:
        train = train.select(range(min(len_train, len(train))))

    try:
        valid = HFDataset.load_from_disk(valid_sorted_save_path)
        test = HFDataset.load_from_disk(test_sorted_save_path)
        print("Loaded sorted valid/test datasets from disk.")
    except:
        print("Sorted valid/test datasets not found. Creating and saving...")
        test_split = test.train_test_split(test_size=test_ratio, seed=42)
        valid = test_split['train']
        test = test_split['test']

        def get_first_eos_position(example):
            tensor = example['labels']
            ones_positions = [i for i, val in enumerate(tensor) if val == tokenizer.eos_token_id]
            return float('inf') if not ones_positions else ones_positions[0]
        
        valid = valid.map(lambda example: {'first_one_position': get_first_eos_position(example)})
        test = test.map(lambda example: {'first_one_position': get_first_eos_position(example)})

        valid = valid.sort('first_one_position').remove_columns(['first_one_position'])
        test = test.sort('first_one_position').remove_columns(['first_one_position'])

        valid.save_to_disk(valid_sorted_save_path)
        test.save_to_disk(test_sorted_save_path)
        print("Sorted valid/test datasets saved to disk.")

    train = train.with_format(type='torch')
    valid = valid.with_format(type='torch')
    test = test.with_format(type='torch')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

    return train_loader, valid_loader, test_loader

# Same return format as downloader.py, without short/long split
def download_data(batch_size, tokenizer_max_len, len_train, test_ratio=0.5):
    return process_korean_data(batch_size, tokenizer_max_len, len_train, test_ratio)
