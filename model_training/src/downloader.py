
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import AutoTokenizer

path = Path(__file__).parent.parent
language = 'de'

bos_token = {'bos_token': '<s>'}
tokenizer = AutoTokenizer.from_pretrained("t5-base")
tokenizer.add_special_tokens(bos_token)

# Returns train/validation/test dataset
def download_data(batch_size, tokenizer_max_len, len_train, short_sentences, test_ratio=0.5):

    try: # Dataset already exists
        if short_sentences:
            train = load_from_disk(path/'data/wmt19-{}-en-train-short-{}'.format(language, tokenizer_max_len))
            test = load_from_disk(path/'data/wmt19-{}-en-test-short-{}'.format(language, tokenizer_max_len))
        else:
            train = load_from_disk(path/'data/wmt19-{}-en-train-long-{}'.format(language, tokenizer_max_len))
            test = load_from_disk(path/'data/wmt19-{}-en-test-long-{}'.format(language, tokenizer_max_len))

    except:
        print('No existing dataset. Downloading Data from the server...')
        
        # Download and split dataset
        train_dataset, test_dataset = load_dataset("wmt19", "{}-en".format(language), split=['train', 'validation'])

        if short_sentences:
            # Short, but not too short...
            short_sentences_train = train_dataset.filter(lambda example: len(example["translation"]["en"].split()) <= 12 and len(example["translation"]["en"].split()) > 8)
            short_sentences_train = short_sentences_train.shuffle(seed=42)
            short_sentences_test = test_dataset.filter(lambda example: len(example["translation"]["en"].split()) <= 12 and len(example["translation"]["en"].split()) > 8)
            train_dataset = short_sentences_train
            test_dataset = short_sentences_test
        else:
            long_sentences_train = train_dataset.filter(lambda example: len(example["translation"]["en"].split()) > 12)
            long_sentences_train = long_sentences_train.shuffle(seed=42)
            long_sentences_test = test_dataset.filter(lambda example: len(example["translation"]["en"].split()) > 12)
            train_dataset = long_sentences_train
            test_dataset = long_sentences_test

        # total_set = train_dataset.shuffle(seed=42)
        # train_dataset = total_set.select(range(len_dataset)) # First 1M samples
        # validation_dataset = total_set.select(range(len_dataset, len_dataset+1000)) # First 20K samples
        # test_dataset = total_set.select(range(len_dataset+1000, len_dataset+2000))

        print('----Dataset Size----')
        print(f'Train: {len(train_dataset)}, / Valid+Test: {len(test_dataset)}')

        # Preprocessing
        def preprocess_function(examples):
            # Get the source and target sentences
            source = [example[language] for example in examples['translation']]
            target = [example['en'] for example in examples['translation']]
            
            # Encode them
            source_encodings = tokenizer(source, padding='max_length', truncation=True, max_length=tokenizer_max_len)
            target_encodings = tokenizer(target, padding='max_length', truncation=True, max_length=tokenizer_max_len)
            
            # Create the final dictionary and return
            encodings = {
                'input_ids': source_encodings['input_ids'],
                'labels': target_encodings['input_ids']
            }
            return encodings
        
        print('Preprocessing the dataset...')
        print(test_dataset[0])
        
        train = train_dataset.map(preprocess_function, batched=True)
        test = test_dataset.map(preprocess_function, batched=True)
        
        print('\n\nDone!')

        # The data should be downloaded into the 'data' folder

        if short_sentences:
            train.save_to_disk(path/'data/wmt19-{}-en-train-short-{}'.format(language, tokenizer_max_len))
            test.save_to_disk(path/'data/wmt19-{}-en-test-short-{}'.format(language, tokenizer_max_len))
        else:
            train.save_to_disk(path/'data/wmt19-{}-en-train-long-{}'.format(language, tokenizer_max_len))
            test.save_to_disk(path/'data/wmt19-{}-en-test-long-{}'.format(language, tokenizer_max_len))

    # Change the format to torch and create dataloader objects

    test_split = test.train_test_split(test_size=test_ratio, seed=42)

    train = train.select(range(len_train))
    train = train.with_format(type='torch')
    valid = test_split['train'].with_format(type='torch')
    test = test_split['test'].with_format(type='torch')

    # Sort valid and test by first one position, for faster generation
    def get_first_eos_position(example):
        # Assuming your data is in a column called 'input_ids' or similar
        # Adjust the column name as needed
        tensor = example['labels']  # or whatever column contains your data
        ones_positions = [i for i, val in enumerate(tensor) if val == 1]
        # Return inf if no 1s found, otherwise return the first position
        return float('inf') if not ones_positions else ones_positions[0]
    
    valid = valid.map( lambda example: {'first_one_position': get_first_eos_position(example)})
    test = test.map( lambda example: {'first_one_position': get_first_eos_position(example)})

    valid = valid.sort('first_one_position')
    valid = valid.remove_columns(['first_one_position'])
    test = test.sort('first_one_position')
    test = test.remove_columns(['first_one_position'])

    train = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8)
    valid = DataLoader(valid, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    test = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)

    return train, valid, test