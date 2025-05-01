# Transformer_from_scratch
Complete implementation of the transformer model in the paper 'Attention is all you need (Vaswani et al., 2017)' using pytorch

## Dataset
Wmt19 en-de (used the part of it; 3M rows)

## Method
- Used only the basic operations (nn.Linear, nn.ReLU, nn.Dropout, etc.)
- Mostly same with the original paper
- Used AutoAMP for faster training
- Stepwise learning: train with short sentences first, and then large sentences (criterion: longer than 16 words)
    - Around 80% of the data are short sentences, but trained on short and long sentences 50:50
    - More training time spent on long sentences
    - This helped fast learning, since we can learn more tokens with large sentences than short ones
    
## Score
- BLEU score on the test set
- Without any techniques, same as the paper: 24.85
- With techniques above, same time of training: 29.47 (better than the original paper!)
- The performance gap becomes larger when it comes to translating large sentences (look at the samples in the otuput_text_pairs.txt)

## Analysis
- Techniques like AutoAMP
- Better tokenizer (T5)
- Better data (wmt14 in original vs wmt19 for this model)