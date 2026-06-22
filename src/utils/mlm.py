import numpy as np
import pickle
import re
import math

from tqdm import tqdm
import torch
from embedding import BertHuggingfaceMLM


class DatasetForTransformer(torch.utils.data.Dataset):
    """
    Torch dataset for transformer and MLM objective. Contains the encodings as returned from the tokenizer,
    adds sample indeces.
    """
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['index'] = idx
        return item

    def __len__(self):
        return len(self.encodings.input_ids)


def mask_texts(bert: BertHuggingfaceMLM, texts: list[str], max_length = 512, return_tokens = True, verbose = False):
    """
    Applies masking strategy to texts and converts back to text. Consider 15% of tokens for masking, of which 80% will be masked,
    10% replaced with a random token and 10% left as they are.

    Args:
        bert (BertHuggingfaceMLM): Instance of a custom wrapper for huggingface transformer models, which includes the tokenizer needed here.
        texts (list): Input texts.
        max_length (int): Maximum length of texts for tokenizer.
        return_tokens (bool): Indicates if tokenized texts should be returned in tokenized form or not (transformed back to str).
            
    Returns:
        dict | list[str] : Either encodings from tokenizer including masked input ids, attention mask and labels or de-tokenized texts with masks (specified via return_tokens).
    """
    inputs = bert.tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True,
                            padding='max_length')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # create special token mask (these are not considered for masking)
    special_tokens_mask = inputs.get('special_tokens_mask', None)
    if special_tokens_mask is None:
        if verbose:
            print("calculate special tokens manually (not provided by tokenizer, might be slower)")
        special_tokens_mask = torch.tensor([
            bert.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True) 
            for ids in input_ids.cpu().tolist()
        ], device='cpu')
    
    # masking strategy:
    # take 15% of tokens, of these replace 80% by mask, replace 10% with random token, leave 10% unchanged
    rand_probs = torch.rand_like(input_ids.float())
    candidate_mask = (~special_tokens_mask.bool()) & (attention_mask.bool())
    to_mask = candidate_mask & (rand_probs < 0.15)
    
    labels = input_ids.clone()
    labels[~to_mask] = -100
    
    split_rand = torch.rand_like(input_ids.float())
    is_mask = to_mask & (split_rand < 0.80)
    input_ids[is_mask] = bert.tokenizer.mask_token_id
    
    is_random = to_mask & (split_rand >= 0.80) & (split_rand < 0.90)
    if is_random.any():
        rows, cols = torch.where(is_random)
        random_tokens = torch.randint(0, len(bert.tokenizer), (len(rows),), device='cpu')
        input_ids[rows, cols] = random_tokens

    inputs['labels'] = labels

    if return_tokens:
        return inputs
    

    # We need to decode row by row because padding might affect batch decoding if lengths vary
    # Although we padded to max_length, batch_decode handles it well if skip_special_tokens=False
    # But we want to see the [MASK] token, so we keep special tokens visible.
    masked_text_list = []
    decoded_strings = bert.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=False)
    
    for text in decoded_strings:
        # Clean up extra spaces caused by padding or specific tokenizer behavior
        cleaned_text = text.replace(bert.tokenizer.pad_token, '').strip()
        masked_text_list.append(cleaned_text)
        
    return masked_text_list
    


def evaluate_mlm(bert: BertHuggingfaceMLM, texts: list[str], max_length = 512, verbose=False):
    """
    Evaluates MLM performance on arbitrary input texts for unmasking accuracy and perplexity.

    Args:
        bert (BertHuggingfaceMLM): Instance of a custom wrapper for huggingface transformer models that implements the actual forward pass with GPU support.
        texts (list): Input texts.
        pooling (str): Pooling strategy for embeddings. Options are 'mean' (mean pooled embedding over entire sentence) or 'mask' (embedding at mask token position)
            
    Returns:
        dict: dictionary with results for keys accuracy, perplexity, total_masked_tokens
    """
    inputs = mask_texts(bert, texts, max_length, return_tokens=True)
    
    # add MLM labels, set up dataloader
    dataset = DatasetForTransformer(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bert.batch_size, shuffle=False)

    # batched forward pass and eval
    n_samples = inputs['input_ids'].size(0)
    total_correct = 0
    total_masked = 0
    total_loss_sum = 0.0
    for batch in tqdm(loader, leave=True):
        if torch.cuda.is_available():
            for key in batch.keys():
                if key != 'index':
                    batch[key] = batch[key].to('cuda')

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        indices = batch['index']

        # forward pass and get the token predictions at positions of interest (the 15% of tokens)
        out = bert.model(input_ids, attention_mask=attention_mask)
        logits = out.logits # shape: batch_size, tokens, vocab size
        predictions = torch.argmax(logits, dim=-1)
        valid = (labels != -100)
        
        if valid.any():
            total_masked += valid.sum().item()
            total_correct += (predictions == labels)[valid].sum().item()
            
            if out.loss is not None:
                total_loss_sum += outputs.loss.item() * valid.sum().item()
            else:
                # fallback if loss is None
                if verbose:
                    print("loss was none, fallback loss computation")
                ce = torch.nn.CrossEntropyLoss(reduction='none')
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                loss_per_token = ce(logits_flat, labels_flat)
                valid_flat = valid.view(-1)
                total_loss_sum += loss_per_token[valid_flat].sum().item()

    # compute accuracy and perplexity
    if total_masked == 0:
        accuracy = 0.0
        perplexity = float('inf')
    else:
        accuracy = total_correct / total_masked
        mean_loss = total_loss_sum / total_masked
        perplexity = math.exp(mean_loss)

    return {
        "accuracy": float(accuracy),
        "perplexity": float(perplexity),
        "total_masked_tokens": int(total_masked)
    }


def forward_mlm_for_bias_eval(bert, texts: list[str], replace_terms: list[str], pooling='mask', max_length = 512):
    """
    Handles the forward pass of bias evaluation data to obtain embeddings and probabilities for terms to replace a mask token.
    
    Requires a batch of input texts with one [MASK] each where a set of terms can be meaningfully inserted (e.g. all texts have a [MASK] 
    where pronouns can be inserted). After the forward pass, mean-pooled or mask embeddings and the probabilities of specified  terms to 
    replace the [MASK] are extracted and returned. Supports batch processing on GPU.
    
    Args:
        bert (BertHuggingfaceMLM): Instance of a custom wrapper for huggingface transformer models that implements the actual forward pass with GPU support.
        texts (list): Input texts, including one [MASK] each (at a position where one of the replace_terms can be meaningfully inserted).
        replace_terms (list): List of terms whose probability to replace the [MASK] is tested - single token words are expected.
        pooling (str): Pooling strategy for embeddings. Options are 'mean' (mean pooled embedding over entire sentence) or 'mask' (embedding at mask token position)
            
    Returns:
        np.ndarray: Embeddings according to specified pooling strategy with shape (n_samples, emb_dim)
        np.ndarray: Model's probabilities of each replace term per input text with shape (n_samples, n_groups)
    """
    emb_dim = bert.model.config.hidden_size
    
    vocab_ids = [bert.tokenizer.get_vocab().get(word) for word in replace_terms]
    inputs = bert.tokenizer(texts, return_tensors='pt', max_length=max_length, truncation=True,
                            padding='max_length')
    dataset = DatasetForTransformer(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=bert.batch_size, shuffle=False)

    output_emb = np.zeros((len(texts), emb_dim))
    output_prob = np.zeros((len(texts), len(vocab_ids)))
    for batch in tqdm(loader, leave=True):
        if torch.cuda.is_available():
            for key in batch.keys():
                if key != 'index':
                    batch[key] = batch[key].to('cuda')

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        indices = batch['index']

        out = bert.model(input_ids, attention_mask=attention_mask)
        logits = out.logits # shape: batch_size, tokens, vocab size
        token_emb = out.hidden_states[-1] # shape: batch size, tokens, emb dim

        # mask with [mask] token positions and assert exactly one mask token per sample
        mask = (input_ids == bert.tokenizer.mask_token_id)
        row_counts = mask.sum(dim=1)
        
        if not torch.all(row_counts == 1):
            for index in indices:
                print(texts[index])
        assert torch.all(row_counts == 1)
        
        # get mask token indices
        masked_indices = torch.nonzero(mask, as_tuple=False)
        batch_ids = masked_indices[:,0]
        token_ids = masked_indices[:,1]

        # get mask token probabilities for selected targets
        masked_logits = logits[batch_ids, token_ids, :]
        probs = masked_logits.softmax(dim=-1)
        target_probs = probs[:, vocab_ids]
        
        if pooling == 'mask':
            # get mask token embeddings
            pooled_emb = token_emb[batch_ids, token_ids, :]
        else:
            # get mean pooled embedding
            attention_repeat = torch.repeat_interleave(attention_mask, token_emb.size()[2]).reshape(token_emb.size())
            pooled_emb = torch.sum(token_emb * attention_repeat, dim=1) / torch.sum(attention_repeat, dim=1)

            attention_repeat = attention_repeat.to('cpu')
            del attention_repeat

        output_emb[indices.numpy()] = pooled_emb.to('cpu').detach().numpy()
        output_prob[indices.numpy()] = target_probs.to('cpu').detach().numpy()

        input_ids = input_ids.to('cpu')
        attention_mask = attention_mask.to('cpu')
        logits = logits.to('cpu')
        token_emb = token_emb.to('cpu')

        del input_ids
        del attention_mask
        del logits
        del token_emb
        torch.cuda.empty_cache()

    return output_emb, output_prob


