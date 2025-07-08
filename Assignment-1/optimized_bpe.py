import collections
import regex as re

pairs = collections.defaultdict(int)
pairs_to_words = collections.defaultdict(set)

def get_stats(vocab):
    """ Returns the pairs of characters or character sequences with their frequencies."""
    pairs = collections.defaultdict(int)
    
    for t_word, freq in vocab.items():
        #iterate through eeach tuple of word
        for pair in zip(t_word, t_word[1:]):
            pairs[pair] += freq
            pairs_to_words[pair].add(t_word)
    
    return pairs


def merge_vocab(pair, vocab_in):
    """Merge the most frequent pair in the vocabulary."""
    pair_merged = pair[0] + pair[1]
    affected_words = list(pairs_to_words[pair])  

    for word in affected_words:
        freq = vocab_in[word]

        # Remove old pair stats for the original word
        word_pairs = list(zip(word, word[1:]))
        for p in word_pairs:
            pairs[p] -= freq
            if pairs[p] <= 0:
                del pairs[p]
            pairs_to_words[p].discard(word)
            if not pairs_to_words[p]:
                del pairs_to_words[p]

        # Merge the pair in the word
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(pair_merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)

        # Add new pair stats for the updated word
        new_pairs = list(zip(new_word, new_word[1:]))
        for p in new_pairs:
            pairs[p] += freq
            pairs_to_words[p].add(new_word)

        vocab_in[new_word] += freq
        del vocab_in[word]

    
    if pair in pairs:
        del pairs[pair]
    if pair in pairs_to_words:
        del pairs_to_words[pair]

    return vocab_in


if __name__ == "__main__":
    
    #config
    vocab_size = 500
    special_tokens = ['<|endoftext|>'] 
    vocab_dict = {}
    with open("../tests/fixtures/corpus.en", "r", encoding="utf-8") as f:
        text = f.read()
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    tokens = re.findall(PAT, text)
    vocab = collections.defaultdict(int)
    
    for token in tokens:
        token_bytes = token.encode("utf-8")
        vocab[tuple([bytes([b]) for b in token_bytes])] += 1 
    
    vocab_dict[bytes(special_tokens[0], 'utf-8')] = 0
    b = 0
    while len(vocab_dict) < 257:
        byte = bytes([b])
        if byte not in vocab_dict:
            vocab_dict[byte] = len(vocab_dict)
        b += 1
    merges = [] 
    
    pairs = get_stats(vocab)
    while len(vocab_dict) < vocab_size:
        if not pairs:
            break
        
        best = max(pairs.items(), key=lambda x: (x[1], x[0]))[0]
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        new_token = best[0] + best[1]
        if new_token not in vocab_dict:
            vocab_dict[new_token] = len(vocab_dict)
            
    vocab_dict = {v:k for k, v in sorted(vocab_dict.items(), key=lambda item: item[1])}
    print(vocab_dict)
    print(merges)
    