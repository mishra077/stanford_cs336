import collections
import regex as re

def  get_stats(vocab):
	""" Returns the pairs of characters or character sequences with their frequencies."""
	# pairs will be a dictionary where keys are tuples of pairs and values are their frequencies
	pairs = collections.defaultdict(int)
	for t_word, freq in vocab.items():
		#iterate through eeach tuple of word
		for i in  range(len(t_word) - 1):
			pairs[(t_word[i], t_word[i + 1])] += freq
	return pairs

def  merge_vocab(pair, vocab_in):
	"""Merge the most frequent pair in the vocabulary."""
	vocab_out = {}
	pair_merged = pair[0] + pair[1]
	for word in vocab_in:
		new_word = []
		i = 0
		while i < len(word):
			if i < len(word) - 1  and word[i] == pair[0] and word[i + 1] == pair[1]:
				new_word.append(pair_merged)
				i += 2
			else:
				new_word.append(word[i])
				i += 1
		vocab_out[tuple(new_word)] = vocab_in[word]
	return vocab_out

if  __name__ == "__main__":
	#config
	vocab_size = 500
	special_tokens = ['<|endoftext|>']
	vocab_dict = {}
	with  open("../tests/fixtures/corpus.en", "r", encoding="utf-8") as f:
		text = f.read()
		
	PAT = r"""'(?:[sdmt]|ll|ve|re)|  ?\p{L}+|  ?\p{N}+|  ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
	tokens = re.findall(PAT, text)
	vocab = collections.defaultdict(int)
	
	for token in tokens:
		token_bytes = token.encode("utf-8")
		vocab[tuple([bytes([b]) for b in token_bytes])] += 1
		vocab_dict[bytes(special_tokens[0], 'utf-8')] = 0
	
	b = 0
	while  len(vocab_dict) < 257:
		byte = bytes([b])
		if byte not  in vocab_dict:
			vocab_dict[byte] = len(vocab_dict)
		b += 1
		
	merges = []
	while  len(vocab_dict) < vocab_size:
		pairs = get_stats(vocab)
		if  not pairs:
			break
		
		best = max(pairs.items(), key=lambda  x: (x[1], x[0]))[0]
		vocab = merge_vocab(best, vocab)
		merges.append(best)

		new_token = best[0] + best[1]
		if new_token not  in vocab_dict:
			vocab_dict[new_token] = len(vocab_dict)
	
	vocab_dict = {v:k for k, v in  sorted(vocab_dict.items(), key=lambda  item: item[1])}
	print(vocab_dict)
	print(merges)