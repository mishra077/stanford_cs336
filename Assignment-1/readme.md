
# Assignment 1 Answers for CS336

## Byte Pair Encoding (BPE)

> Byte Pair Encoding (BPE) is a simple data compression technique that iteratively replaces the most frequent pair of bytes in a sequence with a single, unused byte.

```python
>>> chr(0)
>>> \x00
>>> print(chr(0))
>>>
```
From 0 - 31 are control characters, which means they are not printable. The printable characters start from 32 (space) to 126 (tilde ~).

```python
chr(0) = '\x00'  # Null byte
chr(1) = '\x01'  # Start of Heading
chr(2) = '\x02'  # Start of Text
chr(3) = '\x03'  # End of Text
chr(4) = '\x04'  # End of Transmission
...
```
### Q. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings?		

1. UTF-8 is much more compact, especially for common characters like English letters.
2. While UTF-32 is simple and uniform, it’s highly inefficient in practice, bloating the size of training data.
3. UTF-32 in particular pads everything to 4 bytes, so characters like 'a' (U+0061) become 00 00 00 61. That’s 75% waste for ASCII.
4. UTF-16 and UTF-32 inflate the raw input size, increasing sequence lengths, which leads to:
	- More memory usage during training.
	- Higher compute cost (self-attention is O(n²)).
	- Less context per forward pass (fixed-length limits like 2048 or 4096 tokens hit faster).


### Q. Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results

```python
def  decode_utf8_bytes_to_str_wrong(bytestring: bytes):
	return  "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
**Why is this incorrect?**

UTF-8 is a variable-length encodiong scheme, meaning that some characters are represented by mutilple bytes. When we split in bytes individullaly, the decoder doesn't know how to reconstruct multi-byte characters correctly.

Example where it fails:
```python
>>> test_string = "hello! こんにちは!"
>>> decode_utf8_bytes_to_str_wrong(test_string.encode("utf-8"))
>>> UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
```

This error occurs because the function attempts to decode each byte independently, which fails for multi-byte characters like those in "こんにちは" (Japanese characters), which are encoded as multiple bytes in UTF-8.

### Q. Give a two byte sequence that does not decode to any Unicode character(s)

A two-byte sequence that does not decode to any Unicode character(s) is `b'\x80\x80'`. This `\x80` is a continuation byte in UTF-8, but you need a leading byte to form a valid character. See the table below:

| Byte Type | Binary Prefix | Hex Range | Decimal Range |
|-----------|---------------|-----------|----------------|
| Ascii(1-byte) | 0xxxxxxx | 00-7F | 0-127 |
| Start(2-byte) | 110xxxxx | C2-DF | 194-223 |
| Start(3-byte) | 1110xxxx | E0-EF | 224-239 |
| Start(4-byte) | 11110xxx | F0-F7 | 240-247 |
| Continuation | 10xxxxxx | 80-BF | 128-191 |

So every continuation byte starts with `10` to:
1. Tell the decoder: "Don't interpret me alone."
2. Let the decoder match me with the previous start byte.

### Naive Implementation of BPE

```python
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
```

#### Issue encountered
1. Incorrect merging of Byte pairs:
	- Merged tokens like `b'\x80'` were appearing instead of `b'\xc2\x80'`, leading to test failures.
	- When merging two byte tokens like `b'\xc2'` and `b'\x80'`, you must concatenate them as raw bytes, not re-encoded characters using `str->bytes` again. Using `bytes('...', "utf-8")` on already enconded bytes double-encodes and corrupt the vocab.
2. Tokens were added to vocab using `tuple(token)` instead of `tuple(token.encode("utf-8"))`, leading to character-based rather than byte-based BPE merges.

  

#### Profiling of Naive BPE

  

| Line | CPU % | Copy % | Memory % | Function | Details |
|------|--------|--------|-----------|-----------------------------------|-----------------------------------------------|
| 5 | 35% | | 3% | `get_stats` | Computes all pair frequencies from scratch |
| 17 | 47% | 2% | 5% | `merge_vocab` | Rebuilds vocabulary with merged pairs |
| 11 | 17% | | 1% | └─ loop over `zip(t_word, ...)` | Main loop for pair collection in `get_stats` |
| 12 | 22% | 1% | 2% | └─ `pairs[pair] += freq` | Pair frequency update |
| 25 | 17% | | 1% | └─ `while` loop in `merge_vocab` | Iterates over each word |
| 26 | 15% | | 2% | └─ Merge condition + append | Checks and merges the most frequent pair |
| 30 | 9% | 1% | 1% | └─ `append(word[i])` | Keeps unmerged symbols |
| 32 | 6% | | | └─ `vocab_out[tuple(...)]` | Writes new word to updated vocabulary|
##### Key Observations
-  `get_stats` is expensive because it recomputes **all** pair frequencies every iteration.
-  `merge_vocab` is costly due to full reconstruction of every word per merge.
- Major hotspots:
	- Pair counting via `zip(...)` in `get_stats`.
	- Merge logic inside `while` loop in `merge_vocab`.
	- Total `time taken = 2.12 secs` to get vocab size of 500.


### Optimization in Merging
In the naïve version, each merge step does the following:
1. Scan **every word** in the vocabulary.
2. Count **all adjacent symbol pairs** in each word.
3. Merge the most frequent pair in every word.
4. Rebuild the vocabulary from scratch.

#### Why it's inefficient:
- **Redundant work**: Most words remain unchanged after a merge, yet they are scanned every time.
- **O(V·L)** complexity per merge step, where V is the number of words and L is average word length.
- **No context tracking**: There's no awareness of *which* words contain *which* pairs.

To remove redundancy and accelerate BPE, the optimized approach introduces:
####  1. `pairs: defaultdict(int)`
- Stores frequency counts of all symbol pairs across the vocabulary.
- Updated **incrementally** after each merge.
- Eliminates the need to recompute stats from scratch each iteration.

#### 2. `pairs_to_words: defaultdict(set)`
- Maps each symbol pair to the set of words that contain it.
- Enables fast lookup of **affected words** for a given merge operation.
- Prevents full-vocabulary scans.


**For each merge step**:

1. **Identify most frequent pair**:
   - From `pairs`, which is already up-to-date.
2. **Lookup affected words**:
   - Use `pairs_to_words[pair]` to fetch only words needing updates.
3. **Remove outdated statistics**:
   - Subtract frequency of old pairs found in the word from `pairs`.
   - Remove `word` from the old pairs' entry in `pairs_to_words`.
4. **Merge the pair** in the word:
   - Replace occurrences of `pair` with `pair_merged`.
5. **Add new statistics**:
   - Recompute pairs in the new word.
   - Add frequency to `pairs` and register new word in `pairs_to_words`.
6. **Update vocabulary**:
   - Delete old word, add `new_word` with updated count.
7. **Clean up**:
   - Remove merged pair from both `pairs` and `pairs_to_words`.
This transforms the expensive O(V·L) merge loop into a focused update over a small subset of words.

#### BPE Profiling Summary (Optimized Version)

This summary highlights key bottlenecks and memory usage from profiling the optimized BPE implementation.

#### Function-Level Summary
| Function      | Time % | Native % | Sys % | Copy Rate (MB/s) |
|---------------|--------|----------|-------|------------------|
| `merge_vocab` | 27%    | 8%       | 6%    | 150              |
| `get_stats`   | 2%     | -        | 1%    | -                |

#### Hot Lines (Top Contributors)

| Line No. | Location                     | Operation / Description                                 | Time % | Native % | Sys % |
|----------|------------------------------|----------------------------------------------------------|--------|----------|-------|
| 29       | `merge_vocab`                | Construct `word_pairs` (zip of old word)                 | 6%     | -        | -     |
| 32       | `merge_vocab`                | Check pair count to delete from `pairs`                 | 6%     | 4%       | 3%    |
| 54       | `merge_vocab`                | Add new word to `pairs_to_words`                        | 5%     | 4%       | 3%    |
| 58       | `merge_vocab`                | Delete old word from vocab                              | 3%     | -        | -     |
| 100      | Token byte encoding loop     | Build initial vocab from UTF-8 bytes                    | 8%     | 9%       | 5%    |
| 118      | `max(pairs.items())`         | Select most frequent pair                               | 21%    | 4%       | 5%    |


####  Final Performance

| Metric              | Value            |
|---------------------|------------------|
| **Runtime (optimized)** | `~0.25 sec`        |
| Runtime (naive)     | ~2.2 sec         |
| **Speedup**             | `~9×`              |

### BPE Training on TinyStories
### Q. How many hours and memory did training take? What is the longest token in the vocabulary? Does it make sense?
1. **1 min 30 secs and 250 MB of memory.**
2. Longest Token: `Ġresponsibility`

| Index | Token | Length |
|-------|------------------|--------|
| 0     |      `EOT`       | 13     |
| 1 | `Ġunexpected` | 11 |
| 2 | `Ġremembered` | 11 |
| 3 | `Ġaccidentally` | 13 |
| 4 | `Ġeverywhere` | 11 |
| 5 | `Ġeverything` | 11 |
| 6 | `Ġunderstand` | 11 |
| 7 | `Ġadventures` | 11 |
| 8 | `Ġespecially` | 11 |
| 9 | `Ġinteresting` | 12 |
| 10 | `Ġvegetables` | 11 |
| 11 | `Ġembarrassed` | 12 |
| 12 | `Ġunderstood` | 11 |
| 13 | `Ġcaterpillar` | 12 |
| 14 | `Ġmysterious` | 11 |
| 15 | `Ġcomfortable` | 12 |
| 16 | `Ġfrustrated` | 12 |
| 17 | `Ġincredible` | 11 |
| 18 | `Ġfrightened` | 12 |
| 19 | `Ġadventurous` | 13 |
| 20 | `comfortable` | 11 |
| 21 | `Ġuncomfortable` | 14 |
| 22 | `Ġdisappeared` | 13 |
| 23 | `Ġsuccessful` | 11 |
| 24 | `Ġindependent` | 12 |

### Q. Profile your code. What part of the tokenizer training process takes the most time?
Roughly 67% of the total runtime was spent in the pre-tokenization chunk processor, making it the most time-consuming step.

Memory usage: ▁▁▁▁▁▁▂▂▂▂▂▄▇█▇▄█▇▄█▇████▄█ (max: 245.668 MB, growth rate: 100%)

| CPU % | Mem % | Time % | I/O % | CPU Load % | Mem Used | Memory Bar        | Line # | Function                                |
|-------|-------|--------|-------|------------|----------|--------------------|--------|-----------------------------------------|
| 9%    | 19%   | 67%    |       | 99%        | 175M     | █████████ 97%      | 82     | ForkProcess.pretokenize_chunk_worker    |
|       |       |        |       | 98%        | 10M      | ▂▂                 |        | get_stats                                |
|       |       |        |       |            |          |                    |        | merge_vocab                              |

### BPE Training on OpenWebText
Q. What is the longest token in the vocabulary? Does it make sense?
1. `ÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥ`

| Index | Token                                                  | Length |
|-------|---------------------------------------------------------|--------|
| 0     | `--------------------------------`                        | 32     |
| 1     | `ÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥ`                          | 30     |
| 2     | `âĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶ`                                  | 32     |
| 3     | `________________________________`                        | 32     |
| 4     | `ÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥÃĤÃĥ` | 60     |
| 5     | `----------------------------------------------------------------` | 64     |
| 6     | `================================`                        | 32     |
| 7     | `................................`                        | 32     |
| 8     | `âĢ¦âĢ¦âĢ¦âĢ¦âĢ¦âĢ¦âĢ¦âĢ¦`                                  | 32     |
| 9     | `âĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶâĢĶ`             | 56     |

**Does it make sense?**
Not really. These tokens appear to be a repetitive corrupted Unicode sequence, likely resulting from:
- Character encoding issues (e.g., misinterpreted UTF-8 bytes)
- Decorative elements or separators in the original webpages
- Noise commonly found in large-scale web scrapes like OpenWebText
While it makes sense from a frequency-driven BPE perspective, it doesn’t carry semantic meaning and likely reflects junk or formatting artifacts in the corpus.

### Q. Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText

1. Data Cleanliness & Semantic Quality
- TinyStories: Clean, curated text → tokens are mostly meaningful English words.
- OpenWebText: Web-scraped → contains noisy tokens like corrupted Unicode or repeated symbols.

2. Vocabulary Composition
- TinyStories: Tokens reflect natural language (e.g., 'Ġresponsibility').
- OpenWebText: Mix of real tokens and junk (e.g., 'ÃĤÃĥÃĤÃĥÃĤÃĥ...').

3. Use Case Fit
- TinyStories: Best for small, controlled language models.
- OpenWebText: General-purpose models, but may require token filtering.



### Q. Implementing the tokenizer: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there)

```python
class Tokenizer:
    PAT = re.compile(
        r"'s|'t|'re|'ve|'m|'ll|'d|" # Contractions (explicitly listed)
        r" ?\p{L}+|"                # Optional space then letters
        r" ?\p{N}+|"                # Optional space then numbers
        r" ?[^\s\p{L}\p{N}]+|"      # Optional space then other non-whitespace, non-letter, non-number
        r"\s+(?!\S)|\s+",           # Trailing whitespace (multiple, not followed by non-space) OR any other whitespace
        re.UNICODE 
    )
    def __init__(
            self,
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]],
            special_tokens: Optional[list[str]]=None):
        self.vocab = {v : k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.rev_vocab = vocab
    @classmethod
    def from_files(
            cls,
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: Optional[list[str]]=None):
        
        with open(vocab_filepath) as f:
            vocab = json.load(f)

        with open(merges_filepath) as f:
            merges = [tuple(line.rstrip().split(" ")) for line in f]
            
        for token in special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)

        return cls(vocab, merges, special_tokens)
    
    def encode_helper(self, text: str) -> list[int]:
        if not text:
            return []
        
        segments = self.PAT.findall(text)
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        print(segments)
        final_token_ids = []
        for segment in segments:
            segment_bytes = [bytes([b]) for b in segment.encode("utf-8")]
            
            def get_pairs(tokens: list[bytes]):
                return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            
            tokens = segment_bytes
            while True:
                pairs = get_pairs(tokens)
                curr_rank = float("inf")
                best_pair = None
                for p in pairs:
                    if p in merge_rank and merge_rank[p] < curr_rank:
                        curr_rank = merge_rank[p]
                        best_pair = p
                if best_pair is None:
                    break
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if (
                        i < len(tokens) - 1 and
                        tokens[i] == best_pair[0] and
                        tokens[i + 1] == best_pair[1]
                    ):
                        new_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

            final_token_ids.extend(self.vocab[tok] for tok in tokens)
        
        return final_token_ids

    
    def encode(self, text: str) -> list[int]:
        
        if not self.special_tokens:
            return self.encode_helper(text)
        
        left = 0
        right = 0
        encoded_list = []
        while right < len(text):
            matched = False
            for s_tok in sorted(self.special_tokens, key=len, reverse=True):
                # sorted due to the case for special_tokens = ["<|endoftext|><|endoftext|>". "<|endoftext|>"]
                # you want largest length to be matched first
                if text.startswith(s_tok, right):
                    if left < right:
                        encoded_list += self.encode_helper(text[left:right])
                    encoded_list.append(self.vocab[bytes(s_tok, encoding="utf-8")])
                    right += len(s_tok)
                    left = right
                    matched = True
                    break
            if not matched:
                right += 1
        if left < len(text):
            encoded_list += self.encode_helper(text[left:])
            
        
        return encoded_list
        
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            token_ids = self.encode(text)
            for tok_id in token_ids:
                yield tok_id
    
    def decode(self, ids: list[int]) -> str:
        return b"".join([self.rev_vocab[id] for id in ids]).decode("utf-8", errors="replace")
```

## Notes for implementing BPE tokenizer class
### Special Token Sorting
```python
for s_tok in sorted(self.special_tokens, key=len, reverse=True):
```
- **Why**: To prevent partial matches when one special token is a prefix of another.
- **Example**: If both `<|endoftext|>` and `<|endoftext|><|endoftext|>` are in special_tokens, matching the longer one first avoids prematurely consuming the shorter prefix.
- **Bug**: Without sorting, `<|endoftext|>` would match early, corrupting tokens.
- **Failure Example**:
	```python
	special_tokens = ["<|endoftext|><|endoftext|>", "<|endoftext|>"]
	text = "<|endoftext|><|endoftext|>foo"
	# Incorrect: matches the shorter token first
	```

### Regex-based Pre-tokenization `(PAT)`
```python
PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d|"
    r" ?\p{L}+|"
    r" ?\p{N}+|"
    r" ?[^\s\p{L}\p{N}]+|"
    r"\s+(?!\S)|\s+",
    re.UNICODE
)
```
- **Bug**: Merging occurred incorrectly because raw bytes (e.g., `b"\n\n"`) appeared in sequence without segmentation.
- **Fix**: Apply `PAT.findall(text)` first to segment the input before converting to bytes and applying BPE.
- **Impact**: Prevents undesired merges (like `"\n\n"`), ensuring compatibility with GPT-2.
- **Failure Example**:
	```python
	text = "<|endoftext|>\n\nOnce upon a time"
	# Segments without PAT: [b'<|endoftext|>', b'\n', b'\n', b'Once'] → \n\n may merge
	# Segments with PAT: ['<|endoftext|>', '\n', '\n', 'Once'] → preserves boundary
	```

### Per-token Byte Encoding Before BPE
```python
for segment_str in initial_segments_str:
    tokens.extend([bytes([b]) for b in segment_str.encode("utf-8")])
```
- **Why**: GPT-2’s tokenizer applies BPE after converting each regex-tokenized segment into UTF-8 byte sequences.
- **Bug**: Previously, the entire input string was encoded into bytes before splitting — allowing merges across regex boundaries.
- **Fix**: First tokenize with PAT, then byte-encode each segment individually.
- **Impact**: Ensures BPE merges only happen within allowed segment boundaries, matching GPT-2 behavior.
- **Failure Example**:
	```python
	text = "friend.\n<|endoftext|>\n\nOnce upon\n\n a time"
	# Without segmenting: [b'friend.', b'\n', b'<|endoftext|>', b'\n\nOnce upon'] → incorrect merges
	# With PAT: ['friend', '.', '\n', '\n', 'Once', ' upon', '\n\n', ' a', ' time'] → correct boundaries
	```


### Q. Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
**Compresion Ratio on TinyStories Tokenizer:**
- Compression Ratio for our tokenizer: 4.176 bytes/token
- Compression Ratio for GPT-2 tokenizer: 4.081 bytes/token

**Compression Ratio on OpenWebText Tokenizer:**
- Compression Ratio for our tokenizer: 4.583 bytes/token
- Compression Ratio for GPT-2 tokenizer: 4.618 bytes/token

### Q. What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
**Compression Ratio on OpenWebText using Tinystories Tokenizer:**
- Compression Ratio for our tokenizer: 3.324 bytes/token
- Compression Ratio for GPT-2 tokenizer: 4.653 bytes/token

### Q. Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?

| **Tokenizer Trained On** | **Used On** | **Tokenizer** | **Throughput (bytes/sec)** | **Time to Tokenize Pile (825 GB)** |
| ------------------------ | ----------- | ------------- | -------------------------- | ---------------------------------- |
| TinyStories (10K vocab)  | TinyStories | GPT-2 (50K)   | **12,644,669**             | **\~19.46 hours**                  |
| TinyStories (10K vocab)  | TinyStories | Ours (10K)    | **180,705**                | **\~1,361.70 hours**               |
| OpenWebText (32K vocab)  | OpenWebText | GPT-2 (50K)   | **5,393,412**              | **\~45.62 hours**                  |
| OpenWebText (32K vocab)  | OpenWebText | Ours (32K)    | **39,351**                 | **\~6,253.04 hours**               |

### Analysis:
- GPT-2 tokenizer is significantly faster than both custom-trained tokenizers across both datasets.
- When used on TinyStories, GPT-2 achieves a throughput of ~12.6M bytes/sec, vs ~180K bytes/sec from the custom TinyStories tokenizer — making it ~70× faster.
- On OpenWebText, GPT-2 is still over 137× faster than the custom OpenWebText tokenizer.
- This performance gap is primarily due to:
	- Highly optimized C++ backend in GPT-2’s tokenizer (via tiktoken).
	- Efficient merge logic and precompiled data structures.
	- Custom BPE tokenizers written in Python lack these performance optimizations.

### Q. Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
- Vocabulary size determines how many unique token IDs you'll need to store.
- `uint16` can represent values from 0 to 65,535.
- Our tokenizers use:
	- TinyStories vocab: 10K tokens
	- OpenWebText vocab: 32K tokens
	- Both are well within the range of `uint16`.
- Uses **2 bytes per token**, reducing memory/disk usage compared to `int32` or `int64`