import os
import json
import glob
import requests
from tqdm import tqdm
import random
import collections
from multiprocessing import Pool, Manager, current_process
import regex as re
from typing import BinaryIO
from functools import lru_cache
import mmap
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from typing import IO, BinaryIO, Iterable, Optional, Type, Iterator
import tiktoken
import time


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
        # print(segments)
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
    
@lru_cache()
def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    characters = [chr(n) for n in cs]
    return dict(zip(bs, characters))


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: Optional[list[str]] = None,
):
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)
    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
    # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
    # just return the original bytes, so we don't force students to use
    # any particular encoding scheme.
    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }
    # If any of the special tokens don't exist in the vocab, append them to the vocab.
    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return Tokenizer(vocab, merges, special_tokens)

    
if __name__ == "__main__":
    VOCAB_PATH = "../data_cache/tinystories/vocab_tinystories.json" #"./fixtures/gpt2_vocab.json"
    MERGES_PATH =  "../data_cache/tinystories/merges_tinystories.txt" #"./fixtures/gpt2_merges.txt"
    
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"]
    )
    # with open("../data_cache/tinystories/tiny_10.txt") as f:
    #     text = f.read()
    # text = "friend.\n<|endoftext|>\n\nOnce upon\n\na time"
    encoded_ids = []
    reference_ids = []
    num_bytes = 0
    reference_tokenizer = tiktoken.get_encoding("gpt2")
    i = 0
    start = time.time()
    with open("../data_cache/tinystories/tiny_10.txt") as f:
        for line in tqdm(f):
            # encoded_ids.extend(tokenizer.encode(line))
            reference_ids.extend(reference_tokenizer.encode(
                line, allowed_special={"<|endoftext|>"}))
            num_bytes += len(line.encode("utf-8"))
            i += 1
            if i == 2001:
                break
    end = time.time()
    throughput = num_bytes / (end - start)
    print(f"Throughput: {throughput:.3f}")
    pile_size_bytes = 825 * (1024**3)
    time_required = pile_size_bytes / throughput
    print(f"Time Required to Tokenize Pile Dataset: {(time_required / 3600):.2f}hrs")
    
    cr_tok = num_bytes / len(encoded_ids)
    cr_ref = num_bytes / len(reference_ids)
    print(f"Compression Ratio for our tokenizer: {cr_tok:.3f}" )
    print(f"Compression Ratio for GPT-2 tokenizer: {cr_ref:.3f}" )