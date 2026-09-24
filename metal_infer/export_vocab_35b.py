#!/usr/bin/env python3
import json
import struct
import sys
from pathlib import Path



def bytes_to_unicode():
    """GPT-2 / ByteLevel byte<->unicode table, as used by the tokenizer."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def main():
    tok_path = sys.argv[1] if len(sys.argv) > 1 else "tokenizer.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "vocab.bin"

    tok_path = str(Path(tok_path).expanduser())
    out_path = str(Path(out_path).expanduser())

    with open(tok_path, "r", encoding="utf-8") as f:
        t = json.load(f)

    vocab = t["model"]["vocab"]          # str -> int
    added = t.get("added_tokens", [])    # list of {id, content, ...}

    id_to_text = {}

    # Store each token's RAW BYTES, recovered by inverting the ByteLevel mapping.
    #
    # Do NOT use tokenizer.decode([id]) here. This is a byte-level BPE: one
    # character can span several tokens, and a token holding a lone continuation
    # byte is not valid UTF-8 on its own, so decoding it in isolation yields
    # U+FFFD. Doing that baked 1061 replacement characters into the old vocab.bin
    # and destroyed every emoji and any character whose bytes straddled a token
    # boundary -- irrecoverably, since the real bytes never reached the runtime.
    # Concatenating raw bytes at generation time reassembles them correctly.
    byte_decoder = {ch: b for b, ch in bytes_to_unicode().items()}

    id_to_bytes = {}
    unmapped = 0
    for token_str, token_id in vocab.items():
        try:
            id_to_bytes[token_id] = bytes(byte_decoder[ch] for ch in token_str)
        except KeyError:
            # Not expressible in the ByteLevel alphabet (shouldn't happen for a
            # ByteLevel model); fall back to UTF-8 of the literal token string.
            id_to_bytes[token_id] = token_str.encode("utf-8")
            unmapped += 1

    # Added tokens (<|im_start|> and friends) are literal text, not byte-mapped.
    for tok in added:
        id_to_bytes[tok["id"]] = tok["content"].encode("utf-8")

    id_to_text = id_to_bytes
    if unmapped:
        print(f"  warning: {unmapped} token(s) not in the ByteLevel alphabet")

    max_id = max(id_to_text.keys())
    num_entries = max_id + 1

    with open(out_path, "wb") as f:
        f.write(struct.pack("<I", num_entries))
        f.write(struct.pack("<I", max_id))

        for token_id in range(num_entries):
            b = id_to_text.get(token_id, b"")
            if len(b) > 65535:
                raise ValueError(f"Token {token_id} too long: {len(b)} bytes")
            f.write(struct.pack("<H", len(b)))
            f.write(b)

    print(f"Exported legacy vocab to {out_path}")
    print(f"  num_entries: {num_entries}")
    print(f"  max_id: {max_id}")


if __name__ == "__main__":
    main()
