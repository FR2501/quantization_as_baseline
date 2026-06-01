from abc import ABC, abstractmethod
import os
import subprocess
import shutil

class EntropyCoder(ABC):

    @abstractmethod
    def encode(self, input_path: str, output_path: str) -> None:
        pass

    @abstractmethod
    def decode(self, input_path: str, output_path: str) -> None:
        pass

    @abstractmethod
    def get_unique_identifier(self) -> str:
        pass


class BSC(EntropyCoder):

    def encode(self, input_path: str, output_path: str) -> None:
        subprocess.run(['./bsc', 'e', input_path, output_path, '-b64p', '-e2'], capture_output=True)

    def decode(self, input_path: str, output_path: str) -> None:
        subprocess.run(['./bsc', 'd', input_path, output_path], capture_output=True)

    def get_unique_identifier(self):
        return "bsc"

class Zstd(EntropyCoder):

    def encode(self, input_path: str, output_path: str) -> None:
        subprocess.run(['zstd', '-f', input_path, '-o', str(output_path)], capture_output=False)

    def decode(self, input_path: str, output_path: str) -> None:
        subprocess.run(['zstd', '-df', input_path, '-o', output_path], capture_output=False)

    def get_unique_identifier(self):
        return "zstd"

class GZip(EntropyCoder):

    def encode(self, input_path: str, output_path: str) -> None:
        print("enc", input_path, output_path)
        subprocess.run(['gzip', input_path, '-k'], capture_output=True)
        shutil.move(str(input_path) + ".gz", str(output_path) + ".gz")
        os.rename(str(output_path) + ".gz", output_path)

    def decode(self, input_path: str, output_path: str) -> None:
        print("dec", input_path, output_path)
        os.rename(str(input_path), str(input_path) + ".gz")
        subprocess.run(['gzip', '-d', str(input_path) + ".gz", "-k"], capture_output=False)
        shutil.move(str(input_path), output_path)
        os.rename(str(input_path) + ".gz", str(input_path))

    def get_unique_identifier(self):
        return "gzip"


# ----------------------------------------------------------------------
# Huffman-only implementation by Claude Sonnet 4.6
# ----------------------------------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import dataclass
import struct
from collections import Counter
from heapq import heappush, heappop

@dataclass
class _HuffNode:
    freq: int
    symbol: int | None = None   # leaf: 0–255; internal: None
    left: "_HuffNode | None" = None
    right: "_HuffNode | None" = None

    # heapq uses <, we order by freq only
    def __lt__(self, other: "_HuffNode"):
        return self.freq < other.freq


class HuffmanCompressor(EntropyCoder):
    """
    Byte-oriented, static Huffman-only compressor.

    Format:
      - 4 bytes: magic b'HUF0'
      - 1 byte:  version (0)
      - 256 * 4 bytes: uint32 frequency table
      - 4 bytes: uint32 original size in bytes
      - remaining: Huffman bitstream (MSB-first).
    """

    MAGIC = b"HUF0"
    VERSION = 0

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def get_error_bound(self):
        # Lossless; "no error" – adapt to your framework as needed.
        return 0.0

    def get_unique_identifier(self) -> str:
        return "huffman"

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, input_path, output_path) -> None:
        with open(input_path, "rb") as f:
            data = f.read()

        freqs = self._build_freqs(data)
        codes = self._build_codes(freqs)

        # Encode to bitstring as bytes
        bitstream = self._encode_bytes(data, codes)

        with open(output_path, "wb") as out:
            # header
            out.write(self.MAGIC)
            out.write(struct.pack("B", self.VERSION))
            # frequency table: 256 uint32 little-endian
            for s in range(256):
                out.write(struct.pack("<I", freqs[s]))
            # original size
            out.write(struct.pack("<I", len(data)))
            # payload
            out.write(bitstream)

    def decode(self, input_path, output_path) -> None:
        with open(input_path, "rb") as f:
            magic = f.read(4)
            if magic != self.MAGIC:
                raise ValueError("Not a HuffmanCompressor stream (bad magic).")
            version = struct.unpack("B", f.read(1))[0]
            if version != self.VERSION:
                raise ValueError(f"Unsupported version {version}.")

            # read freq table
            freqs = [struct.unpack("<I", f.read(4))[0] for _ in range(256)]
            original_size = struct.unpack("<I", f.read(4))[0]
            bit_bytes = f.read()

        # rebuild codes and decode
        codes = self._build_codes(freqs)
        data = self._decode_bytes(bit_bytes, codes, original_size)
        with open(output_path, "wb") as out:
            out.write(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_freqs(data: bytes) -> list[int]:
        c = Counter(data)
        freqs = [0] * 256
        for b, count in c.items():
            freqs[b] = count
        # edge case: empty file
        if not data:
            return freqs
        # edge case: all bytes same → ensure at least one more symbol
        if sum(1 for f in freqs if f > 0) == 1:
            # pick some other symbol with freq 0 and set freq 1,
            # so Huffman tree has at least 2 leaves
            s0 = next(i for i, f in enumerate(freqs) if f > 0)
            s1 = (s0 + 1) % 256
            if freqs[s1] == 0:
                freqs[s1] = 1
        return freqs

    @staticmethod
    def _build_codes(freqs: list[int]) -> dict[int, str]:
        # Build Huffman tree
        heap = []
        for sym, f in enumerate(freqs):
            if f > 0:
                heappush(heap, _HuffNode(freq=f, symbol=sym))

        if not heap:
            # empty input; arbitrary single-node tree
            heap.append(_HuffNode(freq=1, symbol=0))

        while len(heap) > 1:
            n1 = heappop(heap)
            n2 = heappop(heap)
            parent = _HuffNode(freq=n1.freq + n2.freq, left=n1, right=n2)
            heappush(heap, parent)

        root = heap[0]

        codes: dict[int, str] = {}

        def traverse(node: _HuffNode, prefix: str):
            if node.symbol is not None:
                # leaf
                codes[node.symbol] = prefix or "0"  # handle single-symbol case
                return
            traverse(node.left, prefix + "0")
            traverse(node.right, prefix + "1")

        traverse(root, "")
        return codes

    @staticmethod
    def _encode_bytes(data: bytes, codes: dict[int, str]) -> bytes:
        bits = []
        append = bits.append
        for b in data:
            append(codes[b])
        bitstring = "".join(bits)

        # pad to full bytes
        pad = (8 - len(bitstring) % 8) % 8
        bitstring += "0" * pad

        # pack bits into bytes (MSB-first)
        out = bytearray()
        for i in range(0, len(bitstring), 8):
            byte = bitstring[i:i+8]
            out.append(int(byte, 2))
        return bytes(out)

    @staticmethod
    def _decode_bytes(bit_bytes: bytes,
                      codes: dict[int, str],
                      original_size: int) -> bytes:
        # Build prefix tree from codes
        root = {}
        for sym, code in codes.items():
            node = root
            for c in code:
                node = node.setdefault(c, {})
            node["sym"] = sym

        # Bit-by-bit decode
        data = bytearray()
        node = root
        count = 0
        for byte in bit_bytes:
            for bit_pos in range(7, -1, -1):
                b = "1" if (byte >> bit_pos) & 1 else "0"
                node = node.get(b)
                if node is None:
                    # this can happen in the padded tail; stop once all bytes recovered
                    return bytes(data)
                if "sym" in node:
                    data.append(node["sym"])
                    count += 1
                    if count == original_size:
                        return bytes(data)
                    node = root
        return bytes(data)


