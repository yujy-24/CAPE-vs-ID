"""Fingerprint a DINOv3 safetensors snapshot without loading the model.

Usage:
    python tools/fingerprint_dino.py /path/to/dinov3-vitb16-pretrain-lvd1689m

The tensor digest hashes each tensor's name, dtype, shape, and raw bytes in
sorted-name order. It remains stable if identical tensors are stored in a
different order or with different safetensors header metadata.
"""

import argparse
import hashlib
import json
import mmap
import struct
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensors(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        if len(mapped) < 8:
            raise ValueError("safetensors file is too short")
        header_size = struct.unpack("<Q", mapped[:8])[0]
        data_start = 8 + header_size
        if data_start > len(mapped):
            raise ValueError("safetensors header extends beyond the file")
        header = json.loads(mapped[8:data_start])
        names = sorted(name for name in header if name != "__metadata__")
        for name in names:
            spec = header[name]
            begin, end = spec["data_offsets"]
            if begin < 0 or end < begin or data_start + end > len(mapped):
                raise ValueError(f"invalid tensor offsets for {name}")
            digest.update(name.encode("utf-8") + b"\0")
            digest.update(spec["dtype"].encode("ascii") + b"\0")
            digest.update(json.dumps(spec["shape"], separators=(",", ":")).encode("ascii") + b"\0")
            for offset in range(begin, end, 4 * 1024 * 1024):
                stop = min(end, offset + 4 * 1024 * 1024)
                digest.update(mapped[data_start + offset:data_start + stop])
    return digest.hexdigest(), len(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    weights = model_dir / "model.safetensors"
    config = model_dir / "config.json"
    preprocessor = model_dir / "preprocessor_config.json"
    for path in (weights, config, preprocessor):
        if not path.is_file():
            parser.error(f"missing {path}")

    tensor_digest, tensor_count = sha256_tensors(weights)
    preprocess_data = json.loads(preprocessor.read_text(encoding="utf-8"))
    record = {
        "model_dir": str(model_dir),
        "weights_size_bytes": weights.stat().st_size,
        "weights_file_sha256": sha256_file(weights),
        "weights_tensor_sha256": tensor_digest,
        "tensor_count": tensor_count,
        "config_sha256": sha256_file(config),
        "preprocessor_sha256": sha256_file(preprocessor),
        "image_mean": preprocess_data.get("image_mean"),
        "image_std": preprocess_data.get("image_std"),
    }
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
