import json
import os
import torch


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_torch(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path)


def load_torch(path):
    return torch.load(path, weights_only=False)


def save_text(text, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def append_jsonl(record, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def make_processed_dirs(output_root):
    """Create and return the standard processed subdirectory paths."""
    common_dir = os.path.join(output_root, "processed", "common")
    atom_dir = os.path.join(output_root, "processed", "atom")
    spe_dir = os.path.join(output_root, "processed", "spe")
    for d in [common_dir, atom_dir, spe_dir]:
        os.makedirs(d, exist_ok=True)
    return common_dir, atom_dir, spe_dir
