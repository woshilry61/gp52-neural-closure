"""Small, deterministic provenance records for reproducible training."""
from pathlib import Path
import hashlib
import torch

def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def initialization_record(model, seed):
    return {"seed": seed, "initial_state_sha256": state_sha256(model.state_dict()),
            "initialization": "fresh parameters; zero-initialized output head",
            "loaded_trained_head": False}
