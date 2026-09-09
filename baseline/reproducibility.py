from __future__ import annotations
import os, random
from typing import Any
import numpy as np
import torch

def seed_everything(seed:int, deterministic:bool=False) -> dict[str,Any]:
    seed=int(seed); os.environ.setdefault("PYTHONHASHSEED", str(seed)); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    return {"seed": seed, "deterministic": bool(deterministic)}

def seed_worker(worker_id:int) -> None:
    seed=torch.initial_seed() % (2**32)
    random.seed(seed); np.random.seed(seed)

def make_generator(seed:int) -> torch.Generator:
    g=torch.Generator(); g.manual_seed(int(seed)); return g

def capture_rng_state(train_loader=None, val_loader=None) -> dict[str,Any]:
    out={"python":random.getstate(),"numpy":np.random.get_state(),"torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    for key, loader in (("train_loader",train_loader),("val_loader",val_loader)):
        gen=getattr(loader,"generator",None) if loader is not None else None
        if gen is not None: out[key]=gen.get_state()
    return out

def restore_rng_state(state:dict[str,Any], train_loader=None, val_loader=None) -> None:
    required={"python","numpy","torch_cpu"}
    if not isinstance(state,dict) or not required <= state.keys(): raise ValueError("rng_state is missing one or more generators")
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"): torch.cuda.set_rng_state_all(state["torch_cuda"])
    for key,loader in (("train_loader",train_loader),("val_loader",val_loader)):
        if loader is not None and key in state and getattr(loader,"generator",None) is not None: loader.generator.set_state(state[key])
