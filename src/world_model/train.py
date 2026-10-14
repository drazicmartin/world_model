"""Single-device training with resumable, weights-only-compatible checkpoints."""

import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, default_convert
from .config import Config
from .data import Windows, split_episodes
from .envs import ActionCodec
from .model import WorldModel
from .progress import Progress, stage, track

UPSTREAM_COMMIT = "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac"


def resolve_device(device):
    return (
        "cuda"
        if device == "auto" and torch.cuda.is_available()
        else ("cpu" if device == "auto" else device)
    )


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format") != 1 or checkpoint["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Unsupported checkpoint format or upstream revision")
    cfg = Config(**checkpoint["config"])
    model = WorldModel(cfg, ActionCodec(checkpoint["action_spec"]).dim)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval(), cfg, checkpoint


def check_dataset(cfg, manifest):
    if not manifest.get("complete") or manifest.get("format") != 1:
        raise ValueError("Dataset is incomplete or unsupported")
    for key in ("env_id", "factory", "env_kwargs", "image_size", "action_repeat"):
        if getattr(cfg, key) != manifest["config"][key]:
            raise ValueError(f"Dataset/config mismatch: {key}")


def reward_stats(root, episodes):
    total = total_sq = 0.0
    count = 0
    for episode in track(episodes, "Computing reward statistics", "episode"):
        x = np.load(Path(root) / episode["name"] / "rewards.npy", mmap_mode="r")
        total += x.sum(dtype=np.float64)
        total_sq += np.square(x.astype(np.float64)).sum()
        count += len(x)
    mean = total / count
    return float(mean), float(max(np.sqrt(max(0, total_sq / count - mean * mean)), 1e-3))


def run_epoch(model, loader, cfg, device, optimizer=None, description="Batches"):
    model.train(optimizer is not None)
    totals, count = {}, 0
    with (
        torch.set_grad_enabled(optimizer is not None),
        Progress(len(loader), description, "batch") as progress,
    ):
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            losses = model.losses(batch, cfg.sigreg_weight)
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Nonfinite loss; inspect data and learning rate")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = len(batch["pixels"])
            count += n
            for key, value in losses.items():
                totals[key] = totals.get(key, 0.0) + value.item() * n
            progress.update(loss=f"{totals['loss'] / count:.4f}")
    return {key: value / count for key, value in totals.items()}


def atomic_save(checkpoint, path):
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def train(cfg, data, output, device="auto", resume=None):
    device = resolve_device(device)
    stage(
        f"Training setup | env={cfg.env_id} | device={device} | "
        f"epochs={cfg.epochs} | batch_size={cfg.batch_size} | data={data} | output={output}"
    )
    seed_all(cfg.seed)
    manifest_text = (Path(data) / "manifest.json").read_text()
    manifest = json.loads(manifest_text)
    check_dataset(cfg, manifest)
    fingerprint = hashlib.sha256(manifest_text.encode()).hexdigest()
    training, validation = split_episodes(manifest, cfg.seed, cfg.validation_fraction)
    datasets = [Windows(data, episodes, cfg.history) for episodes in (training, validation)]
    if cfg.batch_size < 2 or len(datasets[0]) < cfg.batch_size:
        raise ValueError("Need batch_size >= 2 and at least one full training batch for SIGReg")
    size = sum(dataset.nbytes() for dataset in datasets)
    workers = cfg.workers
    if size <= cfg.preload_max_gb * 1e9:
        stage(
            f"Loading dataset into memory | {size / 1e6:.0f} MB <= preload_max_gb="
            f"{cfg.preload_max_gb} | batches are gathered in-process, no loader workers"
        )
        for dataset in datasets:
            dataset.preload()
        workers = 0
    else:
        stage(
            f"Streaming dataset from disk | {size / 1e6:.0f} MB > preload_max_gb="
            f"{cfg.preload_max_gb} | loader workers={workers}"
        )
    generator = torch.Generator().manual_seed(cfg.seed)
    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=workers > 0,
        collate_fn=default_convert,  # Windows.__getitems__ already returns stacked batches
    )
    train_loader = DataLoader(
        datasets[0], shuffle=True, drop_last=True, generator=generator, **loader_kwargs
    )
    val_loader = DataLoader(datasets[1], shuffle=False, **loader_kwargs)
    codec = ActionCodec(manifest["action_spec"])
    stage(
        f"Building model | train_windows={len(datasets[0])} | validation_windows={len(datasets[1])}"
    )
    model = WorldModel(cfg, codec.dim).to(device)
    mean, std = reward_stats(data, training)
    model.reward_mean.fill_(mean)
    model.reward_std.fill_(std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start, best = 0, float("inf")
    output = Path(output)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        if Path(resume).resolve().parent != output.resolve():
            raise ValueError("Resume into the original run directory to preserve best.pt and logs")
        stage(f"Resuming checkpoint: {resume}")
        _, old_cfg, saved = load_checkpoint(resume)
        comparable = cfg.to_dict()
        comparable["epochs"] = old_cfg.epochs
        comparable["preload_max_gb"] = old_cfg.preload_max_gb  # same batches either way
        if comparable != old_cfg.to_dict() or saved["dataset_fingerprint"] != fingerprint:
            raise ValueError(
                "Resume requires same config (except epochs, preload_max_gb) and dataset manifest"
            )
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start, best = saved["epoch"] + 1, saved["best_metric"]
        torch.set_rng_state(saved["torch_rng"])
        if device.startswith("cuda") and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["loader_rng"])
        output.mkdir(parents=True, exist_ok=True)
    if start >= cfg.epochs:
        raise ValueError("epochs must exceed the checkpoint's completed epoch count")
    (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    (output / "split.json").write_text(
        json.dumps(
            {
                "training": [e["name"] for e in training],
                "validation": [e["name"] for e in validation],
                "dataset_fingerprint": fingerprint,
            },
            indent=2,
        )
    )
    for epoch in range(start, cfg.epochs):
        stage(f"Epoch {epoch + 1}/{cfg.epochs} | training, then validation, then checkpoint save")
        train_metrics = run_epoch(
            model,
            train_loader,
            cfg,
            device,
            optimizer,
            description=f"Epoch {epoch + 1}/{cfg.epochs} - train",
        )
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg.seed + 10000)
            val_metrics = run_epoch(
                model,
                val_loader,
                cfg,
                device,
                description=f"Epoch {epoch + 1}/{cfg.epochs} - validation",
            )
        metric = sum(val_metrics[key] for key in ("prediction", "reward", "termination"))
        improved = metric < best
        best = min(best, metric)
        checkpoint = {
            "format": 1,
            "upstream_commit": UPSTREAM_COMMIT,
            "config": cfg.to_dict(),
            "action_spec": codec.spec,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": best,
            "dataset_fingerprint": fingerprint,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_rng": generator.get_state(),
        }
        stage(f"Saving checkpoint | epoch={epoch + 1} | new_best={improved}")
        atomic_save(checkpoint, output / "last.pt")
        if improved:
            atomic_save(checkpoint, output / "best.pt")
        row = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(row) + "\n")
        stage(
            f"Epoch {epoch + 1}/{cfg.epochs} complete | train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | best_metric={best:.4f}"
        )
    stage(
        f"Training complete | best={output / 'best.pt'} | last={output / 'last.pt'} | "
        f"metrics={output / 'metrics.jsonl'}"
    )
    return output / "best.pt"
