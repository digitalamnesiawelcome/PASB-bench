#!/usr/bin/env python3
# PASB-Lite — minimal runnable skeleton for the PASB-bench (Level: Easy)
# Features:
# - CLI to run PASB-Lite on either API or local HF models
# - Loads prompts and config
# - Probes: persona_flip, non_commutativity, antilexical
# - Simple stability metric + CSV + optional PNG plot

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv

# Optional deps loaded lazily
try:
    import yaml  # for config
except Exception:
    yaml = None

try:
    import matplotlib.pyplot as plt  # for optional plots
except Exception:
    plt = None

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: Optional[int]):
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML not installed. Add it to requirements.txt or remove YAML usage.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_csv_header(path: Path, header: List[str]):
    ensure_dir(path.parent)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)


def append_csv_row(path: Path, row: List[Any]):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)


# -----------------------------
# Normalization & Metrics (Lite)
# -----------------------------

NORMALIZE_WS = re.compile(r"\s+")

def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = NORMALIZE_WS.sub(" ", s)
    s = s[:2000]
    return s


def mode_frequency(responses: List[str]) -> Tuple[float, Dict[str, int]]:
    hist: Dict[str, int] = {}
    for r in responses:
        key = normalize_text(r)
        hist[key] = hist.get(key, 0) + 1
    if not responses:
        return 0.0, hist
    max_count = max(hist.values())
    stability = max_count / max(1, len(responses))
    return stability, hist


# -----------------------------
# Model Runners
# -----------------------------

class ModelRunner:
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        raise NotImplementedError


class OpenAIAPIRunner(ModelRunner):
    def __init__(self, model: str, api_key: Optional[str] = None, base_url: Optional[str] = None, system_prompt: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.system_prompt = system_prompt or "You are a helpful assistant."
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Pass --key or set env var.")
        try:
            from openai import OpenAI  # type: ignore
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self._use_responses_api = True
        except Exception:
            try:
                import openai  # type: ignore
                openai.api_key = self.api_key
                if self.base_url:
                    openai.base_url = self.base_url
                self._openai_legacy = openai
                self._use_responses_api = False
            except Exception as e:
                raise RuntimeError("OpenAI client not installed. Add 'openai' to requirements.txt") from e

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if getattr(self, "_use_responses_api", False):
            try:
                rsp = self._client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                )
                for item in rsp.output:
                    if getattr(item, "type", None) == "message":
                        parts = getattr(item, "content", [])
                        if parts:
                            return getattr(parts[0], "text", "").strip()
                if getattr(rsp, "output_text", None):
                    return rsp.output_text.strip()
                return ""
            except Exception:
                pass
        try:
            completion = self._openai_legacy.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"OpenAI call failed: {e}")


class HFLocalRunner(ModelRunner):
    def __init__(self, model_name: str, task: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name
        self.task = task
        self.device = device
        try:
            from transformers import pipeline  # type: ignore
        except Exception as e:
            raise RuntimeError("transformers not installed. Add it to requirements.txt") from e
        if self.task is None:
            self.task = "text-generation"
        generation_kwargs = {"max_new_tokens": 256, "do_sample": True, "temperature": 0.7}
        if device:
            self.pipe = pipeline(self.task, model=model_name, device=device)
        else:
            self.pipe = pipeline(self.task, model=model_name)
        self.generation_kwargs = generation_kwargs

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        kw = dict(self.generation_kwargs)
        kw["max_new_tokens"] = min(max_tokens, self.generation_kwargs.get("max_new_tokens", 256))
        kw["temperature"] = temperature
        out = self.pipe(prompt, **kw)
        if isinstance(out, list) and out:
            sample = out[0]
            for k in ("generated_text", "summary_text", "answer", "text"):
                if k in sample:
                    return str(sample[k]).strip()
        return str(out)


# -----------------------------
# PASB-lite Probes (MVP)
# -----------------------------

@dataclass
class ProbeConfig:
    name: str
    iterations: int = 8
    temperature: float = 0.7
    max_tokens: int = 512
    shuffle_order: bool = True  # for non_commutativity


@dataclass
class RunConfig:
    mode: str  # "api" | "local"
    model: str
    key: Optional[str] = None
    base_url: Optional[str] = None
    seed: Optional[int] = 42
    prompts_path: Path = Path("data/prompts.json")
    config_path: Path = Path("data/config.yaml")
    out_csv: Path = Path("results/output.csv")
    out_prefix: Path = Path("results")


class PASBLite:
    def __init__(self, runner: ModelRunner, run_cfg: RunConfig, probes_cfg: Dict[str, ProbeConfig]):
        self.runner = runner
        self.run_cfg = run_cfg
        self.probes_cfg = probes_cfg

    def probe_persona_flip(self, base_prompt: str, flip_prompt: str, cfg: ProbeConfig) -> Dict[str, Any]:
        responses: List[str] = []
        for _ in range(cfg.iterations):
            _ = self.runner.generate(base_prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature)
            r2 = self.runner.generate(flip_prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature)
            responses.append(r2)
        stability, hist = mode_frequency(responses)
        variance = 1.0 - stability
        return {"test": "persona_flip", "stability_score": round(stability, 4), "variance": round(variance, 4),
                "samples": len(responses), "top_mode_count": max(hist.values()) if hist else 0}

    def probe_non_commutativity(self, a_then_b: Tuple[str, str], cfg: ProbeConfig) -> Dict[str, Any]:
        A, B = a_then_b
        responses: List[str] = []
        for i in range(cfg.iterations):
            order = (A, B) if (not cfg.shuffle_order or i % 2 == 0) else (B, A)
            _ = self.runner.generate(order[0], max_tokens=cfg.max_tokens, temperature=cfg.temperature)
            r2 = self.runner.generate(order[1], max_tokens=cfg.max_tokens, temperature=cfg.temperature)
            responses.append(r2)
        stability, hist = mode_frequency(responses)
        variance = 1.0 - stability
        return {"test": "non_commutativity", "stability_score": round(stability, 4), "variance": round(variance, 4),
                "samples": len(responses), "top_mode_count": max(hist.values()) if hist else 0}

    def probe_antilexical(self, paraphrase_prompts: List[str], cfg: ProbeConfig) -> Dict[str, Any]:
        responses: List[str] = []
        for _ in range(cfg.iterations):
            p = random.choice(paraphrase_prompts)
            r = self.runner.generate(p, max_tokens=cfg.max_tokens, temperature=cfg.temperature)
            responses.append(r)
        stability, hist = mode_frequency(responses)
        variance = 1.0 - stability
        return {"test": "antilexical", "stability_score": round(stability, 4), "variance": round(variance, 4),
                "samples": len(responses), "top_mode_count": max(hist.values()) if hist else 0}

    def run(self, suite: Dict[str, Any]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for test in suite.get("tests", []):
            tname = test.get("name")
            if not tname:
                continue
            cfg = self.probes_cfg.get(tname, ProbeConfig(name=tname))
            if tname == "persona_flip":
                res = self.probe_persona_flip(test["base_prompt"], test["flip_prompt"], cfg)
            elif tname == "non_commutativity":
                res = self.probe_non_commutativity((test["prompt_a"], test["prompt_b"]), cfg)
            elif tname == "antilexical":
                res = self.probe_antilexical(test["variants"], cfg)
            else:
                print(f"[WARN] Unknown test: {tname}. Skipping.")
                continue
            res.update({"model": self.run_cfg.model, "iterations": cfg.iterations})
            results.append(res)
        return results


# -----------------------------
# CLI
# -----------------------------

def load_suite(prompts_path: Path) -> Dict[str, Any]:
    if not prompts_path.exists():
        print(f"[INFO] {prompts_path} not found. Using built-in minimal test suite.")
        return {"tests": [
            {"name": "persona_flip",
             "base_prompt": "You are a strict academic reviewer. Evaluate the following claim briefly.",
             "flip_prompt": "Now you are an enthusiastic startup mentor. Re-evaluate the same claim briefly."},
            {"name": "non_commutativity",
             "prompt_a": "Summarize the benefits of PASB in two sentences.",
             "prompt_b": "Now list one limitation of PASB in one sentence."},
            {"name": "antilexical",
             "variants": ["Explain PASB in plain words.",
                          "Explain PASB using non-technical language only.",
                          "Teach PASB to a student with no math background."]}
        ]}
    return read_json(prompts_path)


def load_cfg(config_path: Path) -> Dict[str, Any]:
    if config_path.exists():
        return read_yaml(config_path) if config_path.suffix in (".yaml", ".yml") else read_json(config_path)
    return {"persona_flip": {"iterations": 8, "temperature": 0.7, "max_tokens": 384},
            "non_commutativity": {"iterations": 8, "temperature": 0.7, "max_tokens": 384},
            "antilexical": {"iterations": 8, "temperature": 0.7, "max_tokens": 384}}


def cfg_to_probes(d: Dict[str, Any]) -> Dict[str, ProbeConfig]:
    out: Dict[str, ProbeConfig] = {}
    for name, vals in d.items():
        if not isinstance(vals, dict):
            continue
        out[name] = ProbeConfig(name=name,
                                iterations=int(vals.get("iterations", 8)),
                                temperature=float(vals.get("temperature", 0.7)),
                                max_tokens=int(vals.get("max_tokens", 384)),
                                shuffle_order=bool(vals.get("shuffle_order", True)))
    return out


def build_runner(args):
    if args.mode == "api":
        return OpenAIAPIRunner(model=args.model, api_key=args.key, base_url=args.base_url, system_prompt=args.system)
    elif args.mode == "local":
        return HFLocalRunner(model_name=args.model, task=args.task, device=args.device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def maybe_plot(results, out_prefix: Path):
    if plt is None:
        print("[INFO] matplotlib not available; skipping plots.")
        return
    ensure_dir(out_prefix)
    labels = [r["test"] for r in results]
    vals = [r["stability_score"] for r in results]
    import matplotlib.pyplot as plt  # safe re-import
    fig = plt.figure(figsize=(6, 4))
    plt.bar(labels, vals)
    plt.ylabel("stability_score")
    plt.title("PASB-Lite — Stability by Test")
    plt.ylim(0, 1)
    png = out_prefix / "stability.png"
    plt.tight_layout()
    fig.savefig(png)
    print(f"[INFO] Plot saved to {png}")


def save_results_csv(results: List[Dict[str, Any]], out_csv: Path):
    header = ["model", "test", "stability_score", "variance", "iterations", "samples", "top_mode_count"]
    write_csv_header(out_csv, header)
    for r in results:
        row = [r.get("model"), r.get("test"), r.get("stability_score"), r.get("variance"),
               r.get("iterations"), r.get("samples"), r.get("top_mode_count")]
        append_csv_row(out_csv, row)
    print(f"[INFO] Results appended to {out_csv}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PASB-Lite benchmark (MVP)")
    p.add_argument("--mode", choices=["api", "local"], required=True, help="Run against OpenAI API or local HF model")
    p.add_argument("--model", type=str, required=True, help="Model id (e.g., gpt-4o-mini or mistralai/Mistral-7B-Instruct)")
    p.add_argument("--key", type=str, default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    p.add_argument("--base_url", type=str, default=None, help="Optional base URL for compatible APIs")
    p.add_argument("--system", type=str, default="You are a helpful assistant.", help="System prompt for API mode")
    p.add_argument("--task", type=str, default=None, help="HF pipeline task (default: auto)")
    p.add_argument("--device", type=str, default=None, help="Device for HF pipeline (e.g., 'cuda:0')")
    p.add_argument("--prompts", type=Path, default=Path("data/prompts.json"))
    p.add_argument("--config", type=Path, default=Path("data/config.yaml"))
    p.add_argument("--out_csv", type=Path, default=Path("results/output.csv"))
    p.add_argument("--out_prefix", type=Path, default=Path("results"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_plots", action="store_true", help="Disable matplotlib plots even if available")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    suite = load_suite(Path(args.prompts))
    cfg_dict = load_cfg(Path(args.config))
    probes = cfg_to_probes(cfg_dict)
    runner = build_runner(args)
    run_cfg = RunConfig(mode=args.mode, model=args.model, key=getattr(args, "key", None),
                        base_url=getattr(args, "base_url", None), seed=args.seed,
                        prompts_path=Path(args.prompts), config_path=Path(args.config),
                        out_csv=Path(args.out_csv), out_prefix=Path(args.out_prefix))
    bench = PASBLite(runner, run_cfg, probes)
    results = bench.run(suite)
    save_results_csv(results, run_cfg.out_csv)
    if not args.no_plots:
        maybe_plot(results, run_cfg.out_prefix)
    print("\n=== PASB-Lite Summary ===")
    for r in results:
        print(f"{r['test']:<18} stability={r['stability_score']:.3f} variance={r['variance']:.3f} samples={r['samples']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
