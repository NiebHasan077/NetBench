"""Loads config.yaml and exposes it as a typed Config object."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Paths(BaseModel):
    corpus: str
    v4_benchmark: str
    data_dir: str
    benchmark_dir: str
    reports_dir: str
    prompts_dir: str
    logs_dir: str
    openai_key_file: Optional[str] = None
    gemini_key_file: Optional[str] = None
    openai_key_env: str = "OPENAI_API_KEY"
    gemini_key_env: str = "GEMINI_API_KEY"


class CalibrationModels(BaseModel):
    weak: str
    mid: str
    strong: str


class Models(BaseModel):
    ollama_host: str
    card_generator: str
    question_generator: str
    question_generator_fallback: str
    critic: str
    calibration: CalibrationModels
    embedding: str
    judge_a: str
    judge_b: str


class OllamaCfg(BaseModel):
    request_timeout_seconds: int
    max_retries: int
    json_retry_temperature: float
    default_temperature: float
    default_num_predict: int


class Chunking(BaseModel):
    window_tokens: int
    overlap_tokens: int


class FilterCfg(BaseModel):
    top_k_papers: int
    min_score: float


class Clustering(BaseModel):
    min_cluster_size: int
    cluster_selection_method: str = "eom"


class Generation(BaseModel):
    per_card_questions: int
    synthesis_pairs_per_cluster: int
    synthesis_target: int
    adversarial_target: int


class Validation(BaseModel):
    ngram_leak_size: int
    leak_cosine_threshold: float
    dedup_cosine_threshold: float


class Splits(BaseModel):
    benchmark_version: str
    dev: int
    test: int
    synthesis: int
    adversarial: int
    total: int


class Config(BaseModel):
    paths: Paths
    models: Models
    ollama: OllamaCfg
    chunking: Chunking
    filter: FilterCfg
    clustering: Clustering
    generation: Generation
    validation: Validation
    splits: Splits
    seed: int

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    def resolve(self, p: str) -> Path:
        path = Path(p)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        return path


@lru_cache(maxsize=4)
def load_config(path: Optional[str] = None) -> Config:
    cfg_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(**data)
