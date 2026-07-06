from __future__ import annotations

import os

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class APIConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    base_url: str
    api_key_env: str


class StepAConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str
    temperature: float
    max_tokens: int
    thinking_mode: bool


class StepBConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    retrieval_mode: str  # rrf | bm25_only | dense_only
    top_bm25: int
    top_dense: int
    top_candidates: int
    k_rrf: int
    dense_model_name: str
    faiss_index_path: str
    corpus_path: str


class StepCConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_name: str | None  # null in base.yaml until Model Comparison Study resolves a winner
    k_verify: int
    snippet_count: int
    temperature: float
    max_tokens: int
    thinking_mode: bool


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    eval_sample_fraction: float
    results_dir: str
    metrics: list[str]


# Do NOT instantiate PipelineConfig directly — use load_config() so that
# api_key is resolved from the environment and the config is fully validated.
class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    pipeline_name: str
    random_seed: int
    api: APIConfig
    step_a: StepAConfig
    step_b: StepBConfig
    step_c: StepCConfig
    evaluation: EvaluationConfig
    # Injected by load_config() from the env var named in api.api_key_env.
    api_key: str | None = None

    @model_validator(mode="after")
    def _check_retrieval_mode(self) -> PipelineConfig:
        valid = {"rrf", "bm25_only", "dense_only"}
        if self.step_b.retrieval_mode not in valid:
            raise ValueError(
                f"step_b.retrieval_mode must be one of {valid}, "
                f"got '{self.step_b.retrieval_mode}'"
            )
        return self

    def assert_step_c_model_set(self) -> None:
        """Call this at the start of Step C to catch null model_name early."""
        if self.step_c.model_name is None:
            raise ValueError(
                "step_c.model_name is None — complete the Model Comparison Study "
                "and set the winning model in configs/base.yaml (or an override) "
                "before running Step C."
            )


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base; override wins on conflicts."""
    merged = dict(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def load_config(base_path: str, override_path: str | None = None) -> PipelineConfig:
    """Load base YAML config, optionally deep-merge an ablation override, validate."""
    with open(base_path) as f:
        data = yaml.safe_load(f)

    if override_path is not None:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        data = _deep_merge(data, override)

    cfg = PipelineConfig.model_validate(data)

    api_key = os.environ.get(cfg.api.api_key_env)
    if api_key is None:
        raise EnvironmentError(
            f"Environment variable '{cfg.api.api_key_env}' is not set. "
            "Export it before running the pipeline."
        )

    # model_copy(update=...) creates a new instance, bypassing frozen=True correctly.
    return cfg.model_copy(update={"api_key": api_key})
