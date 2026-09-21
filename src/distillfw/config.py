"""Configuration schemas for the distillfw Gemini-to-Gemma distillation pipeline."""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class TeacherModel(str, Enum):
    """Supported Gemini teacher models."""

    GEMINI_3_5_PRO = "gemini-3.5-pro"
    GEMINI_3_5_FLASH = "gemini-3.5-flash"
    GEMINI_3_5_FLASH_LITE = "gemini-3.5-flash-lite"
    GEMINI_2_0_FLASH = "gemini-2.0-flash"
    GEMINI_2_0_FLASH_LITE = "gemini-2.0-flash-lite"


class StudentModel(str, Enum):
    """Supported Gemma student models."""

    GEMMA_3_1B_PT = "google/gemma-3-1b-pt"
    GEMMA_3_1B_IT = "google/gemma-3-1b-it"
    GEMMA_3_4B_PT = "google/gemma-3-4b-pt"
    GEMMA_3_4B_IT = "google/gemma-3-4b-it"
    GEMMA_3_12B_PT = "google/gemma-3-12b-pt"
    GEMMA_3_12B_IT = "google/gemma-3-12b-it"
    GEMMA_3_27B_PT = "google/gemma-3-27b-pt"
    GEMMA_3_27B_IT = "google/gemma-3-27b-it"
    GEMMA_2_2B_PT = "google/gemma-2-2b"
    GEMMA_2_2B_IT = "google/gemma-2-2b-it"
    GEMMA_2_9B_PT = "google/gemma-2-9b"
    GEMMA_2_9B_IT = "google/gemma-2-9b-it"
    GEMMA_2_27B_PT = "google/gemma-2-27b"
    GEMMA_2_27B_IT = "google/gemma-2-27b-it"


class PromptFormat(str, Enum):
    """Supported prompt formatting modes."""

    PRETRAIN = "pretrain"
    INSTRUCTION = "instruction"
    CHAT = "chat"
    REASONING = "reasoning"


class DatasetRepresentation(str, Enum):
    """Supported training dataset representations."""

    TEXT = "text"
    TOKENS = "tokens"
    SPARSE_LOGPROBS = "sparse_logprobs"
    PREFERENCE = "preference"


class DistillationParadigm(str, Enum):
    """Supported distillation data collection paradigms."""

    OFF_POLICY = "off_policy"
    ON_POLICY = "on_policy"
    HYBRID = "hybrid"


class TrainingAlgorithm(str, Enum):
    """Supported SOTA distillation & alignment algorithms."""

    SFT_SEQKD = "sft_seqkd"
    FORWARD_KL = "forward_kl"
    REVERSE_KL = "reverse_kl"
    JSD = "jsd"
    SKEW_KL = "skew_kl"
    DISTILLM2 = "distillm2"
    DPO = "dpo"
    SIMPO = "simpo"
    ORPO = "orpo"
    GKD = "gkd"
    GRPO = "grpo"


LOGPROBS_REQUIRED_ALGORITHMS: frozenset[TrainingAlgorithm] = frozenset(
    {
        TrainingAlgorithm.FORWARD_KL,
        TrainingAlgorithm.REVERSE_KL,
        TrainingAlgorithm.JSD,
        TrainingAlgorithm.SKEW_KL,
        TrainingAlgorithm.DISTILLM2,
        TrainingAlgorithm.GKD,
    }
)


class EvaluationMetric(str, Enum):
    """Supported evaluation metrics."""

    BLEU = "bleu"
    ROUGE = "rouge"
    EXACT_MATCH = "exact_match"
    PERPLEXITY = "perplexity"
    KL_DIVERGENCE = "kl_divergence"
    JSON_SCHEMA_VALIDITY = "json_schema_validity"
    LLM_JUDGE = "llm_judge"
    LATENCY = "latency"


class GCPConfig(BaseModel):
    """GCP infrastructure configuration."""

    project_id: str = Field(..., description="GCP Project ID")
    location: str = Field(default="us-central1", description="GCP region for Vertex AI & GCS")
    bucket_name: str = Field(..., description="GCS bucket name for task tracking and artifacts")
    tasks_prefix: str = Field(
        default="tasks", description="Root prefix inside bucket for isolated task workspaces"
    )
    staging_bucket: str | None = Field(
        default=None, description="Optional staging bucket URI for Vertex AI Custom Jobs"
    )

    def get_task_uri(self, task_id: str) -> str:
        prefix = self.tasks_prefix.strip("/")
        return f"gs://{self.bucket_name}/{prefix}/{task_id}"


class LocalConfig(BaseModel):
    """Local filesystem storage configuration."""

    storage_root: str = Field(
        ..., description="Root folder where tasks are stored locally"
    )

    def get_task_uri(self, task_id: str) -> str:
        root = self.storage_root.rstrip("/")
        return f"{root}/{task_id}"


class TeacherConfig(BaseModel):
    """Configuration for querying the Gemini teacher model (Stage 1)."""

    model_id: str = Field(
        default=TeacherModel.GEMINI_3_5_FLASH.value,
        description="Gemini model identifier on Vertex AI / Gemini API",
    )
    location: str | None = Field(
        default=None,
        description="Optional Vertex AI location for teacher inference (e.g., 'global'). Defaults to gcp.location if unset.",
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=2048, gt=0)
    candidate_count: int = Field(
        default=1, ge=1, description="Number of candidates per prompt (for Best-of-N / DPO)"
    )
    response_logprobs: bool = Field(
        default=False, description="Whether to request token-level logprobs from Gemini"
    )
    logprobs_top_k: int | None = Field(
        default=None, ge=1, le=20, description="Number of top-k token logprobs to record per step"
    )
    thinking_budget: int | None = Field(
        default=None, description="Optional thinking token budget for Gemini 3.5 reasoning traces"
    )
    system_instruction: str | None = Field(
        default=None, description="Optional system instruction prepended to teacher queries"
    )
    use_batch_prediction: bool = Field(
        default=False,
        description="Use Vertex AI Batch Prediction instead of async online generation",
    )
    concurrency: int = Field(default=16, ge=1, description="Max concurrent online API requests")
    shard_size: int = Field(
        default=250, ge=1, description="Number of prompts per persisted GCS shard for resumption"
    )
    max_retries: int = Field(
        default=12,
        ge=10,
        description="Number of retries per Gemini API request on transient / 429 RESOURCE_EXHAUSTED errors (minimum 10)",
    )
    initial_retry_delay_seconds: float = Field(
        default=2.0,
        gt=0.0,
        description="Initial wait time in seconds before the first retry",
    )
    max_retry_delay_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="Maximum base wait time in seconds between retries",
    )
    backoff_multiplier: float = Field(
        default=1.8,
        gt=1.0,
        description="Exponential backoff multiplier between consecutive retries",
    )


class StudentConfig(BaseModel):
    """Configuration for the Gemma student model (Stages 2–5)."""

    model_id: str = Field(
        default=StudentModel.GEMMA_3_4B_IT.value,
        description="Gemma model identifier (Vertex AI Model Garden or Hugging Face ID)",
    )
    source: Literal["model_garden", "huggingface"] = Field(default="huggingface")
    peft_method: Literal["none", "lora", "qlora_4bit", "qlora_8bit"] = Field(default="lora")
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    max_seq_length: int = Field(default=2048, gt=0)
    dtype: Literal["bfloat16", "float16", "float32"] = Field(default="bfloat16")


class FormatConfig(BaseModel):
    """Configuration for dataset formatting & tokenization (Stage 2)."""

    prompt_format: PromptFormat = Field(default=PromptFormat.CHAT)
    dataset_representation: DatasetRepresentation = Field(default=DatasetRepresentation.TEXT)
    train_split_ratio: float = Field(default=0.80, gt=0.0, lt=1.0)
    val_split_ratio: float = Field(default=0.10, ge=0.0, lt=1.0)
    test_split_ratio: float = Field(default=0.10, gt=0.0, lt=1.0)
    seed: int = Field(default=42)

    @model_validator(mode="after")
    def validate_splits(self) -> FormatConfig:
        total = round(self.train_split_ratio + self.val_split_ratio + self.test_split_ratio, 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"train/val/test split ratios must sum to 1.0, got {total}"
            )
        return self



class TrainingConfig(BaseModel):
    """Configuration for single-node student training (Stage 3)."""

    paradigm: DistillationParadigm = Field(default=DistillationParadigm.OFF_POLICY)
    algorithm: TrainingAlgorithm = Field(default=TrainingAlgorithm.SFT_SEQKD)
    execution_mode: Literal["local", "vertex_custom_job"] = Field(default="vertex_custom_job")
    num_epochs: int = Field(default=3, ge=1)
    per_device_batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    temperature: float = Field(default=1.0, gt=0.0, description="KD softmax temperature")
    skew_alpha: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Skew KLD interpolation weight (DistiLLM)"
    )
    gkd_lambda: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Student rollout fraction for Generalized KD (0=off-policy, 1=on-policy)",
    )
    gkd_beta: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Generalized JSD beta parameter for GKD"
    )
    dpo_beta: float = Field(default=0.1, gt=0.0, description="Beta regularization for DPO/ORPO")
    save_steps: int = Field(default=100, ge=1)
    logging_steps: int = Field(default=10, ge=1)
    # Single-node Vertex AI Custom Job hardware specifications
    vertex_machine_type: str = Field(default="g2-standard-48")
    vertex_accelerator_type: str = Field(default="NVIDIA_L4")
    vertex_accelerator_count: int = Field(
        default=4, ge=1, le=8, description="Single-node GPU count (1 to 8)"
    )
    vertex_container_uri: str = Field(
        default="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2.py310:latest"
    )


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation (Stage 4)."""

    metrics: list[EvaluationMetric] = Field(
        default_factory=lambda: [
            EvaluationMetric.ROUGE,
            EvaluationMetric.BLEU,
            EvaluationMetric.EXACT_MATCH,
            EvaluationMetric.LLM_JUDGE,
            EvaluationMetric.LATENCY,
        ]
    )
    judge_model_id: str = Field(default=TeacherModel.GEMINI_3_5_PRO.value)
    judge_model_location: str | None = Field(
        default=None,
        description="Optional Vertex AI location for judge inference (e.g., 'global' or 'us-central1'). Defaults to teacher.location or gcp.location.",
    )
    judge_location: str | None = Field(
        default=None,
        description="Alias for judge_model_location. Defaults to teacher.location or gcp.location.",
    )
    max_eval_samples: int = Field(default=200, ge=1)
    use_vllm: bool = Field(default=False)
    json_schema: dict[str, Any] | None = Field(
        default=None, description="Optional JSON schema for validating structured outputs"
    )
    teacher_cost_per_1m_input: float = Field(default=1.25)
    teacher_cost_per_1m_output: float = Field(default=10.00)
    student_hourly_gpu_cost: float = Field(default=1.20)

    @property
    def resolved_judge_location(self) -> str | None:
        """Return judge_model_location if set, falling back to judge_location."""
        return self.judge_model_location or self.judge_location


class DeploymentConfig(BaseModel):
    """Configuration for Vertex AI Endpoint deployment (Stage 5)."""

    merge_lora: bool = Field(default=True, description="Merge LoRA adapter weights before export")
    serving_container_uri: str = Field(
        default="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:latest"
    )
    machine_type: str = Field(default="g2-standard-12")
    accelerator_type: str = Field(default="NVIDIA_L4")
    accelerator_count: int = Field(default=1, ge=1, le=8)
    min_replica_count: int = Field(default=1, ge=1)
    max_replica_count: int = Field(default=3, ge=1)
    endpoint_display_name: str | None = Field(default=None)
    traffic_percentage: int = Field(default=100, ge=0, le=100)
    vllm_args: list[str] = Field(
        default_factory=lambda: ["--max-model-len=4096", "--gpu-memory-utilization=0.90"]
    )


class DistillationConfig(BaseModel):
    """Top-level configuration driving a self-contained distillation task."""

    task_id: str = Field(..., description="Unique identifier for the distillation task")
    description: str = Field(default="", description="Human-readable description of the task")
    gcp: GCPConfig | None = Field(default=None, description="GCP infrastructure configuration")
    local: LocalConfig | None = Field(
        default=None, description="Local filesystem storage configuration"
    )
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    student: StudentConfig = Field(default_factory=StudentConfig)
    formatting: FormatConfig = Field(default_factory=FormatConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    @model_validator(mode="after")
    def validate_gcp_or_local(self) -> DistillationConfig:
        if (self.gcp is None) == (self.local is None):
            raise ValueError(
                "Exactly one of 'gcp' or 'local' must be specified in config.yaml, not both and not neither."
            )
        return self

    @property
    def task_uri(self) -> str:
        """Canonical task workspace URI (`gs://...` for GCP or `<storage_root>/<task_id>` for local)."""
        if self.gcp is not None:
            return self.gcp.get_task_uri(self.task_id)
        assert self.local is not None
        return self.local.get_task_uri(self.task_id)

    def to_yaml(self) -> str:
        """Serialize configuration to a YAML string."""
        data = self.model_dump(mode="json")
        if data.get("gcp") is None:
            data.pop("gcp", None)
        if data.get("local") is None:
            data.pop("local", None)
        return yaml.safe_dump(data, sort_keys=False)

    @classmethod
    def from_yaml(cls, content: str | Path) -> DistillationConfig:
        """Deserialize configuration from a YAML string or file path."""
        if isinstance(content, Path) or (
            isinstance(content, str) and "\n" not in content and Path(content).exists()
        ):
            raw = Path(content).read_text(encoding="utf-8")
        else:
            raw = str(content)
        data = yaml.safe_load(raw)
        return cls.model_validate(data)

    def validate_logprobs_compatibility(self) -> None:
        """Validate that teacher logprobs settings match the selected training algorithm."""
        algo = self.training.algorithm
        needs_logprobs = algo in LOGPROBS_REQUIRED_ALGORITHMS

        if not needs_logprobs:
            if self.teacher.response_logprobs:
                raise ValueError(
                    f"Initialization error: training.algorithm='{algo.value}' does not use "
                    f"teacher logprobs, so teacher.response_logprobs must be set to false."
                )
        else:
            if not self.teacher.response_logprobs:
                raise ValueError(
                    f"Initialization error: training.algorithm='{algo.value}' requires "
                    f"teacher logprobs, so teacher.response_logprobs must be set to true."
                )
            if self.teacher.logprobs_top_k is None:
                raise ValueError(
                    f"Initialization error: training.algorithm='{algo.value}' requires "
                    f"teacher.logprobs_top_k to be explicitly configured when response_logprobs=true."
                )

    def sha256(self) -> str:
        """Deterministic SHA-256 digest of the configuration."""
        canonical = yaml.safe_dump(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
