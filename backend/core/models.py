"""Pydantic schemas and configuration models for DistillFW."""

from typing import List, Dict, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    CONFIGURED = "CONFIGURED"
    DATASET_READY = "DATASET_READY"
    TEACHER_INFERENCE_RUNNING = "TEACHER_INFERENCE_RUNNING"
    TEACHER_INFERENCE_DONE = "TEACHER_INFERENCE_DONE"
    COST_ESTIMATED = "COST_ESTIMATED"
    TRAINING_RUNNING = "TRAINING_RUNNING"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    EVALUATING = "EVALUATING"
    EVALUATED = "EVALUATED"
    DEPLOYED = "DEPLOYED"


class DistillationMethod(str, Enum):
    SEQ_KD = "seq_kd"
    COT_DISTILLATION = "cot_distillation"
    ON_POLICY_GKD = "on_policy_gkd"
    TOPK_KD = "topk_kd"


class LossType(str, Enum):
    CE = "ce"
    COT_WEIGHTED = "cot_weighted"
    KL_DIVERGENCE = "kl_divergence"
    DPO = "dpo"


class ProjectInfoConfig(BaseModel):
    id: str = "distill-gemma-math-v1"
    description: str = "Distill Gemini 2.5 Pro reasoning into Gemma 2 9B for mathematical problem solving"
    gcs_workspace: str = "gs://distillfw-workspaces/distill-gemma-math-v1"


class TeacherModelConfig(BaseModel):
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    include_thinking: bool = True
    response_logprobs: bool = False
    number_inference_threads: int = Field(default=1, ge=1, description="Number of parallel inference threads (>= 1; 1 means sequential)")
    retry_delay_min: float = Field(default=1.0, ge=0.0, description="Minimum retry delay in seconds on 429 rate limit")
    retry_delay_max: float = Field(default=10.0, ge=0.0, description="Maximum retry delay in seconds on 429 rate limit")
    max_retries: int = Field(default=5, ge=1, description="Maximum retry attempts on 429 rate limit")


class StudentModelConfig(BaseModel):
    model_name_or_path: str = "google/gemma-2-9b"
    quantization: str = "4bit"  # "none", "8bit", "4bit"
    trust_remote_code: bool = False


class ModelsConfig(BaseModel):
    teacher: TeacherModelConfig = Field(default_factory=TeacherModelConfig)
    student: StudentModelConfig = Field(default_factory=StudentModelConfig)


class PromptConfig(BaseModel):
    instructions: str = "You are an expert mathematician. Solve this problem stating the final answer."
    template: str = "{instructions}\n\nProblem:\n{prompt}\n\nSolution:"


class SplitRatiosConfig(BaseModel):
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


class DatasetConfig(BaseModel):
    input_path: str = "data/input_dataset.jsonl"
    auto_split: bool = True
    split_ratios: SplitRatiosConfig = Field(default_factory=SplitRatiosConfig)
    random_seed: int = 42


class CotWeightsConfig(BaseModel):
    thinking_weight: float = 0.5
    response_weight: float = 1.0


class PeftConfig(BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


class DistillationSettingsConfig(BaseModel):
    method: str = "cot_distillation"
    loss_type: str = "cot_weighted"
    cot_weights: CotWeightsConfig = Field(default_factory=CotWeightsConfig)
    peft: PeftConfig = Field(default_factory=PeftConfig)


class HardwareConfig(BaseModel):
    accelerator_type: str = "NVIDIA_L4"
    accelerator_count: int = 1
    machine_type: str = "g2-standard-8"


class HyperparametersConfig(BaseModel):
    learning_rate: float = 2.0e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 3
    warmup_ratio: float = 0.05
    optimizer: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 2048
    logging_steps: int = 5
    eval_steps: int = 50
    save_steps: int = 100


class TrainingConfig(BaseModel):
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    hyperparameters: HyperparametersConfig = Field(default_factory=HyperparametersConfig)


class GeminiJudgeConfig(BaseModel):
    model_name: str = "gemini-2.5-flash"
    rubric: List[str] = Field(
        default_factory=lambda: ["correctness", "instruction_following", "coherence", "similarity"]
    )


class EvaluationConfig(BaseModel):
    batch_size: int = 8
    metrics: List[str] = Field(
        default_factory=lambda: ["rouge", "bleu", "exact_match", "gemini_judge", "latency"]
    )
    gemini_judge: GeminiJudgeConfig = Field(default_factory=GeminiJudgeConfig)


class DeploymentConfig(BaseModel):
    serving_framework: str = "vllm"
    machine_type: str = "g2-standard-4"
    accelerator_type: str = "NVIDIA_L4"
    accelerator_count: int = 1
    min_replicas: int = 0
    max_replicas: int = 2
    merge_lora_weights: bool = True


class MasterConfig(BaseModel):
    project: ProjectInfoConfig = Field(default_factory=ProjectInfoConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    distillation: DistillationSettingsConfig = Field(default_factory=DistillationSettingsConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)


class HistoryEntry(BaseModel):
    id: str
    action: str
    status: str  # SUCCESS, FAILED, RUNNING
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    details: Optional[str] = None


class LogMessage(BaseModel):
    timestamp: str
    level: str  # INFO, WARNING, ERROR, SUCCESS
    source: str
    message: str
