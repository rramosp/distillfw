import React, { useState, useEffect } from 'react';
import { 
  Settings2, Save, Sparkles, Brain, Cpu, 
  Layers, HardDrive, Award, Rocket, Check, AlertCircle 
} from 'lucide-react';
import { fetchConfig, saveConfig } from '../api';

const TARGET_MODULE_OPTIONS = [
  'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
];

const METRIC_OPTIONS = [
  'rouge', 'bleu', 'exact_match', 'gemini_judge', 'latency'
];

const RUBRIC_OPTIONS = [
  'correctness', 'instruction_following', 'coherence', 'similarity'
];

export default function ConfigForm({ bucket, projectId, onSaved }) {
  const [config, setConfig] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  useEffect(() => {
    let isMounted = true;
    const load = async () => {
      setLoading(true);
      setErrorMsg(null);
      try {
        const data = await fetchConfig(bucket, projectId);
        if (isMounted) setConfig(data);
      } catch (err) {
        if (isMounted) setErrorMsg(`Failed to load config: ${err.message}`);
      } finally {
        if (isMounted) setLoading(false);
      }
    };
    if (projectId) load();
    return () => { isMounted = false; };
  }, [bucket, projectId]);

  const updateSection = (section, field, value) => {
    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [field]: value
      }
    }));
    setSaveSuccess(false);
  };

  const updateNested = (section, subSection, field, value) => {
    setConfig((prev) => ({
      ...prev,
      [section]: {
        ...prev[section],
        [subSection]: {
          ...prev[section][subSection],
          [field]: value
        }
      }
    }));
    setSaveSuccess(false);
  };

  const handleSave = async (e) => {
    if (e) e.preventDefault();
    setSaving(true);
    setErrorMsg(null);
    try {
      await saveConfig(bucket, projectId, config);
      setSaveSuccess(true);
      setTimeout(() => setSaveSuccess(false), 3000);
      if (onSaved) onSaved();
    } catch (err) {
      setErrorMsg(`Save failed: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center p-12 text-slate-400">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mr-3"></div>
        Loading configuration controls...
      </div>
    );
  }

  if (!config) return null;

  return (
    <form onSubmit={handleSave} className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Top sticky save header */}
      <div className="flex items-center justify-between bg-slate-800/90 backdrop-blur-md p-4 rounded-xl border border-slate-700 sticky top-20 z-20 shadow-xl">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Settings2 className="w-5 h-5 text-cyan-400" />
            Master Configuration (config.yaml)
          </h2>
          <p className="text-xs text-slate-400">
            Edit parameters through native controls. Changes update <span className="font-mono text-cyan-300">gs://{bucket}/{projectId}/config.yaml</span>
          </p>
        </div>

        <div className="flex items-center gap-3">
          {saveSuccess && (
            <span className="flex items-center gap-1 text-xs text-emerald-400 font-semibold animate-fade-in">
              <Check className="w-4 h-4" /> Saved to GCS!
            </span>
          )}
          {errorMsg && (
            <span className="flex items-center gap-1 text-xs text-rose-400 font-semibold">
              <AlertCircle className="w-4 h-4" /> {errorMsg}
            </span>
          )}
          <button
            type="submit"
            disabled={saving}
            className="flex items-center gap-2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-blue-500/25 transition disabled:opacity-50 cursor-pointer"
          >
            <Save className="w-4 h-4" />
            {saving ? 'Saving...' : 'Save Configuration'}
          </button>
        </div>
      </div>

      {/* 1. Project Info Card */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <Layers className="w-4 h-4 text-blue-400" /> 1. Project Identification
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Project ID</label>
            <input
              type="text"
              value={config.project?.id || ''}
              onChange={(e) => updateSection('project', 'id', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div className="md:col-span-2">
            <label className="block text-xs font-medium text-slate-400 mb-1">Description</label>
            <input
              type="text"
              value={config.project?.description || ''}
              onChange={(e) => updateSection('project', 'description', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* 2. Models Card */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-5">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <Brain className="w-4 h-4 text-purple-400" /> 2. Models (Teacher & Student)
        </h3>
        
        {/* Teacher Model */}
        <div className="bg-slate-900/60 border border-slate-700/60 rounded-lg p-4 space-y-3">
          <div className="text-xs font-bold text-purple-300 uppercase tracking-wider">Teacher Model (Vertex AI Gemini)</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Teacher Model Name</label>
              <select
                value={config.models?.teacher?.model_name || 'gemini-2.5-pro'}
                onChange={(e) => updateNested('models', 'teacher', 'model_name', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="gemini-2.5-pro">gemini-2.5-pro (Recommended)</option>
                <option value="gemini-2.5-flash">gemini-2.5-flash</option>
                <option value="gemini-1.5-pro">gemini-1.5-pro</option>
                <option value="gemini-1.5-flash">gemini-1.5-flash</option>
              </select>
            </div>
            <div>
              <div className="flex justify-between text-xs font-medium text-slate-400 mb-1">
                <span>Temperature</span>
                <span className="text-cyan-400 font-mono">{config.models?.teacher?.temperature ?? 0.2}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="1.0"
                step="0.05"
                value={config.models?.teacher?.temperature ?? 0.2}
                onChange={(e) => updateNested('models', 'teacher', 'temperature', parseFloat(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Max Output Tokens</label>
              <input
                type="number"
                value={config.models?.teacher?.max_output_tokens || 4096}
                onChange={(e) => updateNested('models', 'teacher', 'max_output_tokens', parseInt(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-6 pt-1">
            <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
              <input
                type="checkbox"
                checked={config.models?.teacher?.include_thinking ?? true}
                onChange={(e) => updateNested('models', 'teacher', 'include_thinking', e.target.checked)}
                className="rounded bg-slate-900 border-slate-700 text-blue-600 focus:ring-0"
              />
              <span>Extract Thinking Trace (<span className="font-mono text-cyan-300">teacher_thinking</span>)</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer text-xs text-slate-300">
              <input
                type="checkbox"
                checked={config.models?.teacher?.response_logprobs ?? false}
                onChange={(e) => updateNested('models', 'teacher', 'response_logprobs', e.target.checked)}
                className="rounded bg-slate-900 border-slate-700 text-blue-600 focus:ring-0"
              />
              <span>Extract Response Logprobs (Top-5 soft KD)</span>
            </label>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 pt-3 border-t border-slate-800/80">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Inference Threads (<span className="font-mono text-cyan-300">number_inference_threads</span>)
              </label>
              <input
                type="number"
                min="1"
                step="1"
                value={config.models?.teacher?.number_inference_threads ?? 1}
                onChange={(e) => updateNested('models', 'teacher', 'number_inference_threads', Math.max(1, parseInt(e.target.value) || 1))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              />
              <span className="text-[10px] text-slate-500">1 = Sequential (no parallelism)</span>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                429 Retry Min Delay (s)
              </label>
              <input
                type="number"
                min="0"
                step="0.5"
                value={config.models?.teacher?.retry_delay_min ?? 1.0}
                onChange={(e) => updateNested('models', 'teacher', 'retry_delay_min', Math.max(0, parseFloat(e.target.value) || 0))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              />
              <span className="text-[10px] text-slate-500">Default: 1.0s</span>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                429 Retry Max Delay (s)
              </label>
              <input
                type="number"
                min="0"
                step="0.5"
                value={config.models?.teacher?.retry_delay_max ?? 10.0}
                onChange={(e) => updateNested('models', 'teacher', 'retry_delay_max', Math.max(0, parseFloat(e.target.value) || 0))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              />
              <span className="text-[10px] text-slate-500">Default: 10.0s</span>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Max Retries
              </label>
              <input
                type="number"
                min="1"
                step="1"
                value={config.models?.teacher?.max_retries ?? 5}
                onChange={(e) => updateNested('models', 'teacher', 'max_retries', Math.max(1, parseInt(e.target.value) || 1))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              />
              <span className="text-[10px] text-slate-500">Default: 5 retries</span>
            </div>
          </div>
        </div>

        {/* Student Model */}
        <div className="bg-slate-900/60 border border-slate-700/60 rounded-lg p-4 space-y-3">
          <div className="text-xs font-bold text-cyan-300 uppercase tracking-wider">Student Model (Hugging Face / Model Garden)</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            <div className="sm:col-span-2">
              <label className="block text-xs font-medium text-slate-400 mb-1">Student Model Name or Path</label>
              <input
                type="text"
                value={config.models?.student?.model_name_or_path || 'google/gemma-2-9b'}
                onChange={(e) => updateNested('models', 'student', 'model_name_or_path', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
              />
              <div className="flex flex-wrap gap-1.5 mt-2">
                {['google/gemma-2-9b', 'google/gemma-2-2b', 'meta-llama/Llama-3.2-3B', 'meta-llama/Meta-Llama-3.1-8B'].map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => updateNested('models', 'student', 'model_name_or_path', m)}
                    className="text-[10px] bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-300 px-2 py-0.5 rounded font-mono"
                  >
                    {m}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Quantization</label>
              <select
                value={config.models?.student?.quantization || '4bit'}
                onChange={(e) => updateNested('models', 'student', 'quantization', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
              >
                <option value="4bit">4bit (QLoRA - Recommended)</option>
                <option value="8bit">8bit</option>
                <option value="none">None (FP16 / BF16)</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* 3. Prompt Template */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <Sparkles className="w-4 h-4 text-amber-400" /> 3. Prompt Configuration & Template
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">System Instructions</label>
            <textarea
              rows={3}
              value={config.prompt?.instructions || ''}
              onChange={(e) => updateSection('prompt', 'instructions', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs text-white focus:border-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Prompt Template (Supports {'{instructions}'} & {'{prompt}'})</label>
            <textarea
              rows={3}
              value={config.prompt?.template || ''}
              onChange={(e) => updateSection('prompt', 'template', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-xs text-white font-mono focus:border-blue-500 focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* 4. Dataset & Splitting */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <HardDrive className="w-4 h-4 text-emerald-400" /> 4. Dataset & Splitting Ratios
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Train Ratio ({Math.round((config.dataset?.split_ratios?.train ?? 0.8) * 100)}%)</label>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.9"
              value={config.dataset?.split_ratios?.train ?? 0.8}
              onChange={(e) => updateNested('dataset', 'split_ratios', 'train', parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Val Ratio ({Math.round((config.dataset?.split_ratios?.val ?? 0.1) * 100)}%)</label>
            <input
              type="number"
              step="0.05"
              min="0.0"
              max="0.5"
              value={config.dataset?.split_ratios?.val ?? 0.1}
              onChange={(e) => updateNested('dataset', 'split_ratios', 'val', parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Test Ratio ({Math.round((config.dataset?.split_ratios?.test ?? 0.1) * 100)}%)</label>
            <input
              type="number"
              step="0.05"
              min="0.0"
              max="0.5"
              value={config.dataset?.split_ratios?.test ?? 0.1}
              onChange={(e) => updateNested('dataset', 'split_ratios', 'test', parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Random Seed</label>
            <input
              type="number"
              value={config.dataset?.random_seed ?? 42}
              onChange={(e) => updateSection('dataset', 'random_seed', parseInt(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
        </div>
      </div>

      {/* 5. Distillation Formulation & PEFT */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <Brain className="w-4 h-4 text-indigo-400" /> 5. Distillation Algorithm & PEFT LoRA
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Distillation Method</label>
            <select
              value={config.distillation?.method || 'cot_distillation'}
              onChange={(e) => updateSection('distillation', 'method', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
            >
              <option value="cot_distillation">Method 2: Step-by-Step CoT (Distilling CoT)</option>
              <option value="seq_kd">Method 1: Sequence-Level KD (SeqKD)</option>
              <option value="on_policy_gkd">Method 3: Generalized KD (GKD) On-Policy</option>
              <option value="topk_kd">Method 4: Top-K Soft Target KD</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Loss Objective</label>
            <select
              value={config.distillation?.loss_type || 'cot_weighted'}
              onChange={(e) => updateSection('distillation', 'loss_type', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
            >
              <option value="cot_weighted">cot_weighted (Thinking + Response)</option>
              <option value="ce">ce (Standard Cross-Entropy)</option>
              <option value="kl_divergence">kl_divergence (Soft Logits)</option>
              <option value="dpo">dpo (Direct Preference Optimization)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Thinking Weight (λ_think)</label>
            <input
              type="number"
              step="0.1"
              value={config.distillation?.cot_weights?.thinking_weight ?? 0.5}
              onChange={(e) => updateNested('distillation', 'cot_weights', 'thinking_weight', parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Response Weight (λ_resp)</label>
            <input
              type="number"
              step="0.1"
              value={config.distillation?.cot_weights?.response_weight ?? 1.0}
              onChange={(e) => updateNested('distillation', 'cot_weights', 'response_weight', parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
        </div>

        {/* LoRA Parameters */}
        <div className="bg-slate-900/40 p-3 rounded-lg border border-slate-800 space-y-3">
          <div className="text-xs font-bold text-slate-300">LoRA Adapter Hyperparameters</div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-[11px] font-medium text-slate-400 mb-1">Rank (r)</label>
              <input
                type="number"
                value={config.distillation?.peft?.r ?? 16}
                onChange={(e) => updateNested('distillation', 'peft', 'r', parseInt(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-slate-400 mb-1">LoRA Alpha</label>
              <input
                type="number"
                value={config.distillation?.peft?.lora_alpha ?? 32}
                onChange={(e) => updateNested('distillation', 'peft', 'lora_alpha', parseInt(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white"
              />
            </div>
            <div>
              <label className="block text-[11px] font-medium text-slate-400 mb-1">Dropout</label>
              <input
                type="number"
                step="0.01"
                value={config.distillation?.peft?.lora_dropout ?? 0.05}
                onChange={(e) => updateNested('distillation', 'peft', 'lora_dropout', parseFloat(e.target.value))}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white"
              />
            </div>
          </div>
        </div>
      </div>

      {/* 6. Training Hardware & Hyperparameters */}
      <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
          <Cpu className="w-4 h-4 text-cyan-400" /> 6. Training Hardware & Hyperparameters
        </h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Accelerator Type</label>
            <select
              value={config.training?.hardware?.accelerator_type || 'NVIDIA_L4'}
              onChange={(e) => updateNested('training', 'hardware', 'accelerator_type', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            >
              <option value="NVIDIA_L4">NVIDIA L4 (24GB VRAM - Recommended)</option>
              <option value="NVIDIA_A100_80GB">NVIDIA A100 (80GB VRAM)</option>
              <option value="NVIDIA_H100_80GB">NVIDIA H100 (80GB VRAM)</option>
              <option value="NVIDIA_TESLA_T4">NVIDIA T4 (16GB VRAM)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Accelerator Count</label>
            <input
              type="number"
              min="1"
              max="8"
              value={config.training?.hardware?.accelerator_count ?? 1}
              onChange={(e) => updateNested('training', 'hardware', 'accelerator_count', parseInt(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Machine Type</label>
            <input
              type="text"
              value={config.training?.hardware?.machine_type || 'g2-standard-8'}
              onChange={(e) => updateNested('training', 'hardware', 'machine_type', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 pt-2">
          <div>
            <label className="block text-[11px] font-medium text-slate-400 mb-1">Learning Rate</label>
            <input
              type="number"
              step="0.00001"
              value={config.training?.hyperparameters?.learning_rate ?? 0.0002}
              onChange={(e) => updateNested('training', 'hyperparameters', 'learning_rate', parseFloat(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono"
            />
          </div>
          <div>
            <label className="block text-[11px] font-medium text-slate-400 mb-1">Batch Size</label>
            <input
              type="number"
              value={config.training?.hyperparameters?.batch_size ?? 4}
              onChange={(e) => updateNested('training', 'hyperparameters', 'batch_size', parseInt(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-[11px] font-medium text-slate-400 mb-1">Grad Accum</label>
            <input
              type="number"
              value={config.training?.hyperparameters?.gradient_accumulation_steps ?? 4}
              onChange={(e) => updateNested('training', 'hyperparameters', 'gradient_accumulation_steps', parseInt(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-[11px] font-medium text-slate-400 mb-1">Epochs</label>
            <input
              type="number"
              value={config.training?.hyperparameters?.num_train_epochs ?? 3}
              onChange={(e) => updateNested('training', 'hyperparameters', 'num_train_epochs', parseInt(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white"
            />
          </div>
          <div>
            <label className="block text-[11px] font-medium text-slate-400 mb-1">Max Seq Len</label>
            <input
              type="number"
              value={config.training?.hyperparameters?.max_seq_length ?? 2048}
              onChange={(e) => updateNested('training', 'hyperparameters', 'max_seq_length', parseInt(e.target.value))}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white font-mono"
            />
          </div>
        </div>
      </div>

      {/* 7. Evaluation & Deployment */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Evaluation Card */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
            <Award className="w-4 h-4 text-orange-400" /> 7. Evaluation Configuration
          </h3>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1">Gemini Judge Model</label>
            <select
              value={config.evaluation?.gemini_judge?.model_name || 'gemini-2.5-flash'}
              onChange={(e) => updateNested('evaluation', 'gemini_judge', 'model_name', e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
            >
              <option value="gemini-2.5-flash">gemini-2.5-flash (Fast & Cost-Efficient)</option>
              <option value="gemini-2.5-pro">gemini-2.5-pro (Deepest Assessment)</option>
            </select>
          </div>
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-2">Enabled Metrics</label>
            <div className="flex flex-wrap gap-2">
              {METRIC_OPTIONS.map((m) => {
                const isChecked = (config.evaluation?.metrics || []).includes(m);
                return (
                  <button
                    key={m}
                    type="button"
                    onClick={() => {
                      const curr = config.evaluation?.metrics || [];
                      const next = isChecked ? curr.filter((x) => x !== m) : [...curr, m];
                      updateSection('evaluation', 'metrics', next);
                    }}
                    className={`text-xs px-2.5 py-1 rounded border ${isChecked ? 'bg-blue-600/30 border-blue-500 text-cyan-300 font-semibold' : 'bg-slate-900 border-slate-700 text-slate-400'}`}
                  >
                    {isChecked ? '✓ ' : '+ '}{m}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Deployment Card */}
        <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
            <Rocket className="w-4 h-4 text-green-400" /> 8. Production Deployment (vLLM)
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Serving Framework</label>
              <select
                value={config.deployment?.serving_framework || 'vllm'}
                onChange={(e) => updateSection('deployment', 'serving_framework', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none"
              >
                <option value="vllm">vLLM (High-Throughput PagedAttention)</option>
              </select>
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Machine Type</label>
              <input
                type="text"
                value={config.deployment?.machine_type || 'g2-standard-4'}
                onChange={(e) => updateSection('deployment', 'machine_type', e.target.value)}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white font-mono focus:outline-none"
              />
            </div>
          </div>
          <div className="flex items-center justify-between pt-2">
            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={config.deployment?.merge_lora_weights ?? true}
                onChange={(e) => updateSection('deployment', 'merge_lora_weights', e.target.checked)}
                className="rounded bg-slate-900 border-slate-700 text-green-600 focus:ring-0"
              />
              <span>Merge LoRA Weights into Standalone Model</span>
            </label>
            <span className="text-[11px] text-slate-400">Min: {config.deployment?.min_replicas ?? 0} / Max: {config.deployment?.max_replicas ?? 2} Replicas</span>
          </div>
        </div>
      </div>
    </form>
  );
}
