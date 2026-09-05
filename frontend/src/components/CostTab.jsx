import React, { useState, useEffect } from 'react';
import { Calculator, Play, CheckCircle2, AlertTriangle, DollarSign, Clock, Cpu, Zap } from 'lucide-react';
import { runCostProbe, fetchCostEstimate } from '../api';

export default function CostTab({ bucket, projectId, onStatusChange }) {
  const [estimate, setEstimate] = useState(null);
  const [loading, setLoading] = useState(true);
  const [probing, setProbing] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const loadEstimate = async () => {
    setLoading(true);
    try {
      const data = await fetchCostEstimate(bucket, projectId);
      setEstimate(data);
    } catch (err) {
      // not estimated yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) loadEstimate();
  }, [bucket, projectId]);

  const handleRunProbe = async () => {
    setProbing(true);
    setErrorMsg(null);
    try {
      const res = await runCostProbe(bucket, projectId);
      setEstimate(res);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Probe failed: ${err.message}`);
    } finally {
      setProbing(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Calculator className="w-5 h-5 text-amber-400" />
            Cost Estimation & Hardware Calibration Probe
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Validates GPU VRAM footprint (OOM check) and provides an exact, transparent cost forecast before launching Vertex AI training.
          </p>
        </div>

        <button
          onClick={handleRunProbe}
          disabled={probing}
          className="flex items-center gap-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-amber-500/20 disabled:opacity-50"
        >
          <Play className="w-4 h-4" />
          {probing ? 'Probing Hardware Profile...' : 'Run Cost & Hardware Probe'}
        </button>
      </div>

      {errorMsg && (
        <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-lg text-rose-300 text-xs flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {estimate ? (
        <div className="space-y-6">
          {/* Top Total Scorecard Banner */}
          <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border border-amber-500/30 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            <div>
              <div className="text-xs font-bold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                <DollarSign className="w-4 h-4" /> Total Projected Experiment Cost
              </div>
              <div className="text-4xl font-black text-white mt-1">
                ${estimate.summary?.total_experiment_cost_usd?.toFixed(2)} <span className="text-sm font-normal text-slate-400">USD</span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Combines Teacher Knowledge Extraction + Vertex AI PEFT Training Compute.
              </p>
            </div>

            <div className="flex items-center gap-3">
              {estimate.hardware_probe?.oom_risk ? (
                <div className="flex items-center gap-2 bg-rose-950/60 border border-rose-700 px-4 py-2.5 rounded-xl text-xs text-rose-300">
                  <AlertTriangle className="w-5 h-5 text-rose-400 shrink-0" />
                  <div>
                    <div className="font-bold">OOM Risk Detected</div>
                    <div className="text-[11px] text-rose-400/80">Peak VRAM exceeds accelerator limit!</div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-2 bg-emerald-950/60 border border-emerald-700 px-4 py-2.5 rounded-xl text-xs text-emerald-300">
                  <CheckCircle2 className="w-5 h-5 text-emerald-400 shrink-0" />
                  <div>
                    <div className="font-bold">Hardware Sign-off Verified</div>
                    <div className="text-[11px] text-emerald-400/80">Peak VRAM safely fits within {estimate.hardware_probe?.vram_limit_gb}GB VRAM</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Breakdown Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Part 1: Teacher Inference Cost */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
                <Zap className="w-4 h-4 text-purple-400" /> Part 1: Teacher Model Inference
              </h3>

              <div className="space-y-3 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Model Name</span>
                  <span className="font-mono text-purple-300 font-semibold">{estimate.teacher_inference?.model_name}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Dataset Prompts</span>
                  <span className="font-mono text-white">{estimate.teacher_inference?.samples_count} prompts</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Total Input Tokens</span>
                  <span className="font-mono text-slate-300">{estimate.teacher_inference?.prompt_tokens?.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Total Output Tokens (CoT + Response)</span>
                  <span className="font-mono text-slate-300">{estimate.teacher_inference?.completion_tokens?.toLocaleString()} tokens</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Input Cost (${estimate.teacher_inference?.input_rate_per_million}/M)</span>
                  <span className="font-mono text-slate-300">${estimate.teacher_inference?.cost_input_usd?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Output Cost (${estimate.teacher_inference?.output_rate_per_million}/M)</span>
                  <span className="font-mono text-slate-300">${estimate.teacher_inference?.cost_output_usd?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between pt-2 text-sm font-bold">
                  <span className="text-white">Subtotal Teacher Inference</span>
                  <span className="text-purple-400 font-mono">${estimate.teacher_inference?.total_teacher_cost_usd?.toFixed(4)}</span>
                </div>
              </div>
            </div>

            {/* Part 2: Training Hardware Cost Probe */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
              <h3 className="text-sm font-bold text-slate-200 uppercase tracking-wider flex items-center gap-2 border-b border-slate-700/60 pb-2">
                <Cpu className="w-4 h-4 text-cyan-400" /> Part 2: Training Hardware Calibration Probe
              </h3>

              <div className="space-y-3 text-xs">
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Target GPU & Machine</span>
                  <span className="font-mono text-cyan-300 font-semibold">{estimate.hardware_probe?.accelerator_type} ({estimate.hardware_probe?.machine_type})</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Hardware Hourly Rate</span>
                  <span className="font-mono text-slate-300">${estimate.hardware_probe?.hourly_rate_usd}/hr</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Peak VRAM Footprint</span>
                  <span className="font-mono text-white font-bold">{estimate.hardware_probe?.peak_vram_gb} GB / {estimate.hardware_probe?.vram_limit_gb} GB</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Measured Step Duration (T_step)</span>
                  <span className="font-mono text-slate-300">{estimate.hardware_probe?.avg_step_duration_seconds}s per step</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Total Training Steps</span>
                  <span className="font-mono text-slate-300">{estimate.hardware_probe?.total_training_steps} steps</span>
                </div>
                <div className="flex justify-between py-1.5 border-b border-slate-800">
                  <span className="text-slate-400">Estimated Duration (Init + Steps)</span>
                  <span className="font-mono text-slate-300">{estimate.hardware_probe?.estimated_training_hours} hrs ({Math.round(estimate.hardware_probe?.estimated_training_seconds / 60)} mins)</span>
                </div>
                <div className="flex justify-between pt-2 text-sm font-bold">
                  <span className="text-white">Subtotal Training Compute</span>
                  <span className="text-cyan-400 font-mono">${estimate.hardware_probe?.total_training_cost_usd?.toFixed(4)}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Calculator className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">Cost & Hardware Profile Not Probed</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Launch the hardware calibration probe to calculate exact step duration, test VRAM safety, and obtain the cost forecast.
          </p>
          <button
            onClick={handleRunProbe}
            disabled={probing}
            className="bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold px-4 py-2 rounded-lg"
          >
            Run Calibration Probe
          </button>
        </div>
      )}
    </div>
  );
}
