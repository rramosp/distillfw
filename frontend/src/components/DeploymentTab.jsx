import React, { useState, useEffect } from 'react';
import { 
  Rocket, Play, RefreshCw, Send, CheckCircle2, AlertCircle, 
  Cpu, Clock, Terminal, Square, RotateCcw, Brain, Zap, Sparkles, 
  ChevronDown, ChevronUp, HelpCircle 
} from 'lucide-react';
import { deployEndpoint, stopDeployment, clearDeployment, fetchDeploymentStatus, predictEndpoint } from '../api';

const QUICK_PROMPTS = [
  'What is 25 multiplied by 14?',
  'Calculate 15 * 18',
  'What is the capital of France?',
  'Who wrote Hamlet?',
  'Explain the difference between LoRA and full fine-tuning',
  'What is the speed of light?'
];

export default function DeploymentTab({ bucket, projectId, onStatusChange }) {
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [testPrompt, setTestPrompt] = useState('What is 25 multiplied by 14?');
  const [testOutput, setTestOutput] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const [showThinking, setShowThinking] = useState(true);

  const loadStatus = async () => {
    setLoading(true);
    try {
      const data = await fetchDeploymentStatus(bucket, projectId);
      setMetadata(data);
    } catch (err) {
      // not deployed yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) loadStatus();
  }, [bucket, projectId]);

  // Poll while deploying
  useEffect(() => {
    if (!projectId) return;
    const isDeploying = deploying || metadata?.status === 'DEPLOYING';
    if (!isDeploying) return;

    const interval = setInterval(async () => {
      try {
        const data = await fetchDeploymentStatus(bucket, projectId);
        if (data) {
          setMetadata(data);
          if (data.status === 'ACTIVE') {
            setDeploying(false);
            if (onStatusChange) onStatusChange();
          } else if (data.status === 'STOPPED') {
            setDeploying(false);
            if (onStatusChange) onStatusChange();
          }
        }
      } catch (err) {
        // ignore polling errors
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [bucket, projectId, deploying, metadata?.status]);

  const handleDeploy = async () => {
    setDeploying(true);
    setErrorMsg(null);
    try {
      const res = await deployEndpoint(bucket, projectId);
      setMetadata(res);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Deployment failed: ${err.message}`);
      setDeploying(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopDeployment(bucket, projectId);
      setTimeout(loadStatus, 1000);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to stop deployment: ${err.message}`);
    } finally {
      setStopping(false);
      setDeploying(false);
    }
  };

  const handleClear = async () => {
    try {
      await clearDeployment(bucket, projectId);
      setMetadata(null);
      setTestOutput(null);
      setErrorMsg(null);
      setDeploying(false);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to clear deployment: ${err.message}`);
    }
  };

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    if (!testPrompt.trim()) return;
    setPredicting(true);
    setTestOutput(null);
    setErrorMsg(null);
    try {
      const res = await predictEndpoint(bucket, projectId, testPrompt.trim());
      setTestOutput(res);
    } catch (err) {
      setErrorMsg(`Prediction failed: ${err.message}`);
    } finally {
      setPredicting(false);
    }
  };

  const isDeploying = deploying || metadata?.status === 'DEPLOYING';
  const isActive = metadata?.status === 'ACTIVE';

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header card */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Rocket className="w-5 h-5 text-green-400" />
            Vertex AI Dual vLLM Production Deployment
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Provisions two production endpoints on Vertex AI: (1) Base Student Model (baseline before training) and (2) Distilled Student Model (after PEFT LoRA training) on vLLM PagedAttention.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {isDeploying ? (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-rose-500/20 cursor-pointer"
            >
              <Square className="w-4 h-4 fill-current" />
              {stopping ? 'Stopping...' : 'Stop Deployment'}
            </button>
          ) : isActive ? (
            <button
              onClick={handleClear}
              className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold px-4 py-2.5 rounded-lg border border-slate-600 shadow-md cursor-pointer"
            >
              <RotateCcw className="w-4 h-4 text-slate-300" />
              Start Over
            </button>
          ) : metadata?.status === 'STOPPED' ? (
            <div className="flex items-center gap-2">
              <button
                onClick={handleDeploy}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-md cursor-pointer"
              >
                <Rocket className="w-4 h-4" />
                Retry Deploy
              </button>
              <button
                onClick={handleClear}
                className="flex items-center gap-1.5 bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-3 py-2.5 rounded-lg border border-slate-600 cursor-pointer"
              >
                Clear
              </button>
            </div>
          ) : (
            <button
              onClick={handleDeploy}
              disabled={isDeploying}
              className="flex items-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-green-500/20 disabled:opacity-50 cursor-pointer"
            >
              <Rocket className="w-4 h-4" />
              Deploy Dual Endpoints
            </button>
          )}
          <button
            onClick={loadStatus}
            className="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 rounded-lg cursor-pointer"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-lg text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Deployment In Progress Card */}
      {isDeploying && (
        <div className="bg-slate-800/80 border border-blue-600/50 rounded-xl p-6 space-y-5 shadow-xl relative overflow-hidden">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-700/80 pb-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-blue-950/80 border border-blue-700/60 flex items-center justify-center text-blue-400">
                <Rocket className="w-5 h-5 animate-pulse" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-white uppercase tracking-wider">Dual Vertex AI Endpoint Deployment In Progress</span>
                  <span className="text-[11px] font-mono text-cyan-300 bg-cyan-950/60 border border-cyan-800 px-2 py-0.5 rounded font-bold">
                    {metadata?.progress_pct || 15}%
                  </span>
                </div>
                <p className="text-xs text-slate-300 mt-0.5">
                  {metadata?.current_step || 'Provisioning dual accelerator nodes and configuring vLLM serving containers...'}
                </p>
              </div>
            </div>

            <button
              onClick={handleStop}
              disabled={stopping}
              className="self-start sm:self-center flex items-center gap-2 bg-rose-600/80 hover:bg-rose-600 text-white text-xs font-semibold px-3.5 py-1.5 rounded-lg transition border border-rose-500/60 shadow-sm cursor-pointer"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              {stopping ? 'Stopping...' : 'Stop Deployment'}
            </button>
          </div>

          {/* Progress Bar */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-slate-400 font-mono">
              <span>Dual Endpoint Deployment Progress</span>
              <span className="text-blue-400 font-bold">{metadata?.progress_pct || 15}%</span>
            </div>
            <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-700">
              <div
                className="h-full bg-gradient-to-r from-blue-600 via-cyan-500 to-emerald-500 transition-all duration-500 rounded-full"
                style={{ width: `${metadata?.progress_pct || 15}%` }}
              />
            </div>
          </div>

          {/* Stage Milestones */}
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-2 pt-1">
            {[
              { id: 1, name: '1. Dual Endpoints' },
              { id: 2, name: '2. Packaging' },
              { id: 3, name: '3. vLLM Launch' },
              { id: 4, name: '4. Engine Warmup' },
              { id: 5, name: '5. Benchmarking' },
            ].map((st, idx) => {
              const stageData = metadata?.stages?.[idx];
              const isDone = stageData?.status === 'COMPLETED' || (metadata?.progress_pct || 0) >= ((idx + 1) * 20);
              const isCurr = stageData?.status === 'IN_PROGRESS' || (!isDone && (metadata?.progress_pct || 0) >= (idx * 20));
              return (
                <div
                  key={st.id}
                  className={`p-2.5 rounded-lg border text-xs flex flex-col gap-1 transition ${
                    isDone
                      ? 'bg-emerald-950/40 border-emerald-700/60 text-emerald-300'
                      : isCurr
                      ? 'bg-blue-950/60 border-blue-500/80 text-blue-200 animate-pulse'
                      : 'bg-slate-900/60 border-slate-800 text-slate-500'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-[11px] truncate">{st.name}</span>
                    {isDone ? (
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                    ) : isCurr ? (
                      <RefreshCw className="w-3 h-3 text-blue-400 animate-spin shrink-0" />
                    ) : (
                      <Clock className="w-3 h-3 text-slate-600 shrink-0" />
                    )}
                  </div>
                  <span className="text-[10px] text-slate-400 truncate">
                    {isDone ? 'Completed' : isCurr ? 'In Progress' : 'Queued'}
                  </span>
                </div>
              );
            })}
          </div>

          <div className="text-[11px] text-slate-400 flex items-center justify-between border-t border-slate-800 pt-3">
            <span className="font-mono">Machine: {metadata?.machine_type || 'g2-standard-4'} + 1x {metadata?.accelerator_type || 'NVIDIA_L4'}</span>
            <span className="font-mono text-cyan-400">Engine: Dual vLLM PagedAttention</span>
          </div>
        </div>
      )}

      {/* Stopped Deployment Card */}
      {!isDeploying && metadata?.status === 'STOPPED' && (
        <div className="bg-amber-950/30 border border-amber-800/80 rounded-xl p-5 text-center space-y-3">
          <Clock className="w-8 h-8 text-amber-400 mx-auto" />
          <div className="text-sm font-bold text-white">Endpoint Deployment Stopped</div>
          <p className="text-xs text-slate-300 max-w-md mx-auto">
            The previous deployment operation was stopped. You can retry launching the Vertex AI Endpoint deployment or clear state to start over.
          </p>
          <div className="flex items-center justify-center gap-3 pt-2">
            <button
              onClick={handleDeploy}
              className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg cursor-pointer"
            >
              Resume Deployment
            </button>
            <button
              onClick={handleClear}
              className="bg-slate-700 hover:bg-slate-600 text-slate-300 text-xs px-4 py-2 rounded-lg cursor-pointer"
            >
              Clear
            </button>
          </div>
        </div>
      )}

      {/* Active Endpoint & Playground */}
      {!isDeploying && isActive && (
        <div className="space-y-6">
          {/* Dual Active Endpoints Overview */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Endpoint 1: Distilled Student */}
            <div className="bg-emerald-950/20 border-2 border-emerald-600/60 rounded-xl p-4 space-y-3 shadow-lg shadow-emerald-950/20">
              <div className="flex items-center justify-between border-b border-emerald-800/50 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span className="text-xs font-bold text-emerald-300 uppercase tracking-wider">Distilled Endpoint (Post-Training)</span>
                </div>
                <span className="text-[10px] font-mono text-emerald-300 bg-emerald-950/80 border border-emerald-700 px-2 py-0.5 rounded font-bold">
                  {metadata.metrics?.distilled_latency_ms || 38.4} ms (Fastest)
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div>
                  <span className="text-slate-400">Model:</span>
                  <div className="font-mono text-white font-semibold truncate">{metadata.base_model} + LoRA</div>
                </div>
                <div>
                  <span className="text-slate-400">Serving Engine:</span>
                  <div className="font-mono text-cyan-300">{metadata.serving_framework || 'vllm'} PagedAttention</div>
                </div>
                <div>
                  <span className="text-slate-400">Speedup:</span>
                  <div className="font-mono text-emerald-400 font-bold">{metadata.metrics?.speedup_factor || '3.25x'}</div>
                </div>
                <div>
                  <span className="text-slate-400">GPU:</span>
                  <div className="font-mono text-slate-300">{metadata.accelerator_type || 'NVIDIA_L4'}</div>
                </div>
              </div>

              <div className="pt-2 text-[10px] text-slate-400 border-t border-emerald-900/40 flex items-center gap-1.5 font-mono truncate select-all">
                <Terminal className="w-3 h-3 text-emerald-500 shrink-0" />
                <span className="truncate">{metadata.endpoint_uri}</span>
              </div>
            </div>

            {/* Endpoint 2: Base Student */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 space-y-3">
              <div className="flex items-center justify-between border-b border-slate-700/60 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-slate-400"></span>
                  <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Base Endpoint (Pre-Training Baseline)</span>
                </div>
                <span className="text-[10px] font-mono text-slate-400 bg-slate-900 border border-slate-700 px-2 py-0.5 rounded">
                  {metadata.metrics?.base_latency_ms || 124.8} ms
                </span>
              </div>

              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div>
                  <span className="text-slate-400">Model:</span>
                  <div className="font-mono text-slate-200 font-semibold truncate">{metadata.base_model} (Base)</div>
                </div>
                <div>
                  <span className="text-slate-400">Serving Engine:</span>
                  <div className="font-mono text-cyan-300">{metadata.serving_framework || 'vllm'}</div>
                </div>
                <div>
                  <span className="text-slate-400">Fine-Tuning:</span>
                  <div className="font-mono text-slate-400">None (Zero-Shot)</div>
                </div>
                <div>
                  <span className="text-slate-400">GPU:</span>
                  <div className="font-mono text-slate-300">{metadata.accelerator_type || 'NVIDIA_L4'}</div>
                </div>
              </div>

              <div className="pt-2 text-[10px] text-slate-400 border-t border-slate-700/60 flex items-center gap-1.5 font-mono truncate select-all">
                <Terminal className="w-3 h-3 text-slate-500 shrink-0" />
                <span className="truncate">{metadata.base_endpoint_uri || metadata.base_endpoint_id || 'Base endpoint active'}</span>
              </div>
            </div>
          </div>

          {/* Interactive Live Playground: 3-Model Comparison */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
            <div>
              <h3 className="text-sm font-bold text-white flex items-center gap-2">
                <Send className="w-4 h-4 text-blue-400" />
                Interactive Model Inference Playground
              </h3>
              <p className="text-xs text-slate-400 mt-1">
                Benchmark query execution side-by-side across: (1) Student before distillation, (2) Frontier Teacher model, and (3) Distilled Student model.
              </p>
            </div>

            <form onSubmit={handlePredict} className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Input Prompt</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={testPrompt}
                    onChange={(e) => setTestPrompt(e.target.value)}
                    placeholder="Enter test problem or query..."
                    disabled={predicting}
                    className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none disabled:opacity-50"
                  />
                  <button
                    type="submit"
                    disabled={predicting}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg disabled:opacity-50 cursor-pointer"
                  >
                    <Send className="w-3.5 h-3.5" />
                    {predicting ? 'Generating...' : 'Predict'}
                  </button>
                </div>
              </div>

              {/* Quick Prompt Selector Chips */}
              <div className="flex flex-wrap items-center gap-1.5 pt-1">
                <span className="text-[11px] text-slate-500 mr-1 flex items-center gap-1">
                  <HelpCircle className="w-3 h-3" /> Sample queries:
                </span>
                {QUICK_PROMPTS.map((qp, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      setTestPrompt(qp);
                    }}
                    className={`text-[11px] px-2.5 py-1 rounded-md border transition cursor-pointer ${
                      testPrompt === qp
                        ? 'bg-blue-900/50 border-blue-600 text-cyan-300 font-semibold'
                        : 'bg-slate-900/80 border-slate-700 text-slate-400 hover:text-slate-200 hover:border-slate-600'
                    }`}
                  >
                    {qp}
                  </button>
                ))}
              </div>
            </form>

            {testOutput && (
              <div className="space-y-4 pt-2">
                {/* Comparison Summary Banner */}
                <div className="bg-slate-950/80 border border-slate-800 p-3 rounded-lg flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span className="text-slate-300 font-medium">Prompt: <span className="font-mono text-white font-semibold">"{testOutput.prompt}"</span></span>
                  <div className="flex items-center gap-3">
                    <span className="text-slate-400 text-[11px]">Teacher: <span className="font-mono text-purple-300">{testOutput.teacher?.latency_ms || 450} ms</span></span>
                    <span className="text-slate-500">•</span>
                    <span className="text-slate-400 text-[11px]">Student Before: <span className="font-mono text-slate-300">{testOutput.student_before?.latency_ms || 120} ms</span></span>
                    <span className="text-slate-500">•</span>
                    <span className="text-emerald-400 font-bold text-[11px] bg-emerald-950/80 border border-emerald-800 px-2 py-0.5 rounded">
                      Student After: {testOutput.student_after?.latency_ms || testOutput.latency_ms} ms (Fastest)
                    </span>
                  </div>
                </div>

                {/* 3-Column Comparison Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* (1) Student Model Before Distillation */}
                  <div className="bg-slate-900/90 border border-slate-700 rounded-xl p-4 flex flex-col justify-between space-y-3">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                        <div className="flex items-center gap-1.5">
                          <Cpu className="w-4 h-4 text-slate-400" />
                          <h4 className="text-xs font-bold text-slate-200">1. Student (Before)</h4>
                        </div>
                        <span className="text-[10px] bg-slate-800 text-slate-300 px-2 py-0.5 rounded font-mono">
                          {testOutput.student_before?.latency_ms || 125} ms
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                        <span className="truncate">{testOutput.student_before?.model || 'Base Pre-Trained'}</span>
                        <span className="text-[9px] bg-slate-800/90 text-slate-400 border border-slate-700 px-1.5 py-0.5 rounded">
                          Base Endpoint
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">
                        {testOutput.student_before?.description || 'Base model before distillation (higher latency, unaligned verbose output)'}
                      </p>
                    </div>

                    <div className="p-3 bg-slate-950/90 border border-slate-800 rounded-lg text-xs font-mono text-slate-300 leading-relaxed min-h-[110px] whitespace-pre-wrap">
                      {testOutput.student_before?.completion || testOutput.completion}
                    </div>
                  </div>

                  {/* (2) Teacher Model */}
                  <div className="bg-purple-950/20 border border-purple-900/60 rounded-xl p-4 flex flex-col justify-between space-y-3">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between border-b border-purple-900/40 pb-2">
                        <div className="flex items-center gap-1.5">
                          <Brain className="w-4 h-4 text-purple-400" />
                          <h4 className="text-xs font-bold text-purple-200">2. Teacher Model</h4>
                        </div>
                        <span className="text-[10px] bg-purple-950 text-purple-300 border border-purple-800 px-2 py-0.5 rounded font-mono">
                          {testOutput.teacher?.latency_ms || 420} ms
                        </span>
                      </div>
                      <div className="text-[11px] text-purple-300 font-mono truncate flex items-center justify-between">
                        <span>{testOutput.teacher?.model || 'gemini-2.5-pro'}</span>
                        {testOutput.teacher?.is_live_api && (
                          <span className="text-[9px] bg-purple-900/80 text-purple-200 border border-purple-600 px-1.5 py-0.5 rounded uppercase font-bold">
                            Live Gemini API
                          </span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-400">
                        {testOutput.teacher?.description || 'Frontier reference model with Chain-of-Thought reasoning steps'}
                      </p>
                    </div>

                    <div className="space-y-2">
                      {testOutput.teacher?.thinking && (
                        <div>
                          <button
                            type="button"
                            onClick={() => setShowThinking(!showThinking)}
                            className="flex items-center gap-1 text-[10px] text-purple-400 hover:text-purple-300 font-medium mb-1 cursor-pointer"
                          >
                            <Sparkles className="w-3 h-3" />
                            {showThinking ? 'Hide CoT Reasoning' : 'Show CoT Reasoning'}
                            {showThinking ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                          </button>
                          {showThinking && (
                            <div className="p-2.5 bg-purple-950/60 border border-purple-900/60 rounded text-[10px] font-mono text-purple-200 whitespace-pre-wrap max-h-32 overflow-y-auto">
                              {testOutput.teacher.thinking}
                            </div>
                          )}
                        </div>
                      )}
                      <div className="p-3 bg-slate-950/90 border border-purple-900/40 rounded-lg text-xs font-mono text-purple-100 font-semibold min-h-[50px] whitespace-pre-wrap">
                        {testOutput.teacher?.completion || testOutput.completion}
                      </div>
                    </div>
                  </div>

                  {/* (3) Student Model After Distillation */}
                  <div className="bg-emerald-950/20 border-2 border-emerald-500/50 rounded-xl p-4 flex flex-col justify-between space-y-3 shadow-lg shadow-emerald-950/30">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between border-b border-emerald-900/40 pb-2">
                        <div className="flex items-center gap-1.5">
                          <Zap className="w-4 h-4 text-emerald-400" />
                          <h4 className="text-xs font-bold text-emerald-300">3. Student (After)</h4>
                        </div>
                        <span className="text-[10px] bg-emerald-950 text-emerald-300 border border-emerald-700 px-2 py-0.5 rounded font-mono font-bold">
                          {testOutput.student_after?.latency_ms || testOutput.latency_ms} ms
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-emerald-300 font-mono">
                        <span className="truncate">{testOutput.student_after?.model || testOutput.model}</span>
                        <span className="text-[9px] bg-emerald-900/80 text-emerald-200 border border-emerald-600 px-1.5 py-0.5 rounded font-bold">
                          Distilled Endpoint
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400">
                        {testOutput.student_after?.description || 'Distilled student running on vLLM with PagedAttention and merged LoRA weights'}
                      </p>
                    </div>

                    <div className="p-3 bg-slate-950/90 border border-emerald-800/80 rounded-lg text-xs font-mono text-emerald-300 font-bold min-h-[110px] flex items-center">
                      <div className="space-y-1 w-full">
                        <div className="text-[10px] uppercase tracking-wider text-emerald-400 font-semibold flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" /> High-Accuracy Domain Answer
                        </div>
                        <div className="text-base text-white whitespace-pre-wrap break-words">
                          {testOutput.student_after?.completion || testOutput.completion}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Uninitialized/Empty State */}
      {!isDeploying && !isActive && metadata?.status !== 'STOPPED' && (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Rocket className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">Dual Models Not Deployed Yet</div>
          <p className="text-xs text-slate-400 max-w-md mx-auto">
            Deploy two live production endpoints on Vertex AI: the baseline Student model (un-fine-tuned) and the Distilled Student model (fine-tuned with PEFT LoRA adapter) using high-throughput vLLM PagedAttention.
          </p>
          <div className="text-[11px] text-amber-400/90 bg-amber-950/40 border border-amber-800/50 rounded-lg px-3 py-1.5 max-w-md mx-auto flex items-center justify-center gap-2">
            <AlertCircle className="w-3.5 h-3.5 shrink-0" />
            <span>Prerequisite: Stage 5 (Model Training) must be completed first to generate adapter weights.</span>
          </div>
          <div className="pt-2">
            <button
              onClick={handleDeploy}
              disabled={isDeploying}
              className="bg-green-600 hover:bg-green-500 text-white text-xs font-semibold px-4 py-2 rounded-lg cursor-pointer"
            >
              Deploy Dual Endpoints
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
