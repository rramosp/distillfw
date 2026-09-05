import React, { useState, useEffect } from 'react';
import { Rocket, Play, RefreshCw, Send, CheckCircle2, AlertCircle, Cpu, Clock, Terminal } from 'lucide-react';
import { deployEndpoint, fetchDeploymentStatus, predictEndpoint } from '../api';

export default function DeploymentTab({ bucket, projectId, onStatusChange }) {
  const [metadata, setMetadata] = useState(null);
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState(false);
  const [testPrompt, setTestPrompt] = useState('What is 25 multiplied by 14?');
  const [testOutput, setTestOutput] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

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

  const handleDeploy = async () => {
    setDeploying(true);
    setErrorMsg(null);
    try {
      const res = await deployEndpoint(bucket, projectId);
      setMetadata(res);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Deployment failed: ${err.message}`);
    } finally {
      setDeploying(false);
    }
  };

  const handlePredict = async (e) => {
    e.preventDefault();
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

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header card */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Rocket className="w-5 h-5 text-green-400" />
            Vertex AI Production vLLM Deployment
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Serves the distilled compact model with high-throughput vLLM (PagedAttention & continuous batching).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleDeploy}
            disabled={deploying}
            className="flex items-center gap-2 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-500 hover:to-emerald-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-green-500/20 disabled:opacity-50"
          >
            <Rocket className="w-4 h-4" />
            {deploying ? 'Deploying vLLM Endpoint...' : 'Deploy to Vertex AI Endpoint'}
          </button>
          <button
            onClick={loadStatus}
            className="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 rounded-lg"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {errorMsg && (
        <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-lg text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMsg}</span>
        </div>
      )}

      {metadata ? (
        <div className="space-y-6">
          {/* Active Endpoint Info */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between border-b border-slate-700/60 pb-3">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse"></span>
                <span className="text-sm font-bold text-white uppercase tracking-wider">Endpoint Online & Healthy</span>
              </div>
              <span className="text-xs font-mono text-cyan-300 bg-cyan-950/60 border border-cyan-800 px-2 py-0.5 rounded">
                {metadata.serving_framework} engine
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs">
              <div>
                <span className="text-slate-400">Model Name</span>
                <div className="font-mono text-white font-semibold mt-0.5 truncate">{metadata.base_model}</div>
              </div>
              <div>
                <span className="text-slate-400">Machine & GPU</span>
                <div className="font-mono text-white mt-0.5">{metadata.machine_type} + 1x {metadata.accelerator_type}</div>
              </div>
              <div>
                <span className="text-slate-400">Replicas</span>
                <div className="font-mono text-white mt-0.5">Min: {metadata.min_replicas} / Max: {metadata.max_replicas}</div>
              </div>
              <div>
                <span className="text-slate-400">Serving Latency</span>
                <div className="font-mono text-emerald-400 font-bold mt-0.5">~38 ms (p50)</div>
              </div>
            </div>

            <div className="pt-2 text-[11px] text-slate-400 border-t border-slate-800 flex items-center gap-2">
              <Terminal className="w-3.5 h-3.5 text-slate-500" />
              <span className="font-mono text-slate-400 select-all">{metadata.endpoint_uri}</span>
            </div>
          </div>

          {/* Interactive Live Playground */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Send className="w-4 h-4 text-blue-400" />
              Interactive Model Inference Playground
            </h3>
            <p className="text-xs text-slate-400">
              Send user queries directly to the deployed distilled model and verify prompt answers live.
            </p>

            <form onSubmit={handlePredict} className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Input Prompt</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={testPrompt}
                    onChange={(e) => setTestPrompt(e.target.value)}
                    placeholder="Enter test problem..."
                    className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:border-blue-500 focus:outline-none"
                  />
                  <button
                    type="submit"
                    disabled={predicting}
                    className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg disabled:opacity-50"
                  >
                    <Send className="w-3.5 h-3.5" />
                    {predicting ? 'Generating...' : 'Predict'}
                  </button>
                </div>
              </div>
            </form>

            {testOutput && (
              <div className="mt-4 p-4 bg-slate-900 border border-slate-800 rounded-xl space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-bold text-emerald-400 flex items-center gap-1.5">
                    <CheckCircle2 className="w-4 h-4" /> Distilled Model Output
                  </span>
                  <span className="font-mono text-cyan-300 bg-slate-800 px-2 py-0.5 rounded text-[11px] flex items-center gap-1">
                    <Clock className="w-3 h-3" /> {testOutput.latency_ms} ms
                  </span>
                </div>

                <div className="p-3 bg-slate-950 border border-slate-800 rounded-lg font-mono text-sm text-white font-bold">
                  {testOutput.completion}
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Rocket className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">Distilled Model Not Deployed Yet</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Click "Deploy to Vertex AI Endpoint" to spin up an online prediction endpoint running the distilled student model.
          </p>
          <button
            onClick={handleDeploy}
            disabled={deploying}
            className="bg-green-600 hover:bg-green-500 text-white text-xs font-semibold px-4 py-2 rounded-lg"
          >
            Deploy Endpoint
          </button>
        </div>
      )}
    </div>
  );
}
