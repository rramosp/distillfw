import React, { useState, useEffect } from 'react';
import { Cpu, Play, RefreshCw, Activity, CheckCircle2, AlertCircle, HardDrive, BarChart3, Square, RotateCcw, ExternalLink } from 'lucide-react';
import { startTraining, stopTraining, clearTraining, fetchTrainingMetrics, fetchTrainingHeartbeat } from '../api';

export default function TrainingTab({ bucket, projectId, onStatusChange }) {
  const [metrics, setMetrics] = useState([]);
  const [heartbeat, setHeartbeat] = useState(null);
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [dryRun, setDryRun] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const loadTelemetry = async () => {
    try {
      const [m, hb] = await Promise.all([
        fetchTrainingMetrics(bucket, projectId),
        fetchTrainingHeartbeat(bucket, projectId)
      ]);
      if (Array.isArray(m)) setMetrics(m);
      if (hb) setHeartbeat(hb);
    } catch (err) {
      // silent polling error
    }
  };

  useEffect(() => {
    if (projectId) {
      loadTelemetry();
      const interval = setInterval(loadTelemetry, 3000);
      return () => clearInterval(interval);
    }
  }, [bucket, projectId]);

  const isBusy = starting || heartbeat?.status === 'RUNNING' || heartbeat?.status === 'PENDING';
  const isFinished = !isBusy && (heartbeat?.status === 'COMPLETED' || (metrics.length > 0 && heartbeat?.status !== 'FAILED'));

  const handleStart = async () => {
    setStarting(true);
    setErrorMsg(null);
    try {
      await startTraining(bucket, projectId, dryRun);
      setTimeout(loadTelemetry, 1500);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to launch training: ${err.message}`);
    } finally {
      setStarting(false);
    }
  };

  const handleStop = async () => {
    setStopping(true);
    try {
      await stopTraining(bucket, projectId);
      setTimeout(loadTelemetry, 1000);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to stop training: ${err.message}`);
    } finally {
      setStopping(false);
    }
  };

  const handleClear = async () => {
    try {
      await clearTraining(bucket, projectId);
      setMetrics([]);
      setHeartbeat(null);
      setErrorMsg(null);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to clear training state: ${err.message}`);
    }
  };

  const latest = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  // Simple SVG Line Chart for Loss
  const renderLossChart = () => {
    if (metrics.length === 0) return null;
    const width = 500;
    const height = 180;
    const padding = 30;

    const validLosses = metrics.map((m) => m.train_loss).filter((l) => l !== null && !isNaN(l));
    if (validLosses.length === 0) return null;

    const maxLoss = Math.max(...validLosses, 1.0);
    const minLoss = Math.min(...validLosses, 0.0);

    const getX = (idx) => padding + (idx / Math.max(1, metrics.length - 1)) * (width - 2 * padding);
    const getY = (loss) => height - padding - ((loss - minLoss) / Math.max(0.01, maxLoss - minLoss)) * (height - 2 * padding);

    const points = metrics
      .map((m, idx) => (m.train_loss != null ? `${getX(idx)},${getY(m.train_loss)}` : null))
      .filter(Boolean)
      .join(' ');

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-44">
        {/* Gridlines */}
        <line x1={padding} y1={padding} x2={width - padding} y2={padding} stroke="#334155" strokeDasharray="3 3" />
        <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#334155" strokeDasharray="3 3" />
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#475569" />

        {/* Labels */}
        <text x={padding - 5} y={padding + 4} fill="#94a3b8" fontSize="9" textAnchor="end">{maxLoss.toFixed(2)}</text>
        <text x={padding - 5} y={height - padding + 4} fill="#94a3b8" fontSize="9" textAnchor="end">{minLoss.toFixed(2)}</text>

        {/* Loss line */}
        <polyline fill="none" stroke="#38bdf8" strokeWidth="2.5" points={points} />
        {/* Data dots */}
        {metrics.map((m, idx) => m.train_loss != null && (
          <circle key={idx} cx={getX(idx)} cy={getY(m.train_loss)} r="3" fill="#0284c7" />
        ))}
      </svg>
    );
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header card */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Vertex AI Custom Training & Live Telemetry
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Standard PyTorch <code className="text-cyan-300 font-mono">transformers.Trainer</code> loop with PEFT LoRA, streaming <code className="text-cyan-300 font-mono">training/metrics.jsonl</code> directly to GCS.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={dryRun}
              onChange={(e) => setDryRun(e.target.checked)}
              disabled={isBusy}
              className="rounded bg-slate-900 border-slate-700 text-indigo-600 focus:ring-0 disabled:opacity-50"
            />
            <span title="Simulates full training steps locally without submitting a live GCP Vertex CustomJob">
              Dry-run / Local Worker
            </span>
          </label>
          {isBusy ? (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-rose-500/20"
            >
              <Square className="w-4 h-4 fill-current" />
              {stopping ? 'Stopping...' : 'Stop Training'}
            </button>
          ) : isFinished ? (
            <button
              onClick={handleClear}
              className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold px-4 py-2.5 rounded-lg border border-slate-600 shadow-md"
            >
              <RotateCcw className="w-4 h-4 text-slate-300" />
              Start Over
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={isBusy}
              className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-indigo-500/20 disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              {starting ? 'Launching...' : 'Launch Training Job'}
            </button>
          )}
          <button
            onClick={loadTelemetry}
            className="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 rounded-lg"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {(errorMsg || heartbeat?.error) && (
        <div className="p-4 bg-rose-950/80 border border-rose-800 rounded-lg text-rose-300 text-xs flex items-start gap-2">
          <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold">Training Error:</span>
            <p className="font-mono">{errorMsg || heartbeat?.error}</p>
          </div>
        </div>
      )}

      {/* Vertex AI Job Details Card */}
      {heartbeat?.job_id && (
        <div className="bg-slate-800/80 border border-indigo-900/50 rounded-xl p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase tracking-wider bg-indigo-900/60 text-indigo-300 border border-indigo-700">
                {heartbeat?.mode === 'vertex_ai' ? 'Vertex AI CustomJob' : 'Local Worker'}
              </span>
              <span className={`font-semibold ${heartbeat?.status === 'RUNNING' ? 'text-emerald-400' : heartbeat?.status === 'COMPLETED' ? 'text-blue-400' : heartbeat?.status === 'FAILED' ? 'text-rose-400' : 'text-slate-300'}`}>
                {heartbeat?.status}
              </span>
            </div>
            <div className="text-slate-300 font-mono text-[11px] break-all">
              {heartbeat?.job_id}
            </div>
            {heartbeat?.machine_type && (
              <div className="text-slate-400 text-[11px]">
                Hardware: <span className="text-cyan-300 font-mono">{heartbeat.machine_type}</span> ({heartbeat.accelerator_type || 'GPU'})
              </div>
            )}
          </div>
          {heartbeat?.web_url && (
            <a
              href={heartbeat.web_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/30 transition-colors whitespace-nowrap text-xs"
            >
              <span>Vertex AI Console</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          )}
        </div>
      )}

      {/* Heartbeat Status Indicator */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${heartbeat?.status === 'RUNNING' ? 'bg-emerald-400 animate-pulse' : heartbeat?.status === 'COMPLETED' ? 'bg-blue-400' : heartbeat?.status === 'FAILED' ? 'bg-rose-500' : 'bg-slate-600'}`} />
          <div>
            <div className="text-[10px] text-slate-400 uppercase font-semibold">Worker Heartbeat</div>
            <div className="text-sm font-bold text-white mt-0.5">{heartbeat?.status || 'IDLE'}</div>
          </div>
        </div>

        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Global Step</div>
          <div className="text-xl font-bold text-cyan-300 mt-0.5">{latest?.step || 0}</div>
        </div>

        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Current Loss</div>
          <div className="text-xl font-bold text-white mt-0.5">{latest?.train_loss ? latest.train_loss.toFixed(4) : '--'}</div>
        </div>

        <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
          <div className="text-[10px] text-slate-400 uppercase font-semibold">Throughput</div>
          <div className="text-xl font-bold text-emerald-400 mt-0.5">{latest?.tokens_per_sec || 0} <span className="text-xs font-normal text-slate-400">tok/s</span></div>
        </div>
      </div>

      {/* Live Charts Card */}
      {metrics.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Train Loss Chart */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold uppercase text-slate-300 tracking-wider flex items-center gap-2">
                <Activity className="w-4 h-4 text-cyan-400" />
                Training Loss Curve
              </h3>
              <span className="text-xs text-slate-400 font-mono">{metrics.length} data points</span>
            </div>
            <div className="bg-slate-900/60 rounded-lg p-2 border border-slate-800">
              {renderLossChart()}
            </div>
          </div>

          {/* Hardware & Memory Metrics */}
          <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-bold uppercase text-slate-300 tracking-wider flex items-center gap-2">
              <HardDrive className="w-4 h-4 text-purple-400" />
              Live Hardware Utilization
            </h3>

            <div className="space-y-4 pt-2">
              <div>
                <div className="flex justify-between text-xs font-medium text-slate-300 mb-1">
                  <span>GPU Compute Utilization</span>
                  <span className="text-cyan-400 font-mono">{latest?.gpu_utilization_pct || 0}%</span>
                </div>
                <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="bg-cyan-500 h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, latest?.gpu_utilization_pct || 0)}%` }}
                  />
                </div>
              </div>

              <div>
                <div className="flex justify-between text-xs font-medium text-slate-300 mb-1">
                  <span>GPU VRAM Allocated</span>
                  <span className="text-purple-400 font-mono">{latest?.memory_allocated_gb || 0} GB / 24 GB</span>
                </div>
                <div className="w-full bg-slate-900 h-2.5 rounded-full overflow-hidden border border-slate-800">
                  <div
                    className="bg-purple-500 h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.min(100, ((latest?.memory_allocated_gb || 0) / 24.0) * 100)}%` }}
                  />
                </div>
              </div>

              <div className="pt-4 border-t border-slate-700/60 grid grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="text-slate-500">Learning Rate:</span>
                  <div className="font-mono text-white mt-0.5">{latest?.learning_rate ? latest.learning_rate.toExponential(2) : '--'}</div>
                </div>
                <div>
                  <span className="text-slate-500">Current Epoch:</span>
                  <div className="font-mono text-white mt-0.5">{latest?.epoch || 0}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Cpu className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">No Training Run Started Yet</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Click "Start Distillation Training" below or in the header to start custom fine-tuning with PEFT QLoRA.
          </p>
        </div>
      )}

      {/* Bottom Action Section: Start Distillation Training */}
      <div className="pt-6 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4 bg-slate-800/40 p-5 rounded-xl border border-slate-700/50">
        <div>
          <h4 className="text-sm font-bold text-white">Distillation Pipeline Trigger</h4>
          <p className="text-xs text-slate-400">
            {isBusy
              ? 'Training job actively in execution on Vertex AI / local worker...'
              : isFinished
              ? 'Training completed. Check metrics above or start over to launch a new run.'
              : 'Launch student fine-tuning with LoRA adapter and streaming telemetry.'}
          </p>
        </div>

        <div>
          {isBusy ? (
            <button
              onClick={handleStop}
              disabled={stopping}
              className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold px-5 py-3 rounded-lg shadow-lg shadow-rose-500/20"
            >
              <Square className="w-4 h-4 fill-current" />
              {stopping ? 'Stopping...' : 'Stop Distillation Training'}
            </button>
          ) : isFinished ? (
            <button
              onClick={handleClear}
              className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-semibold px-5 py-3 rounded-lg border border-slate-600 shadow-md"
            >
              <RotateCcw className="w-4 h-4 text-slate-300" />
              Start Over
            </button>
          ) : (
            <button
              onClick={handleStart}
              disabled={isBusy}
              className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 via-purple-600 to-blue-600 hover:from-indigo-500 hover:to-blue-500 text-white text-xs font-semibold px-6 py-3 rounded-lg shadow-xl shadow-indigo-500/25 disabled:opacity-50"
            >
              <Play className="w-4 h-4" />
              {starting ? 'Launching Distillation...' : 'start distillation training'}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
