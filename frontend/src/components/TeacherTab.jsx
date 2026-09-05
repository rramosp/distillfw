import React, { useState, useEffect } from 'react';
import { Sparkles, Play, RefreshCw, Eye, Brain, CheckCircle2, AlertCircle, Square, RotateCcw, ShieldAlert, ChevronDown, ChevronUp } from 'lucide-react';
import { runTeacherInference, fetchTeacherStatus, stopTeacherInference, clearTeacherInferences } from '../api';

export default function TeacherTab({ bucket, projectId, onStatusChange }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [showErrorDetails, setShowErrorDetails] = useState(false);
  const [limit, setLimit] = useState(0); // 0 means all
  const [selectedSample, setSelectedSample] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const loadStatus = async () => {
    setLoading(true);
    try {
      const res = await fetchTeacherStatus(bucket, projectId, 20);
      setData(res);
      if (res?.samples?.length > 0 && !selectedSample) {
        setSelectedSample(res.samples[0]);
      }
    } catch (err) {
      setErrorMsg(`Failed to load teacher inferences: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) loadStatus();
  }, [bucket, projectId]);

  const handleRunInference = async () => {
    setRunning(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      await runTeacherInference(bucket, projectId, limit > 0 ? limit : null);
      setSuccessMsg('Teacher inference initiated. Streaming completions...');
      setTimeout(loadStatus, 2000);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to trigger teacher inference: ${err.message}`);
      setRunning(false);
    }
  };

  const handleStop = async () => {
    try {
      await stopTeacherInference(bucket, projectId);
      setRunning(false);
      setErrorMsg('Teacher inference stopped by user.');
      setTimeout(loadStatus, 1000);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to stop inference: ${err.message}`);
    }
  };

  const handleStartOver = async () => {
    if (!confirm('Start over? This will clear all teacher completions and retry diagnostics for this project.')) {
      return;
    }
    setClearing(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      await clearTeacherInferences(bucket, projectId);
      setData(null);
      setSelectedSample(null);
      setSuccessMsg('Teacher inference data cleared. You can now start over.');
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to clear teacher data: ${err.message}`);
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Action Header Card */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            Teacher Model Inference & CoT Knowledge Extraction
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Extracts Chain-of-Thought (<span className="font-mono text-purple-300">teacher_thinking</span>) and reference answers (<span className="font-mono text-cyan-300">teacher_response</span>) via Gemini.
          </p>
        </div>

        <div className="flex items-center gap-3">
          {running ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-lg shadow-rose-600/30 cursor-pointer"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop Inference</span>
            </button>
          ) : data?.exists ? (
            <button
              onClick={handleStartOver}
              disabled={clearing}
              className="flex items-center gap-2 bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-lg shadow-amber-600/20 disabled:opacity-50 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>{clearing ? 'Clearing...' : 'Start Over'}</span>
            </button>
          ) : (
            <>
              <div className="flex items-center gap-1 text-xs text-slate-300">
                <span>Limit:</span>
                <select
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value))}
                  disabled={running}
                  className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white disabled:opacity-50"
                >
                  <option value={0}>All Prompts</option>
                  <option value={10}>First 10 Prompts</option>
                  <option value={25}>First 25 Prompts</option>
                  <option value={50}>First 50 Prompts</option>
                </select>
              </div>
              <button
                onClick={handleRunInference}
                disabled={running}
                className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-lg shadow-purple-500/20 disabled:opacity-50 cursor-pointer"
              >
                <Play className="w-3.5 h-3.5" />
                <span>Run Gemini Inference</span>
              </button>
            </>
          )}

          <button
            onClick={loadStatus}
            className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 rounded-lg cursor-pointer"
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

      {successMsg && (
        <div className="p-4 bg-emerald-950/80 border border-emerald-800 rounded-lg text-emerald-300 text-xs flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

      {data?.exists ? (
        <div className="space-y-6">
          {/* Progress / Stats & Diagnostics Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Total Completions */}
            <div className="flex items-center gap-3 bg-slate-800/40 p-4 rounded-xl border border-slate-700/60">
              <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center shrink-0">
                <CheckCircle2 className="w-5 h-5 text-purple-400" />
              </div>
              <div>
                <div className="text-sm font-bold text-white">{data.total} Enriched Completions</div>
                <div className="text-[11px] text-slate-400">
                  Ready in <code className="text-purple-300 font-mono">teacher_inferences.jsonl</code>
                </div>
              </div>
            </div>

            {/* Retries Diagnostics Card */}
            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/60 flex items-center justify-between">
              <div>
                <div className="text-[10px] text-slate-400 uppercase font-semibold">Gemini API Retries</div>
                <div className="text-xl font-bold text-white mt-0.5 flex items-center gap-2">
                  <span>{data.retries_count || 0}</span>
                  <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold border ${
                    (data.retries_count || 0) > 0 ? 'bg-amber-950 text-amber-300 border-amber-800' : 'bg-emerald-950 text-emerald-400 border-emerald-800'
                  }`}>
                    {(data.retries_count || 0) > 0 ? 'Auto-Recovered' : 'Zero Failures'}
                  </span>
                </div>
              </div>
              <ShieldAlert className={`w-6 h-6 ${(data.retries_count || 0) > 0 ? 'text-amber-400' : 'text-emerald-400'}`} />
            </div>

            {/* Error Types Diagnostics Card */}
            <div className="bg-slate-800/40 p-4 rounded-xl border border-slate-700/60">
              <div className="text-[10px] text-slate-400 uppercase font-semibold mb-1">Error Types Breakdown</div>
              {data.error_types && Object.keys(data.error_types).length > 0 ? (
                <div className="flex flex-wrap gap-1.5">
                  {Object.entries(data.error_types).map(([errType, count]) => (
                    <span key={errType} className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-950/80 border border-rose-800 text-rose-300">
                      {errType}: {count}
                    </span>
                  ))}
                </div>
              ) : (
                <div className="text-xs text-emerald-400 font-medium flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" /> No API Errors Encountered
                </div>
              )}
            </div>
          </div>

          {/* Collapsible Error History Drawer */}
          {data.errors_encountered && data.errors_encountered.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <button
                onClick={() => setShowErrorDetails(!showErrorDetails)}
                className="w-full flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white cursor-pointer"
              >
                <span className="flex items-center gap-1.5">
                  <AlertCircle className="w-4 h-4 text-amber-400" />
                  Retry & Error Event Log ({data.errors_encountered.length} recorded events)
                </span>
                {showErrorDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>

              {showErrorDetails && (
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-800 text-slate-400 uppercase text-[10px]">
                        <th className="py-2 px-2">Time</th>
                        <th className="py-2 px-2">Attempt</th>
                        <th className="py-2 px-2">Error Type</th>
                        <th className="py-2 px-2">Delay</th>
                        <th className="py-2 px-2">Prompt Snippet</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800 font-mono text-[11px]">
                      {data.errors_encountered.map((item, i) => (
                        <tr key={i} className="hover:bg-slate-800/40">
                          <td className="py-1.5 px-2 text-slate-400">{item.timestamp ? item.timestamp.slice(11, 19) : '--'}</td>
                          <td className="py-1.5 px-2 text-white">#{item.attempt}</td>
                          <td className="py-1.5 px-2 text-amber-400">{item.error_type}</td>
                          <td className="py-1.5 px-2 text-slate-300">{item.retry_delay_seconds}s</td>
                          <td className="py-1.5 px-2 text-slate-300 truncate max-w-xs">{item.prompt_snippet}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}


          {/* Interactive Inspector: Prompts list + CoT View */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* List */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 space-y-2">
              <h3 className="text-xs font-bold uppercase text-slate-400 tracking-wider mb-2">Inference Samples</h3>
              <div className="space-y-1.5 max-h-[500px] overflow-y-auto pr-1">
                {(data.samples || []).map((s, idx) => (
                  <button
                    key={idx}
                    onClick={() => setSelectedSample(s)}
                    className={`w-full text-left p-2.5 rounded-lg border text-xs transition ${
                      selectedSample === s
                        ? 'bg-purple-950/40 border-purple-600 text-white shadow'
                        : 'bg-slate-900/60 border-slate-800 text-slate-300 hover:bg-slate-800/80'
                    }`}
                  >
                    <div className="font-medium truncate">{idx + 1}. {s.prompt}</div>
                    <div className="text-[10px] text-slate-500 mt-1 flex items-center gap-2">
                      <span className="uppercase font-bold text-slate-400">{s.split}</span>
                      <span>•</span>
                      <span>Resp: {s.teacher_response?.slice(0, 15)}...</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Selected Sample Detail */}
            <div className="md:col-span-2 bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
              {selectedSample ? (
                <>
                  <div>
                    <span className="text-[10px] font-bold uppercase text-slate-400 tracking-wider">Input Prompt</span>
                    <div className="mt-1 p-3 bg-slate-900 border border-slate-800 rounded-lg text-xs font-mono text-slate-200">
                      {selectedSample.prompt}
                    </div>
                  </div>

                  <div>
                    <span className="text-[10px] font-bold uppercase text-purple-400 tracking-wider flex items-center gap-1.5">
                      <Brain className="w-3.5 h-3.5 text-purple-400" />
                      Gemini Chain-of-Thought (teacher_thinking)
                    </span>
                    <div className="mt-1 p-3 bg-purple-950/20 border border-purple-900/40 rounded-lg text-xs font-mono text-purple-200 whitespace-pre-wrap leading-relaxed max-h-56 overflow-y-auto">
                      {selectedSample.teacher_thinking || 'No thinking trace requested.'}
                    </div>
                  </div>

                  <div>
                    <span className="text-[10px] font-bold uppercase text-cyan-400 tracking-wider">
                      Teacher Reference Answer (teacher_response)
                    </span>
                    <div className="mt-1 p-3 bg-cyan-950/20 border border-cyan-900/40 rounded-lg text-xs font-mono text-cyan-200 font-bold">
                      {selectedSample.teacher_response}
                    </div>
                  </div>

                  {selectedSample.teacher_tokens && (
                    <div className="text-[11px] text-slate-400 flex items-center gap-4 pt-2 border-t border-slate-700/60">
                      <span>Prompt Tokens: <strong className="text-white">{selectedSample.teacher_tokens.prompt_tokens}</strong></span>
                      <span>Completion Tokens: <strong className="text-white">{selectedSample.teacher_tokens.completion_tokens}</strong></span>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-slate-500 text-xs text-center py-16">
                  Select a prompt from the list to inspect reasoning traces.
                </div>
              )}
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Sparkles className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">No Teacher Inferences Generated Yet</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Run the teacher model on your dataset to extract reference answers and chain-of-thought rationales.
          </p>
          {running ? (
            <button
              onClick={handleStop}
              className="bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold px-4 py-2 rounded-lg flex items-center gap-2 mx-auto cursor-pointer"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop Inference</span>
            </button>
          ) : (
            <button
              onClick={handleRunInference}
              disabled={running}
              className="bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold px-4 py-2 rounded-lg cursor-pointer disabled:opacity-50"
            >
              Start Teacher Inference
            </button>
          )}
        </div>
      )}
    </div>
  );
}
