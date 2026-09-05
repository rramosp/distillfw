import React, { useState, useEffect } from 'react';
import { Sparkles, Play, RefreshCw, Eye, Brain, CheckCircle2, AlertCircle } from 'lucide-react';
import { runTeacherInference, fetchTeacherStatus } from '../api';

export default function TeacherTab({ bucket, projectId, onStatusChange }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [limit, setLimit] = useState(0); // 0 means all
  const [selectedSample, setSelectedSample] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);

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
    try {
      await runTeacherInference(bucket, projectId, limit > 0 ? limit : null);
      setTimeout(loadStatus, 2000);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to trigger teacher inference: ${err.message}`);
    } finally {
      setRunning(false);
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
          <div className="flex items-center gap-1 text-xs text-slate-300">
            <span>Limit:</span>
            <select
              value={limit}
              onChange={(e) => setLimit(parseInt(e.target.value))}
              className="bg-slate-900 border border-slate-700 rounded px-2 py-1 text-xs text-white"
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
            className="flex items-center gap-2 bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white text-xs font-semibold px-4 py-2 rounded-lg shadow-lg shadow-purple-500/20 disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5" />
            {running ? 'Extracting...' : 'Run Gemini Inference'}
          </button>
          <button
            onClick={loadStatus}
            className="p-2 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 rounded-lg"
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

      {data?.exists ? (
        <div className="space-y-6">
          {/* Progress / Stats pill */}
          <div className="flex items-center gap-4 bg-slate-800/40 p-4 rounded-xl border border-slate-700/60">
            <div className="w-10 h-10 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center">
              <CheckCircle2 className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <div className="text-sm font-bold text-white">{data.total} Enriched Teacher Completions</div>
              <div className="text-xs text-slate-400">
                Ready in <code className="text-purple-300 font-mono">data/teacher_inferences.jsonl</code> with reasoning traces.
              </div>
            </div>
          </div>

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
          <button
            onClick={handleRunInference}
            disabled={running}
            className="bg-purple-600 hover:bg-purple-500 text-white text-xs font-semibold px-4 py-2 rounded-lg"
          >
            Start Teacher Inference
          </button>
        </div>
      )}
    </div>
  );
}
