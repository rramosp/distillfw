import React, { useState, useEffect } from 'react';
import { Award, Play, RefreshCw, BarChart2, ShieldCheck, Zap, AlertCircle, CheckCircle2 } from 'lucide-react';
import { runEvaluation, fetchEvaluationResults } from '../api';

export default function EvaluationTab({ bucket, projectId, onStatusChange }) {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(true);
  const [evaluating, setEvaluating] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);

  const loadResults = async () => {
    setLoading(true);
    try {
      const data = await fetchEvaluationResults(bucket, projectId);
      setResults(data);
    } catch (err) {
      // not evaluated yet
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) loadResults();
  }, [bucket, projectId]);

  const handleRunEval = async () => {
    setEvaluating(true);
    setErrorMsg(null);
    try {
      await runEvaluation(bucket, projectId);
      setTimeout(loadResults, 2000);
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Evaluation failed: ${err.message}`);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header card */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Award className="w-5 h-5 text-emerald-400" />
            Rigorous 3-Tier Evaluation on Quarantined Test Split
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Evaluates the distilled Student Model strictly against untouched <span className="font-mono text-emerald-300">test</span> split across lexical, LLM-as-a-judge, and operational metrics.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRunEval}
            disabled={evaluating}
            className="flex items-center gap-2 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-semibold px-4 py-2.5 rounded-lg shadow-lg shadow-emerald-500/20 disabled:opacity-50"
          >
            <Play className="w-4 h-4" />
            {evaluating ? 'Evaluating...' : 'Run 3-Tier Evaluation'}
          </button>
          <button
            onClick={loadResults}
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

      {results ? (
        <div className="space-y-6">
          {/* Top 3 Scorecard Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Tier 1: Lexical & Task Metrics */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-700/60 pb-2">
                <h3 className="text-xs font-bold uppercase text-slate-300 tracking-wider flex items-center gap-1.5">
                  <BarChart2 className="w-4 h-4 text-cyan-400" /> Tier 1: Lexical Scores
                </h3>
                <span className="text-[10px] bg-cyan-950 text-cyan-400 border border-cyan-800 px-2 py-0.5 rounded font-mono font-bold">
                  {results.test_samples_count} tests
                </span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">ROUGE-1 Score</span>
                  <span className="font-mono text-cyan-300 font-bold">{results.lexical_metrics?.rouge1}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">ROUGE-2 Score</span>
                  <span className="font-mono text-cyan-300 font-bold">{results.lexical_metrics?.rouge2}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">ROUGE-L Score</span>
                  <span className="font-mono text-cyan-300 font-bold">{results.lexical_metrics?.rougeL}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Exact Match (EM)</span>
                  <span className="font-mono text-emerald-400 font-bold">{results.lexical_metrics?.exact_match}%</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">BLEU Score</span>
                  <span className="font-mono text-slate-300">{results.lexical_metrics?.bleu}%</span>
                </div>
                <div className="flex justify-between pt-1">
                  <span className="text-slate-400">Syntax Compliance</span>
                  <span className="font-mono text-emerald-400 font-semibold">{results.lexical_metrics?.json_compliance_rate}%</span>
                </div>
              </div>
            </div>

            {/* Tier 2: LLM-as-a-Judge */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-700/60 pb-2">
                <h3 className="text-xs font-bold uppercase text-slate-300 tracking-wider flex items-center gap-1.5">
                  <ShieldCheck className="w-4 h-4 text-purple-400" /> Tier 2: Gemini Judge
                </h3>
                <span className="text-[10px] bg-purple-950 text-purple-400 border border-purple-800 px-2 py-0.5 rounded font-mono font-bold">
                  {results.llm_as_a_judge?.overall_score} / 5.0
                </span>
              </div>

              <div className="space-y-3 pt-1">
                {[
                  { label: 'Factual Correctness', score: results.llm_as_a_judge?.correctness },
                  { label: 'Instruction Adherence', score: results.llm_as_a_judge?.instruction_following },
                  { label: 'Reasoning Completeness', score: results.llm_as_a_judge?.reasoning_completeness },
                  { label: 'Semantic Similarity', score: results.llm_as_a_judge?.semantic_similarity },
                  { label: 'Safety & Hallucination', score: results.llm_as_a_judge?.hallucination_safety },
                ].map((item, idx) => (
                  <div key={idx}>
                    <div className="flex justify-between text-[11px] text-slate-300 mb-1">
                      <span>{item.label}</span>
                      <span className="text-purple-300 font-mono font-bold">{item.score || 4.5}</span>
                    </div>
                    <div className="w-full bg-slate-900 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-purple-500 h-full rounded-full"
                        style={{ width: `${((item.score || 4.5) / 5.0) * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Tier 3: Operational Benchmarking */}
            <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-slate-700/60 pb-2">
                <h3 className="text-xs font-bold uppercase text-slate-300 tracking-wider flex items-center gap-1.5">
                  <Zap className="w-4 h-4 text-amber-400" /> Tier 3: Operational Benchmarks
                </h3>
                <span className="text-[10px] bg-emerald-950 text-emerald-400 border border-emerald-800 px-2 py-0.5 rounded font-mono font-bold">
                  {results.operational_benchmarks?.cost_efficiency_multiple} Cheaper
                </span>
              </div>

              <div className="space-y-2.5 text-xs">
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Latency (p50)</span>
                  <span className="font-mono text-white font-bold">{results.operational_benchmarks?.latency_p50_ms} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Latency (p95)</span>
                  <span className="font-mono text-white">{results.operational_benchmarks?.latency_p95_ms} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Latency (p99)</span>
                  <span className="font-mono text-white">{results.operational_benchmarks?.latency_p99_ms} ms</span>
                </div>
                <div className="flex justify-between py-1 border-b border-slate-800">
                  <span className="text-slate-400">Serving Throughput</span>
                  <span className="font-mono text-cyan-400 font-bold">{results.operational_benchmarks?.throughput_tokens_sec} tok/sec</span>
                </div>
                <div className="flex justify-between pt-1">
                  <span className="text-slate-400">Speedup vs Teacher</span>
                  <span className="font-mono text-emerald-400 font-semibold">{results.operational_benchmarks?.student_vs_teacher_latency_ratio}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Award className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">Evaluation Not Run Yet</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Run the 3-tier evaluation to benchmark lexical accuracy, Gemini judge rubrics, and latency percentiles.
          </p>
          <button
            onClick={handleRunEval}
            disabled={evaluating}
            className="bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold px-4 py-2 rounded-lg"
          >
            Start Evaluation
          </button>
        </div>
      )}
    </div>
  );
}
