import React, { useState, useEffect } from 'react';
import { Database, Upload, Scissors, AlertCircle, CheckCircle2, FileText, Square, RotateCcw } from 'lucide-react';
import { fetchDatasetSummary, uploadDataset, splitDataset, clearDataset } from '../api';

export default function DatasetTab({ bucket, projectId, onStatusChange }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [splitting, setSplitting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const [rawText, setRawText] = useState('');
  const [activeTab, setActiveTab] = useState('summary'); // 'summary' | 'upload'
  const [errorMsg, setErrorMsg] = useState(null);
  const [successMsg, setSuccessMsg] = useState(null);

  const isBusy = uploading || splitting;

  const loadSummary = async () => {
    setLoading(true);
    try {
      const data = await fetchDatasetSummary(bucket, projectId);
      setSummary(data);
    } catch (err) {
      setErrorMsg(`Failed to load dataset: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) loadSummary();
  }, [bucket, projectId]);

  const handleStop = () => {
    setUploading(false);
    setSplitting(false);
    setErrorMsg('Operation stopped by user.');
  };

  const handleStartOver = async () => {
    if (!confirm('Start over? This will clear the current dataset and splits for this project.')) {
      return;
    }
    setClearing(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      await clearDataset(bucket, projectId);
      setSummary(null);
      setRawText('');
      setActiveTab('summary');
      setSuccessMsg('Dataset cleared. You can now start over with a new dataset.');
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Failed to clear dataset: ${err.message}`);
    } finally {
      setClearing(false);
    }
  };

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!rawText.trim()) return;
    setUploading(true);
    setErrorMsg(null);
    setSuccessMsg(null);
    try {
      const res = await uploadDataset(bucket, projectId, rawText);
      setSuccessMsg(`Uploaded and split ${res.counts?.total || 0} samples successfully!`);
      setActiveTab('summary');
      await loadSummary();
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Upload failed: ${err.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setRawText(event.target.result);
      setActiveTab('upload');
    };
    reader.readAsText(file);
  };

  const handleAutoSplit = async () => {
    setSplitting(true);
    setErrorMsg(null);
    try {
      await splitDataset(bucket, projectId, { train_ratio: 0.8, val_ratio: 0.1, test_ratio: 0.1, random_seed: 42 });
      setSuccessMsg('Dataset auto-split into Train (80%), Val (10%), Test (10%)');
      await loadSummary();
      if (onStatusChange) onStatusChange();
    } catch (err) {
      setErrorMsg(`Split failed: ${err.message}`);
    } finally {
      setSplitting(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header card */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <Database className="w-5 h-5 text-cyan-400" />
            Dataset Management & Splits
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Format: JSON Lines (<span className="font-mono text-cyan-300">.jsonl</span>) with <span className="font-mono text-cyan-300">"prompt"</span> and optional <span className="font-mono text-cyan-300">"split"</span> ("train", "val", "test").
          </p>
        </div>

        <div className="flex items-center gap-2">
          {isBusy && (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold px-3 py-2 rounded-lg transition shadow-md shadow-rose-600/30 cursor-pointer"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Stop</span>
            </button>
          )}

          {summary?.has_dataset && !isBusy ? (
            <button
              onClick={handleStartOver}
              disabled={clearing}
              className="flex items-center gap-1.5 bg-amber-600 hover:bg-amber-500 text-white text-xs font-semibold px-3 py-2 rounded-lg transition shadow-md shadow-amber-600/20 disabled:opacity-50 cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>{clearing ? 'Clearing...' : 'Start Over'}</span>
            </button>
          ) : (
            <>
              <label className={`cursor-pointer bg-slate-700 hover:bg-slate-600 text-white text-xs font-semibold px-3 py-2 rounded-lg transition flex items-center gap-1.5 border border-slate-600 ${isBusy ? 'opacity-50 pointer-events-none' : ''}`}>
                <Upload className="w-3.5 h-3.5" /> Upload File (.jsonl)
                <input type="file" accept=".jsonl,.json,.txt" onChange={handleFileUpload} disabled={isBusy} className="hidden" />
              </label>
              <button
                onClick={() => setActiveTab(activeTab === 'upload' ? 'summary' : 'upload')}
                disabled={isBusy}
                className={`text-xs font-semibold px-3 py-2 rounded-lg transition border disabled:opacity-50 cursor-pointer ${activeTab === 'upload' ? 'bg-blue-600 border-blue-500 text-white' : 'bg-slate-800 border-slate-700 text-slate-300'}`}
              >
                {activeTab === 'upload' ? 'View Summary' : 'Paste Raw Data'}
              </button>
            </>
          )}
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

      {/* Upload/Paste View */}
      {activeTab === 'upload' ? (
        <form onSubmit={handleUpload} className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-4">
          <h3 className="text-sm font-semibold text-white">Paste JSON Lines Data</h3>
          <p className="text-xs text-slate-400">
            Each row must be valid JSON with a <code className="text-cyan-300 font-mono">"prompt"</code> property.
          </p>
          <textarea
            rows={10}
            value={rawText}
            onChange={(e) => setRawText(e.target.value)}
            placeholder={'{"prompt": "Calculate 15 * 18"}\n{"prompt": "Solve 3x + 15 = 45"}'}
            className="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs font-mono text-slate-200 focus:border-blue-500 focus:outline-none"
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={() => setActiveTab('summary')}
              className="px-4 py-2 text-xs text-slate-400 hover:text-white"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={uploading}
              className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg disabled:opacity-50"
            >
              {uploading ? 'Validating & Uploading...' : 'Upload & Auto-Split'}
            </button>
          </div>
        </form>
      ) : (
        /* Summary View */
        <div className="space-y-5">
          {summary?.has_dataset ? (
            <>
              {/* Split Distribution Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
                <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4">
                  <div className="text-[11px] text-slate-400 font-semibold uppercase">Total Prompts</div>
                  <div className="text-2xl font-bold text-white mt-1">{summary.counts?.total || 0}</div>
                  <div className="text-[11px] text-slate-500 mt-0.5">Input rows validated</div>
                </div>
                <div className="bg-slate-800/60 border border-blue-900/40 rounded-xl p-4">
                  <div className="text-[11px] text-blue-400 font-semibold uppercase">Train Split</div>
                  <div className="text-2xl font-bold text-blue-300 mt-1">{summary.counts?.train || 0}</div>
                  <div className="text-[11px] text-blue-400/70 mt-0.5">Active for PEFT fine-tuning</div>
                </div>
                <div className="bg-slate-800/60 border border-amber-900/40 rounded-xl p-4">
                  <div className="text-[11px] text-amber-400 font-semibold uppercase">Val Split</div>
                  <div className="text-2xl font-bold text-amber-300 mt-1">{summary.counts?.val || 0}</div>
                  <div className="text-[11px] text-amber-400/70 mt-0.5">Loss tracking during training</div>
                </div>
                <div className="bg-slate-800/60 border border-emerald-900/40 rounded-xl p-4">
                  <div className="text-[11px] text-emerald-400 font-semibold uppercase">Test Split</div>
                  <div className="text-2xl font-bold text-emerald-300 mt-1">{summary.counts?.test || 0}</div>
                  <div className="text-[11px] text-emerald-400/70 mt-0.5">Quarantined for final eval</div>
                </div>
              </div>

              {/* Sample Rows Table */}
              <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                    <FileText className="w-4 h-4 text-cyan-400" />
                    Dataset Preview (First 5 Rows)
                  </h3>
                  <button
                    onClick={handleAutoSplit}
                    disabled={isBusy}
                    className="text-xs bg-slate-700 hover:bg-slate-600 text-slate-200 px-3 py-1.5 rounded-lg flex items-center gap-1.5 transition disabled:opacity-50 cursor-pointer"
                  >
                    <Scissors className="w-3.5 h-3.5 text-cyan-400" />
                    {splitting ? 'Splitting...' : 'Re-run Auto-Split (80/10/10)'}
                  </button>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs">
                    <thead>
                      <tr className="border-b border-slate-700 text-slate-400 uppercase text-[10px]">
                        <th className="py-2 px-3">#</th>
                        <th className="py-2 px-3">Split</th>
                        <th className="py-2 px-3">Prompt</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80">
                      {(summary.samples || []).map((row, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/40 font-mono">
                          <td className="py-2.5 px-3 text-slate-500">{idx + 1}</td>
                          <td className="py-2.5 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold border ${
                              row.split === 'train' ? 'bg-blue-950 text-blue-400 border-blue-800' :
                              row.split === 'val' ? 'bg-amber-950 text-amber-400 border-amber-800' :
                              'bg-emerald-950 text-emerald-400 border-emerald-800'
                            }`}>
                              {row.split}
                            </span>
                          </td>
                          <td className="py-2.5 px-3 text-slate-200 max-w-xl truncate">
                            {row.prompt}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          ) : (
            /* Empty State */
            <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
              <Database className="w-10 h-10 text-slate-500 mx-auto" />
              <div className="text-slate-300 font-semibold text-sm">No Dataset Uploaded Yet</div>
              <p className="text-xs text-slate-500 max-w-md mx-auto">
                Upload your domain-specific input dataset in JSON Lines format to start extracting knowledge from Gemini.
              </p>
              <button
                onClick={() => setActiveTab('upload')}
                className="bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg"
              >
                Upload or Paste Dataset
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
