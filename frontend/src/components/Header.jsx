import React, { useState } from 'react';
import { Layers, FolderPlus, Database, RefreshCw, ChevronDown } from 'lucide-react';
import StatusBadge from './StatusBadge';

export default function Header({
  buckets,
  selectedBucket,
  onSelectBucket,
  projects,
  selectedProject,
  onSelectProject,
  onCreateProject,
  status,
  statusDetail,
  onRefresh
}) {
  const [showNewModal, setShowNewModal] = useState(false);
  const [newProjectId, setNewProjectId] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [isCustomBucket, setIsCustomBucket] = useState(false);
  const [customBucketInput, setCustomBucketInput] = useState('');

  const handleCreate = async (e) => {
    e.preventDefault();
    if (!newProjectId.trim()) return;
    await onCreateProject(newProjectId.trim(), newDescription.trim());
    setNewProjectId('');
    setNewDescription('');
    setShowNewModal(false);
  };

  const handleCustomBucketSubmit = (e) => {
    e.preventDefault();
    if (customBucketInput.trim()) {
      onSelectBucket(customBucketInput.trim());
      setIsCustomBucket(false);
    }
  };

  return (
    <header className="bg-slate-900/90 border-b border-slate-800 sticky top-0 z-30 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
            <Layers className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-lg text-white tracking-tight">DistillFW</span>
              <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">GCP</span>
            </div>
            <p className="text-[11px] text-slate-400 -mt-0.5">Managed Model Distillation</p>
          </div>
        </div>

        {/* Combobox Selectors: Main Bucket & Workspace Folder */}
        <div className="flex items-center gap-3 flex-1 max-w-2xl justify-center">
          {/* Main Bucket Combobox */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700 rounded-lg px-2.5 py-1.5">
            <Database className="w-4 h-4 text-cyan-400 shrink-0" />
            <div className="flex flex-col">
              <span className="text-[10px] text-slate-400 uppercase font-semibold leading-none">Bucket</span>
              {!isCustomBucket ? (
                <div className="flex items-center gap-1">
                  <select
                    value={selectedBucket}
                    onChange={(e) => {
                      if (e.target.value === '__custom__') {
                        setIsCustomBucket(true);
                      } else {
                        onSelectBucket(e.target.value);
                      }
                    }}
                    className="bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer pr-4"
                  >
                    {buckets.map((b) => (
                      <option key={b} value={b} className="bg-slate-800 text-slate-200">
                        {b}
                      </option>
                    ))}
                    <option value="__custom__" className="bg-slate-800 text-blue-400">+ Custom Bucket...</option>
                  </select>
                </div>
              ) : (
                <form onSubmit={handleCustomBucketSubmit} className="flex items-center gap-1">
                  <input
                    type="text"
                    value={customBucketInput}
                    onChange={(e) => setCustomBucketInput(e.target.value)}
                    placeholder="gs://bucket-name"
                    className="bg-slate-900 border border-slate-600 rounded px-1.5 py-0.5 text-xs text-white focus:outline-none"
                    autoFocus
                  />
                  <button type="submit" className="text-[11px] bg-blue-600 hover:bg-blue-500 text-white px-1.5 py-0.5 rounded">Set</button>
                  <button type="button" onClick={() => setIsCustomBucket(false)} className="text-[11px] text-slate-400 hover:text-white">✕</button>
                </form>
              )}
            </div>
          </div>

          <span className="text-slate-600">/</span>

          {/* Workspace Folder Combobox */}
          <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700 rounded-lg px-2.5 py-1.5 flex-1 max-w-xs">
            <div className="flex flex-col flex-1">
              <span className="text-[10px] text-slate-400 uppercase font-semibold leading-none">Workspace Project</span>
              <select
                value={selectedProject}
                onChange={(e) => onSelectProject(e.target.value)}
                className="bg-transparent text-xs font-medium text-slate-200 focus:outline-none cursor-pointer truncate"
              >
                {projects.length === 0 && (
                  <option value="" className="bg-slate-800 text-slate-400">No projects found</option>
                )}
                {projects.map((p) => (
                  <option key={p} value={p} className="bg-slate-800 text-slate-200">
                    {p}
                  </option>
                ))}
              </select>
            </div>
            <button
              onClick={() => setShowNewModal(true)}
              title="Create New Project Workspace"
              className="p-1 rounded hover:bg-slate-700 text-slate-400 hover:text-cyan-400 transition"
            >
              <FolderPlus className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Status Badge & Refresh */}
        <div className="flex items-center gap-3">
          <StatusBadge status={status} detail={statusDetail} />
          <button
            onClick={onRefresh}
            title="Refresh Status"
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 transition"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* New Project Modal */}
      {showNewModal && (
        <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-slate-800 border border-slate-700 rounded-xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2">
              <FolderPlus className="w-5 h-5 text-blue-400" />
              Create Project Workspace
            </h3>
            <p className="text-xs text-slate-400 mb-4">
              Will initialize isolated workspace under <span className="text-cyan-400 font-mono">gs://{selectedBucket}/&lt;project-id&gt;/</span>
            </p>
            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Project ID</label>
                <input
                  type="text"
                  required
                  pattern="^[a-z0-9-]+$"
                  title="Lowercase alphanumeric and hyphens only"
                  placeholder="e.g. distill-gemma-math-v2"
                  value={newProjectId}
                  onChange={(e) => setNewProjectId(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-300 mb-1">Description (Optional)</label>
                <textarea
                  rows={2}
                  placeholder="Experiment objective and distillation notes..."
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:border-blue-500 focus:outline-none"
                />
              </div>
              <div className="flex justify-end gap-2 pt-2">
                <button
                  type="button"
                  onClick={() => setShowNewModal(false)}
                  className="px-4 py-2 text-xs font-medium text-slate-300 hover:text-white bg-slate-700/50 hover:bg-slate-700 rounded-lg"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-4 py-2 text-xs font-medium text-white bg-blue-600 hover:bg-blue-500 rounded-lg shadow-lg shadow-blue-500/20"
                >
                  Create Workspace
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </header>
  );
}
