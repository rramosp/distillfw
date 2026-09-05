import React, { useState, useEffect } from 'react';
import { History, RefreshCw, CheckCircle2, AlertCircle, Clock } from 'lucide-react';
import { fetchProjectHistory } from '../api';

export default function HistoryTab({ bucket, projectId }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await fetchProjectHistory(bucket, projectId);
      if (Array.isArray(data)) setHistory(data.reverse());
    } catch (err) {
      // ignore
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) loadHistory();
  }, [bucket, projectId]);

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Header */}
      <div className="flex items-center justify-between bg-slate-800/80 p-5 rounded-xl border border-slate-700">
        <div>
          <h2 className="text-base font-bold text-white flex items-center gap-2">
            <History className="w-5 h-5 text-blue-400" />
            Project Execution History (history.json)
          </h2>
          <p className="text-xs text-slate-400 mt-1">
            Auditable chronicle of all operations, hyperparameters, start/end timestamps, and execution states.
          </p>
        </div>

        <button
          onClick={loadHistory}
          className="p-2.5 bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white border border-slate-700 rounded-lg"
        >
          <RefreshCw className="w-4 h-4" />
        </button>
      </div>

      {history.length > 0 ? (
        <div className="bg-slate-800/60 border border-slate-700 rounded-xl overflow-hidden shadow-lg">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-slate-700 text-slate-400 uppercase text-[10px] bg-slate-900/60">
                  <th className="py-3 px-4">Action</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4">Start Time</th>
                  <th className="py-3 px-4">Details & Parameters</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {history.map((entry, idx) => (
                  <tr key={idx} className="hover:bg-slate-800/40">
                    <td className="py-3 px-4 font-mono font-bold text-slate-200">
                      {entry.action}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold border ${
                        entry.status === 'SUCCESS' ? 'bg-emerald-950 text-emerald-400 border-emerald-800' :
                        entry.status === 'RUNNING' ? 'bg-blue-950 text-blue-400 border-blue-800 animate-pulse' :
                        'bg-rose-950 text-rose-400 border-rose-800'
                      }`}>
                        {entry.status}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-slate-400 font-mono text-[11px] whitespace-nowrap">
                      {entry.start_time?.slice(0, 19).replace('T', ' ')}
                    </td>
                    <td className="py-3 px-4 text-slate-300 text-xs font-mono max-w-md">
                      <div className="truncate text-white">{entry.details}</div>
                      {entry.parameters && Object.keys(entry.parameters).length > 0 && (
                        <div className="text-[10px] text-slate-500 truncate mt-0.5">
                          {JSON.stringify(entry.parameters)}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <History className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">No Action History Recorded Yet</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            All stage executions and configuration updates will be logged here.
          </p>
        </div>
      )}
    </div>
  );
}
