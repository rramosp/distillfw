import React, { useState, useEffect, useRef } from 'react';
import { Terminal, ChevronUp, ChevronDown, Trash2, Shield, Pause, Play } from 'lucide-react';
import { fetchLogs, clearLogs } from '../api';

export default function CollapsibleLogPanel({ projectId }) {
  const [isOpen, setIsOpen] = useState(false);
  const [logs, setLogs] = useState([]);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterLevel, setFilterLevel] = useState('ALL');
  const scrollRef = useRef(null);

  useEffect(() => {
    let isMounted = true;

    const loadLogs = async () => {
      try {
        const data = await fetchLogs(projectId);
        if (isMounted && Array.isArray(data)) {
          setLogs(data);
        }
      } catch (err) {
        // silent polling error
      }
    };

    loadLogs();
    const interval = setInterval(loadLogs, 2000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, [projectId]);

  useEffect(() => {
    if (isOpen && autoScroll && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, isOpen, autoScroll]);

  const handleClear = async () => {
    await clearLogs();
    setLogs([]);
  };

  const filteredLogs = logs.filter((log) => {
    if (filterLevel === 'ALL') return true;
    return log.level === filterLevel;
  });

  const getLevelBadge = (level) => {
    switch (level) {
      case 'SUCCESS':
        return 'bg-emerald-950 text-emerald-400 border-emerald-800';
      case 'WARNING':
        return 'bg-amber-950 text-amber-400 border-amber-800';
      case 'ERROR':
        return 'bg-rose-950 text-rose-400 border-rose-800';
      default:
        return 'bg-slate-800 text-blue-400 border-slate-700';
    }
  };

  return (
    <div className="fixed bottom-0 left-0 right-0 z-40 bg-slate-900 border-t border-slate-800 shadow-2xl transition-all duration-300">
      {/* Header bar (always visible) */}
      <div className="h-10 px-4 flex items-center justify-between bg-slate-950/80 hover:bg-slate-950 cursor-pointer select-none"
           onClick={() => setIsOpen(!isOpen)}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
            <Terminal className="w-4 h-4 text-cyan-400" />
            <span>Operations Telemetry & Execution Log</span>
            <span className="px-2 py-0.5 rounded-full text-[10px] bg-slate-800 text-slate-400">
              {logs.length} entries
            </span>
          </div>

          {/* Last log preview if collapsed */}
          {!isOpen && logs.length > 0 && (
            <span className="text-xs text-slate-500 truncate max-w-md hidden sm:inline">
              Latest: <span className="text-slate-300 font-mono">[{logs[logs.length - 1].source}] {logs[logs.length - 1].message}</span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-2" onClick={(e) => e.stopPropagation()}>
          {isOpen && (
            <>
              {/* Level Filter */}
              <div className="flex items-center gap-1 bg-slate-900 rounded p-0.5 border border-slate-800 text-[11px]">
                {['ALL', 'INFO', 'SUCCESS', 'ERROR'].map((lvl) => (
                  <button
                    key={lvl}
                    onClick={() => setFilterLevel(lvl)}
                    className={`px-2 py-0.5 rounded ${filterLevel === lvl ? 'bg-blue-600 text-white font-medium' : 'text-slate-400 hover:text-white'}`}
                  >
                    {lvl}
                  </button>
                ))}
              </div>

              {/* Auto scroll toggle */}
              <button
                onClick={() => setAutoScroll(!autoScroll)}
                title={autoScroll ? 'Pause Auto-scroll' : 'Resume Auto-scroll'}
                className={`p-1.5 rounded text-xs flex items-center gap-1 border ${autoScroll ? 'bg-slate-800 border-slate-700 text-cyan-400' : 'bg-slate-900 border-slate-800 text-slate-500'}`}
              >
                {autoScroll ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                <span className="text-[10px] hidden md:inline">Auto-scroll</span>
              </button>

              {/* Clear */}
              <button
                onClick={handleClear}
                title="Clear Logs"
                className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-rose-400 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </>
          )}

          <button
            onClick={() => setIsOpen(!isOpen)}
            className="p-1 rounded text-slate-400 hover:text-white"
          >
            {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Expanded panel body */}
      {isOpen && (
        <div
          ref={scrollRef}
          className="h-64 overflow-y-auto p-3 font-mono text-xs space-y-1.5 bg-slate-950 border-t border-slate-800/80"
        >
          {filteredLogs.length === 0 ? (
            <div className="text-slate-600 text-center py-8 italic">
              No operations logged yet. Actions like teacher inference, probe, training, and evaluation will stream live output here.
            </div>
          ) : (
            filteredLogs.map((entry) => (
              <div key={entry.id} className="flex items-start gap-2.5 leading-relaxed hover:bg-slate-900/50 px-2 py-0.5 rounded">
                <span className="text-slate-500 shrink-0 select-none text-[11px]">{entry.timestamp}</span>
                <span className={`px-1.5 py-0.2 rounded text-[10px] uppercase font-bold border shrink-0 ${getLevelBadge(entry.level)}`}>
                  {entry.level}
                </span>
                <span className="text-cyan-400 font-semibold shrink-0 select-none">
                  [{entry.source}]
                </span>
                <span className="text-slate-200 break-all flex-1">
                  {entry.message}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  );
}
