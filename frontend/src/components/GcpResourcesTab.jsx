import React, { useState, useEffect } from 'react';
import { 
  Cloud, ExternalLink, Copy, Check, Database, Cpu, 
  Sparkles, Rocket, Layers, ShieldCheck, Server, FileText, 
  RefreshCw, Search, Filter, CheckCircle2, Clock, AlertCircle, ArrowUpRight, Box
} from 'lucide-react';
import { fetchProjectResources } from '../api';

const SERVICE_ICONS = {
  'Cloud Storage': Database,
  'Vertex AI Training': Cpu,
  'Vertex AI Prediction': Rocket,
  'Vertex AI Model Registry': Layers,
  'Vertex AI Gemini API': Sparkles,
  'Artifact Registry': Box,
  'Cloud IAM': ShieldCheck,
  'Cloud Run': Server,
  'Cloud Logging': FileText,
};

const CATEGORIES = [
  { id: 'all', label: 'All Resources' },
  { id: 'Storage', label: 'Storage' },
  { id: 'Training', label: 'Custom Training' },
  { id: 'Serving', label: 'Online Serving' },
  { id: 'Models', label: 'Vertex Models' },
  { id: 'Registry', label: 'Artifact Registry' },
  { id: 'Security & IAM', label: 'IAM & Security' },
  { id: 'Compute', label: 'Cloud Run Compute' },
  { id: 'Observability', label: 'Cloud Logging' },
];

export default function GcpResourcesTab({ bucket, projectId }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [copiedId, setCopiedId] = useState(null);

  const loadResources = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchProjectResources(bucket, projectId);
      if (res && res.resources) {
        setData(res);
      } else {
        setData(null);
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch GCP resources');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (projectId) {
      loadResources();
    }
  }, [bucket, projectId]);

  const copyToClipboard = (text, id) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'ACTIVE':
      case 'SERVING':
      case 'STREAMING':
      case 'AVAILABLE':
      case 'CONFIGURED':
      case 'REGISTERED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-emerald-950/80 text-emerald-400 border border-emerald-700/60 shadow-sm">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            {status}
          </span>
        );
      case 'RUNNING':
      case 'INITIALIZING':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-blue-950/80 text-blue-400 border border-blue-700/60 shadow-sm animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-ping"></span>
            {status}
          </span>
        );
      case 'COMPLETED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-cyan-950/80 text-cyan-400 border border-cyan-700/60 shadow-sm">
            <CheckCircle2 className="w-3 h-3 text-cyan-400" />
            {status}
          </span>
        );
      case 'STOPPED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-amber-950/80 text-amber-400 border border-amber-700/60 shadow-sm">
            <Clock className="w-3 h-3 text-amber-400" />
            {status}
          </span>
        );
      case 'INITIALIZED':
      case 'NOT_STARTED':
      case 'READY_TO_REGISTER':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-slate-800 text-slate-300 border border-slate-700">
            <Clock className="w-3 h-3 text-slate-400" />
            {status}
          </span>
        );
      case 'NOT_DEPLOYED':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-slate-800/60 text-slate-500 border border-slate-700/60">
            {status}
          </span>
        );
      case 'FAILED':
      case 'ERROR':
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-rose-950/80 text-rose-400 border border-rose-700/60 shadow-sm">
            <AlertCircle className="w-3 h-3 text-rose-400" />
            {status}
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-bold uppercase tracking-wider bg-slate-800 text-slate-400 border border-slate-700">
            {status}
          </span>
        );
    }
  };

  const filteredResources = (data?.resources || []).filter((resource) => {
    const matchesCategory = selectedCategory === 'all' || resource.category === selectedCategory;
    const matchesSearch = !searchTerm || 
      resource.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resource.service.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resource.role.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resource.status.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  return (
    <div className="space-y-6 max-w-6xl mx-auto pb-16">
      {/* Workspace GCP Header Card */}
      <div className="bg-gradient-to-r from-blue-950/70 via-slate-900 to-indigo-950/70 border border-blue-800/60 rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-extrabold uppercase tracking-wider bg-blue-500/20 text-blue-300 border border-blue-500/40 flex items-center gap-1">
                <Cloud className="w-3 h-3" /> GCP Managed Workspace Resources
              </span>
              <span className="text-xs text-slate-400 font-mono">
                Region: <strong className="text-white">{data?.region || 'us-central1'}</strong>
              </span>
            </div>
            
            <h1 className="text-2xl font-black text-white flex items-center gap-3">
              <span>{projectId}</span>
            </h1>

            <p className="text-xs text-slate-300 max-w-2xl leading-relaxed">
              Real-time directory of all Google Cloud Platform services, storage buckets, Vertex AI endpoints, custom jobs, container registries, and IAM identities bound to this distillation experiment.
            </p>

            <div className="flex flex-wrap items-center gap-3 pt-1">
              <div className="flex items-center gap-1.5 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700 text-xs text-slate-300">
                <span className="text-slate-400">GCP Project:</span>
                <a 
                  href={`https://console.cloud.google.com/home/dashboard?project=${data?.gcp_project_id || 'distillfw'}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-mono text-cyan-300 hover:text-cyan-200 underline decoration-cyan-500/40 flex items-center gap-1 font-semibold"
                >
                  {data?.gcp_project_id || 'distillfw'}
                  <ExternalLink className="w-3 h-3" />
                </a>
              </div>

              <div className="flex items-center gap-1.5 bg-slate-900/80 px-3 py-1.5 rounded-lg border border-slate-700 text-xs text-slate-300">
                <span className="text-slate-400">Workspace Path:</span>
                <code className="font-mono text-amber-300 text-[11px]">gs://{bucket}/{projectId}/</code>
                <button
                  onClick={() => copyToClipboard(`gs://${bucket}/${projectId}/`, 'ws_path')}
                  className="text-slate-400 hover:text-white transition p-0.5"
                  title="Copy GCS URI"
                >
                  {copiedId === 'ws_path' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
          </div>

          <div className="flex flex-row lg:flex-col items-center lg:items-end justify-between gap-3">
            <button
              onClick={loadResources}
              disabled={loading}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-xs font-semibold px-4 py-2.5 rounded-xl shadow-lg shadow-blue-500/25 transition cursor-pointer"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
              <span>Refresh Statuses</span>
            </button>

            <a
              href={`https://console.cloud.google.com/home/dashboard?project=${data?.gcp_project_id || 'distillfw'}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-xs text-blue-300 hover:text-blue-200 transition font-medium"
            >
              <span>Open GCP Project Console</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      </div>

      {/* Summary Scorecards */}
      {data?.summary && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-4">
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Total Resources</div>
            <div className="text-2xl font-black text-white mt-1">{data.summary.total_resources}</div>
            <div className="text-[11px] text-slate-500 mt-1">Configured for workspace</div>
          </div>

          <div className="bg-slate-800/60 border border-emerald-900/40 rounded-xl p-4">
            <div className="text-[11px] font-bold text-emerald-400 uppercase tracking-wider">Active & Serving</div>
            <div className="text-2xl font-black text-emerald-300 mt-1">{data.summary.active_count}</div>
            <div className="text-[11px] text-slate-500 mt-1">Operational & live</div>
          </div>

          <div className="bg-slate-800/60 border border-blue-900/40 rounded-xl p-4">
            <div className="text-[11px] font-bold text-blue-400 uppercase tracking-wider">In Progress</div>
            <div className="text-2xl font-black text-blue-300 mt-1">{data.summary.in_progress_count}</div>
            <div className="text-[11px] text-slate-500 mt-1">Active jobs or probing</div>
          </div>

          <div className="bg-slate-800/60 border border-slate-700/80 rounded-xl p-4">
            <div className="text-[11px] font-bold text-slate-400 uppercase tracking-wider">Pending / Standby</div>
            <div className="text-2xl font-black text-slate-300 mt-1">{data.summary.ready_count + data.summary.not_deployed_count}</div>
            <div className="text-[11px] text-slate-500 mt-1">Ready to trigger or deploy</div>
          </div>
        </div>
      )}

      {/* Filter and Search Bar */}
      <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-4 space-y-3">
        <div className="flex flex-col sm:flex-row gap-3 items-center justify-between">
          <div className="relative w-full sm:w-80">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search resource, service, or role..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-9 pr-4 py-1.5 bg-slate-900/80 border border-slate-700 rounded-lg text-xs text-white placeholder-slate-500 focus:outline-none focus:border-blue-500"
            />
          </div>

          <div className="text-xs text-slate-400 self-start sm:self-center">
            Showing <strong className="text-white">{filteredResources.length}</strong> of {data?.resources?.length || 0} resources
          </div>
        </div>

        {/* Category Pills */}
        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setSelectedCategory(cat.id)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold whitespace-nowrap transition cursor-pointer ${
                selectedCategory === cat.id
                  ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                  : 'bg-slate-900/80 text-slate-400 hover:text-white hover:bg-slate-700 border border-slate-700/60'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>
      </div>

      {/* Resources Table / List */}
      {loading && !data ? (
        <div className="bg-slate-800/40 border border-slate-700 rounded-xl p-16 text-center space-y-3">
          <RefreshCw className="w-8 h-8 text-blue-400 animate-spin mx-auto" />
          <div className="text-sm font-semibold text-white">Inspecting GCP Resources...</div>
          <p className="text-xs text-slate-400">Querying storage, Vertex AI endpoints, training jobs, and IAM configurations.</p>
        </div>
      ) : filteredResources.length === 0 ? (
        <div className="bg-slate-800/40 border border-dashed border-slate-700 rounded-xl p-12 text-center space-y-3">
          <Cloud className="w-10 h-10 text-slate-500 mx-auto" />
          <div className="text-slate-300 font-semibold text-sm">No Matching Resources Found</div>
          <p className="text-xs text-slate-500 max-w-md mx-auto">
            Try adjusting your search query or category filter.
          </p>
          <button
            onClick={() => { setSearchTerm(''); setSelectedCategory('all'); }}
            className="px-3 py-1.5 bg-slate-700 hover:bg-slate-600 text-xs font-semibold text-white rounded-lg transition"
          >
            Reset Filters
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          {filteredResources.map((resource) => {
            const Icon = SERVICE_ICONS[resource.service] || Cloud;
            return (
              <div
                key={resource.id}
                className="bg-slate-800/70 border border-slate-700 hover:border-slate-600 rounded-xl p-4 transition shadow-md flex flex-col md:flex-row md:items-center justify-between gap-4"
              >
                {/* Left: Icon, Service, and Name */}
                <div className="flex items-start gap-3.5 flex-1 min-w-0">
                  <div className="w-10 h-10 rounded-xl bg-slate-900 border border-slate-700 flex items-center justify-center text-cyan-400 shrink-0 mt-0.5">
                    <Icon className="w-5 h-5" />
                  </div>

                  <div className="space-y-1 min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-xs font-bold text-white tracking-wide">
                        {resource.service}
                      </span>
                      <span className="text-[10px] uppercase font-semibold text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-700">
                        {resource.type}
                      </span>
                      <span className="text-[10px] text-slate-500 font-medium">
                        • {resource.category}
                      </span>
                    </div>

                    <div className="flex items-center gap-2 min-w-0">
                      <div 
                        className="text-xs font-mono font-bold text-cyan-300 truncate max-w-xl"
                        title={resource.name}
                      >
                        {resource.name}
                      </div>
                      <button
                        onClick={() => copyToClipboard(resource.resource_uri || resource.name, resource.id)}
                        className="text-slate-500 hover:text-white transition p-0.5 shrink-0"
                        title="Copy Resource URI / Name"
                      >
                        {copiedId === resource.id ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                    </div>

                    <p className="text-[11px] text-slate-300 leading-snug">
                      {resource.role}
                    </p>

                    <div className="text-[11px] text-slate-400 flex items-center gap-1.5 pt-0.5">
                      <span className="text-slate-500">Status detail:</span>
                      <span className="text-slate-300">{resource.status_detail}</span>
                    </div>
                  </div>
                </div>

                {/* Right: Status and GCP Console Link */}
                <div className="flex items-center justify-between md:justify-end gap-3 shrink-0 pt-2 md:pt-0 border-t md:border-t-0 border-slate-700/60">
                  <div className="flex flex-col items-start md:items-end gap-1">
                    {getStatusBadge(resource.status)}
                  </div>

                  <a
                    href={resource.console_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 bg-blue-600/20 hover:bg-blue-600/30 text-blue-300 hover:text-blue-200 border border-blue-500/40 hover:border-blue-400 text-xs font-semibold px-3.5 py-2 rounded-lg transition shadow-sm whitespace-nowrap"
                  >
                    <span>Open in GCP Console</span>
                    <ExternalLink className="w-3.5 h-3.5 text-blue-400" />
                  </a>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
