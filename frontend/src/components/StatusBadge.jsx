import React from 'react';
import { 
  AlertCircle, CheckCircle2, Clock, Cpu, 
  Database, Play, Rocket, Sparkles, Award 
} from 'lucide-react';

const STATUS_CONFIGS = {
  UNINITIALIZED: {
    label: 'Uninitialized',
    bg: 'bg-slate-800 text-slate-400 border-slate-700',
    icon: AlertCircle
  },
  CONFIGURED: {
    label: 'Configured',
    bg: 'bg-blue-950 text-blue-300 border-blue-800',
    icon: Clock
  },
  DATASET_READY: {
    label: 'Dataset Ready',
    bg: 'bg-cyan-950 text-cyan-300 border-cyan-800',
    icon: Database
  },
  TEACHER_INFERENCE_RUNNING: {
    label: 'Teacher Inference Running',
    bg: 'bg-purple-950 text-purple-300 border-purple-800 animate-pulse',
    icon: Sparkles
  },
  TEACHER_INFERENCE_DONE: {
    label: 'Teacher Inferences Ready',
    bg: 'bg-purple-950 text-purple-300 border-purple-800',
    icon: Sparkles
  },
  COST_ESTIMATED: {
    label: 'Cost Estimated',
    bg: 'bg-amber-950 text-amber-300 border-amber-800',
    icon: CheckCircle2
  },
  TRAINING_RUNNING: {
    label: 'Training in Progress',
    bg: 'bg-indigo-950 text-indigo-300 border-indigo-800 animate-pulse',
    icon: Cpu
  },
  TRAINING_COMPLETED: {
    label: 'Training Completed',
    bg: 'bg-teal-950 text-teal-300 border-teal-800',
    icon: CheckCircle2
  },
  EVALUATING: {
    label: 'Evaluating Test Split',
    bg: 'bg-orange-950 text-orange-300 border-orange-800 animate-pulse',
    icon: Play
  },
  EVALUATED: {
    label: 'Evaluated',
    bg: 'bg-emerald-950 text-emerald-300 border-emerald-800',
    icon: Award
  },
  DEPLOYING: {
    label: 'Deploying vLLM Endpoint',
    bg: 'bg-emerald-950 text-emerald-300 border-emerald-700 animate-pulse',
    icon: Rocket
  },
  DEPLOYED: {
    label: 'Deployed (vLLM Live)',
    bg: 'bg-green-950 text-green-300 border-green-700',
    icon: Rocket
  }
};

export default function StatusBadge({ status, detail }) {
  const config = STATUS_CONFIGS[status] || STATUS_CONFIGS.UNINITIALIZED;
  const Icon = config.icon;

  return (
    <div className="flex items-center gap-2">
      <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${config.bg}`}>
        <Icon className="w-3.5 h-3.5" />
        {config.label}
      </span>
      {detail && (
        <span className="text-xs text-slate-400 hidden md:inline">
          ({detail})
        </span>
      )}
    </div>
  );
}
