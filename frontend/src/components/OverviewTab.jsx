import React from 'react';
import { 
  FolderGit2, Database, Sparkles, Calculator, 
  Cpu, Award, Rocket, ArrowRight, CheckCircle2, CircleDot, Clock 
} from 'lucide-react';

const STAGES = [
  { id: 'config', name: '1. Project & Config', icon: FolderGit2, desc: 'Master config.yaml and GCS workspace initialization' },
  { id: 'dataset', name: '2. Dataset & Split', icon: Database, desc: 'Train (80%), Val (10%), Test (10%) split validation' },
  { id: 'teacher', name: '3. Teacher Inference', icon: Sparkles, desc: 'Gemini reasoning & CoT knowledge extraction' },
  { id: 'cost', name: '4. Hardware Probe', icon: Calculator, desc: 'Peak VRAM test, step duration, and budget scorecard' },
  { id: 'training', name: '5. Custom Training', icon: Cpu, desc: 'Vertex AI CustomJob PEFT QLoRA with live telemetry' },
  { id: 'evaluation', name: '6. 3-Tier Evaluation', icon: Award, desc: 'Lexical, Gemini judge, and operational benchmarks' },
  { id: 'deployment', name: '7. vLLM Deployment', icon: Rocket, desc: 'High-throughput online prediction serving endpoint' },
];

export default function OverviewTab({ bucket, projectId, status, onNavigate }) {
  const getStageState = (stageId) => {
    switch (stageId) {
      case 'config':
        return status !== 'UNINITIALIZED';
      case 'dataset':
        return ['DATASET_READY', 'TEACHER_INFERENCE_RUNNING', 'TEACHER_INFERENCE_DONE', 'COST_ESTIMATED', 'TRAINING_RUNNING', 'TRAINING_COMPLETED', 'EVALUATING', 'EVALUATED', 'DEPLOYED'].includes(status);
      case 'teacher':
        return ['TEACHER_INFERENCE_DONE', 'COST_ESTIMATED', 'TRAINING_RUNNING', 'TRAINING_COMPLETED', 'EVALUATING', 'EVALUATED', 'DEPLOYED'].includes(status);
      case 'cost':
        return ['COST_ESTIMATED', 'TRAINING_RUNNING', 'TRAINING_COMPLETED', 'EVALUATING', 'EVALUATED', 'DEPLOYED'].includes(status);
      case 'training':
        return ['TRAINING_COMPLETED', 'EVALUATING', 'EVALUATED', 'DEPLOYED'].includes(status);
      case 'evaluation':
        return ['EVALUATED', 'DEPLOYED'].includes(status);
      case 'deployment':
        return status === 'DEPLOYED';
      default:
        return false;
    }
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto pb-16">
      {/* Workspace Banner */}
      <div className="bg-gradient-to-r from-blue-950/60 via-slate-900 to-indigo-950/60 border border-blue-800/50 rounded-2xl p-6 shadow-xl">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="text-[11px] font-bold uppercase tracking-wider text-blue-400">Current Project Workspace</div>
            <h1 className="text-2xl font-black text-white mt-1">{projectId}</h1>
            <p className="text-xs font-mono text-cyan-300 mt-1">
              gs://{bucket}/{projectId}/
            </p>
          </div>

          <button
            onClick={() => onNavigate('config')}
            className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold px-4 py-2 rounded-lg self-start sm:self-center shadow-lg shadow-blue-500/20"
          >
            Edit Configuration <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* 7-Stage Pipeline Flow */}
      <div className="bg-slate-800/60 border border-slate-700 rounded-xl p-6 space-y-4">
        <h2 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2 border-b border-slate-700 pb-3">
          <CircleDot className="w-4 h-4 text-cyan-400" /> End-to-End Distillation Lifecycle Pipeline
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
          {STAGES.map((s, idx) => {
            const isCompleted = getStageState(s.id);
            const Icon = s.icon;
            return (
              <div
                key={s.id}
                onClick={() => onNavigate(s.id)}
                className={`p-4 rounded-xl border transition cursor-pointer flex flex-col justify-between ${
                  isCompleted
                    ? 'bg-slate-900/80 border-slate-700 hover:border-blue-500 shadow-md'
                    : 'bg-slate-900/40 border-slate-800/80 opacity-70 hover:opacity-100 hover:border-slate-700'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between mb-2">
                    <div className="w-8 h-8 rounded-lg bg-slate-800 flex items-center justify-center text-cyan-400">
                      <Icon className="w-4 h-4" />
                    </div>
                    {isCompleted ? (
                      <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-emerald-400 bg-emerald-950/60 border border-emerald-800 px-2 py-0.5 rounded-full">
                        <CheckCircle2 className="w-3 h-3" /> Done
                      </span>
                    ) : (
                      <span className="flex items-center gap-1 text-[10px] font-bold uppercase text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full">
                        Pending
                      </span>
                    )}
                  </div>

                  <h3 className="text-xs font-bold text-white">{s.name}</h3>
                  <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">{s.desc}</p>
                </div>

                <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-[11px] text-blue-400 font-semibold">
                  <span>Open Stage</span>
                  <ArrowRight className="w-3 h-3" />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
