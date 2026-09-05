import React, { useState, useEffect } from 'react';
import { 
  fetchBuckets, fetchProjects, createProject, fetchProjectStatus 
} from './api';

import Header from './components/Header';
import CollapsibleLogPanel from './components/CollapsibleLogPanel';
import OverviewTab from './components/OverviewTab';
import ConfigForm from './components/ConfigForm';
import DatasetTab from './components/DatasetTab';
import TeacherTab from './components/TeacherTab';
import CostTab from './components/CostTab';
import TrainingTab from './components/TrainingTab';
import EvaluationTab from './components/EvaluationTab';
import DeploymentTab from './components/DeploymentTab';
import HistoryTab from './components/HistoryTab';

import { 
  Layers, Settings2, Database, Sparkles, 
  Calculator, Cpu, Award, Rocket, History 
} from 'lucide-react';

const TABS = [
  { id: 'overview', label: 'Pipeline Overview', icon: Layers },
  { id: 'config', label: '1. Config Form', icon: Settings2 },
  { id: 'dataset', label: '2. Dataset Split', icon: Database },
  { id: 'teacher', label: '3. Teacher CoT', icon: Sparkles },
  { id: 'cost', label: '4. Hardware Probe', icon: Calculator },
  { id: 'training', label: '5. Model training', icon: Cpu },
  { id: 'evaluation', label: '6. 3-Tier Eval', icon: Award },
  { id: 'deployment', label: '7. vLLM Deploy', icon: Rocket },
  { id: 'history', label: 'Audit History', icon: History },
];

export default function App() {
  const [buckets, setBuckets] = useState(['distillfw-workspaces']);
  const [selectedBucket, setSelectedBucket] = useState('distillfw-workspaces');
  const [projects, setProjects] = useState([]);
  const [selectedProject, setSelectedProject] = useState('');
  const [status, setStatus] = useState('UNINITIALIZED');
  const [statusDetail, setStatusDetail] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  // Load buckets on mount
  useEffect(() => {
    const initBuckets = async () => {
      try {
        const bList = await fetchBuckets();
        if (Array.isArray(bList) && bList.length > 0) {
          setBuckets(bList);
          if (!bList.includes(selectedBucket)) {
            setSelectedBucket(bList[0]);
          }
        }
      } catch (e) {
        // use default fallback
      }
    };
    initBuckets();
  }, []);

  // Load projects whenever selectedBucket changes
  const loadProjects = async () => {
    try {
      const pList = await fetchProjects(selectedBucket);
      if (Array.isArray(pList)) {
        setProjects(pList);
        if (pList.length > 0) {
          if (!selectedProject || !pList.includes(selectedProject)) {
            setSelectedProject(pList[0]);
          }
        } else {
          setSelectedProject('');
        }
      }
    } catch (e) {
      setProjects([]);
    }
  };

  useEffect(() => {
    if (selectedBucket) loadProjects();
  }, [selectedBucket]);

  // Load status whenever selectedProject or selectedBucket changes
  const loadStatus = async () => {
    if (!selectedBucket || !selectedProject) {
      setStatus('UNINITIALIZED');
      setStatusDetail('');
      return;
    }
    try {
      const res = await fetchProjectStatus(selectedBucket, selectedProject);
      if (res?.status) {
        setStatus(res.status);
        setStatusDetail(res.detail || '');
      }
    } catch (e) {
      setStatus('UNINITIALIZED');
    }
  };

  useEffect(() => {
    if (selectedProject) {
      loadStatus();
      const interval = setInterval(loadStatus, 4000);
      return () => clearInterval(interval);
    }
  }, [selectedBucket, selectedProject]);

  const handleCreateProject = async (newId, description) => {
    try {
      await createProject(newId, selectedBucket, description);
      await loadProjects();
      setSelectedProject(newId);
    } catch (e) {
      console.error(e);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-sans">
      {/* Header with Comboboxes and Status */}
      <Header
        buckets={buckets}
        selectedBucket={selectedBucket}
        onSelectBucket={setSelectedBucket}
        projects={projects}
        selectedProject={selectedProject}
        onSelectProject={setSelectedProject}
        onCreateProject={handleCreateProject}
        status={status}
        statusDetail={statusDetail}
        onRefresh={loadStatus}
      />

      {/* Main Tab Navigation */}
      <nav className="bg-slate-900/60 border-b border-slate-800 sticky top-16 z-20 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-1 overflow-x-auto py-2 scrollbar-none">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition cursor-pointer ${
                    isActive
                      ? 'bg-blue-600 text-white shadow-md shadow-blue-600/30'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className="w-3.5 h-3.5" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {!selectedProject ? (
          <div className="bg-slate-800/40 border border-slate-700 rounded-2xl p-12 text-center max-w-lg mx-auto mt-12 space-y-4">
            <Layers className="w-12 h-12 text-blue-400 mx-auto" />
            <h2 className="text-base font-bold text-white">Select or Create a Distillation Project Workspace</h2>
            <p className="text-xs text-slate-400">
              No project workspace selected. Use the folder selector in the header to pick an existing project under <code className="text-cyan-300 font-mono">gs://{selectedBucket}/</code> or create a new one.
            </p>
          </div>
        ) : (
          <>
            {activeTab === 'overview' && (
              <OverviewTab
                bucket={selectedBucket}
                projectId={selectedProject}
                status={status}
                onNavigate={setActiveTab}
              />
            )}
            {activeTab === 'config' && (
              <ConfigForm
                bucket={selectedBucket}
                projectId={selectedProject}
                onSaved={loadStatus}
              />
            )}
            {activeTab === 'dataset' && (
              <DatasetTab
                bucket={selectedBucket}
                projectId={selectedProject}
                onStatusChange={loadStatus}
              />
            )}
            {activeTab === 'teacher' && (
              <TeacherTab
                bucket={selectedBucket}
                projectId={selectedProject}
                onStatusChange={loadStatus}
              />
            )}
            {activeTab === 'cost' && (
              <CostTab
                bucket={selectedBucket}
                projectId={selectedProject}
                onStatusChange={loadStatus}
              />
            )}
            {activeTab === 'training' && (
              <TrainingTab
                bucket={selectedBucket}
                projectId={selectedProject}
                onStatusChange={loadStatus}
              />
            )}
            {activeTab === 'evaluation' && (
              <EvaluationTab
                bucket={selectedBucket}
                projectId={selectedProject}
                onStatusChange={loadStatus}
              />
            )}
            {activeTab === 'deployment' && (
              <DeploymentTab
                bucket={selectedBucket}
                projectId={selectedProject}
                onStatusChange={loadStatus}
              />
            )}
            {activeTab === 'history' && (
              <HistoryTab
                bucket={selectedBucket}
                projectId={selectedProject}
              />
            )}
          </>
        )}
      </main>

      {/* Collapsible Operations Bottom Panel */}
      <CollapsibleLogPanel projectId={selectedProject} />
    </div>
  );
}
