/** API client for DistillFW backend */

const API_BASE = '/api';

export async function fetchBuckets() {
  const res = await fetch(`${API_BASE}/workspaces/buckets`);
  return res.json();
}

export async function fetchProjects(bucket) {
  const res = await fetch(`${API_BASE}/workspaces/projects?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function createProject(project_id, bucket, description) {
  const res = await fetch(`${API_BASE}/workspaces/projects`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ project_id, bucket, description })
  });
  return res.json();
}

export async function fetchProjectStatus(bucket, project_id) {
  const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(project_id)}/status?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function fetchProjectHistory(bucket, project_id) {
  const res = await fetch(`${API_BASE}/workspaces/${encodeURIComponent(project_id)}/history?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function fetchConfig(bucket, project_id) {
  const res = await fetch(`${API_BASE}/config/${encodeURIComponent(project_id)}?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function saveConfig(bucket, project_id, config) {
  const res = await fetch(`${API_BASE}/config/${encodeURIComponent(project_id)}?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config)
  });
  return res.json();
}

export async function uploadDataset(bucket, project_id, content) {
  const res = await fetch(`${API_BASE}/dataset/${encodeURIComponent(project_id)}/upload?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content })
  });
  return res.json();
}

export async function splitDataset(bucket, project_id, splitParams) {
  const res = await fetch(`${API_BASE}/dataset/${encodeURIComponent(project_id)}/split?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(splitParams)
  });
  return res.json();
}

export async function fetchDatasetSummary(bucket, project_id) {
  const res = await fetch(`${API_BASE}/dataset/${encodeURIComponent(project_id)}/summary?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function clearDataset(bucket, project_id) {
  const res = await fetch(`${API_BASE}/dataset/${encodeURIComponent(project_id)}/clear?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function runTeacherInference(bucket, project_id, limit = null) {
  const res = await fetch(`${API_BASE}/teacher/${encodeURIComponent(project_id)}/run?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ limit })
  });
  return res.json();
}

export async function stopTeacherInference(bucket, project_id) {
  const res = await fetch(`${API_BASE}/teacher/${encodeURIComponent(project_id)}/stop?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function clearTeacherInferences(bucket, project_id) {
  const res = await fetch(`${API_BASE}/teacher/${encodeURIComponent(project_id)}/clear?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function fetchTeacherStatus(bucket, project_id, limit = 10) {
  const res = await fetch(`${API_BASE}/teacher/${encodeURIComponent(project_id)}/status?bucket=${encodeURIComponent(bucket)}&limit=${limit}`);
  return res.json();
}

export async function fetchTeacherRetries(bucket, project_id) {
  const res = await fetch(`${API_BASE}/teacher/${encodeURIComponent(project_id)}/retries?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function runCostProbe(bucket, project_id) {
  const res = await fetch(`${API_BASE}/cost/${encodeURIComponent(project_id)}/probe?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function stopCostProbe(bucket, project_id) {
  const res = await fetch(`${API_BASE}/cost/${encodeURIComponent(project_id)}/stop?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function clearCostEstimate(bucket, project_id) {
  const res = await fetch(`${API_BASE}/cost/${encodeURIComponent(project_id)}/clear?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function fetchCostEstimate(bucket, project_id) {
  const res = await fetch(`${API_BASE}/cost/${encodeURIComponent(project_id)}/estimate?bucket=${encodeURIComponent(bucket)}`);
  if (!res.ok) return null;
  return res.json();
}

export async function startTraining(bucket, project_id, dry_run = false) {
  const res = await fetch(`${API_BASE}/training/${encodeURIComponent(project_id)}/start?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dry_run })
  });
  return res.json();
}

export async function stopTraining(bucket, project_id) {
  const res = await fetch(`${API_BASE}/training/${encodeURIComponent(project_id)}/stop?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function clearTraining(bucket, project_id) {
  const res = await fetch(`${API_BASE}/training/${encodeURIComponent(project_id)}/clear?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function fetchTrainingMetrics(bucket, project_id) {
  const res = await fetch(`${API_BASE}/training/${encodeURIComponent(project_id)}/metrics?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function fetchTrainingHeartbeat(bucket, project_id) {
  const res = await fetch(`${API_BASE}/training/${encodeURIComponent(project_id)}/heartbeat?bucket=${encodeURIComponent(bucket)}`);
  return res.json();
}

export async function runEvaluation(bucket, project_id) {
  const res = await fetch(`${API_BASE}/evaluation/${encodeURIComponent(project_id)}/run?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function stopEvaluation(bucket, project_id) {
  const res = await fetch(`${API_BASE}/evaluation/${encodeURIComponent(project_id)}/stop?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function clearEvaluation(bucket, project_id) {
  const res = await fetch(`${API_BASE}/evaluation/${encodeURIComponent(project_id)}/clear?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function fetchEvaluationResults(bucket, project_id) {
  const res = await fetch(`${API_BASE}/evaluation/${encodeURIComponent(project_id)}/results?bucket=${encodeURIComponent(bucket)}`);
  if (!res.ok) return null;
  return res.json();
}

export async function deployEndpoint(bucket, project_id) {
  const res = await fetch(`${API_BASE}/deployment/${encodeURIComponent(project_id)}/deploy?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function stopDeployment(bucket, project_id) {
  const res = await fetch(`${API_BASE}/deployment/${encodeURIComponent(project_id)}/stop?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function clearDeployment(bucket, project_id) {
  const res = await fetch(`${API_BASE}/deployment/${encodeURIComponent(project_id)}/clear?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST'
  });
  return res.json();
}

export async function fetchDeploymentStatus(bucket, project_id) {
  const res = await fetch(`${API_BASE}/deployment/${encodeURIComponent(project_id)}/status?bucket=${encodeURIComponent(bucket)}`);
  if (!res.ok) return null;
  return res.json();
}

export async function predictEndpoint(bucket, project_id, prompt, temperature = 0.2) {
  const res = await fetch(`${API_BASE}/deployment/${encodeURIComponent(project_id)}/predict?bucket=${encodeURIComponent(bucket)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, temperature })
  });
  return res.json();
}

export async function fetchLogs(project_id = null) {
  const url = project_id ? `${API_BASE}/logs?project_id=${encodeURIComponent(project_id)}` : `${API_BASE}/logs`;
  const res = await fetch(url);
  return res.json();
}

export async function clearLogs() {
  const res = await fetch(`${API_BASE}/logs/clear`, { method: 'POST' });
  return res.json();
}

