"use strict";

(() => {
  const state = {
    rootUri: "",
    tasks: [],
    selectedTaskUri: "",
    taskDetails: null,
    currentPanelUri: "",
  };

  // DOM Elements
  const gcsRootInput = document.getElementById("gcs-root-input");
  const loadTasksBtn = document.getElementById("load-tasks-btn");
  const taskSelect = document.getElementById("task-select");
  const tasksCountBadge = document.getElementById("tasks-count-badge");
  const refreshTaskBtn = document.getElementById("refresh-task-btn");
  const aboutDistillationBtn = document.getElementById("about-distillation-btn");
  const statusBanner = document.getElementById("status-banner");
  const emptyState = document.getElementById("empty-state");
  const taskWorkspace = document.getElementById("task-workspace");

  const taskOverviewHeading = document.getElementById("task-overview-heading");
  const taskOverallStatus = document.getElementById("task-overall-status");
  const taskDescription = document.getElementById("task-description");
  const openConfigBtn = document.getElementById("open-config-btn");
  const taskMetadataStrip = document.getElementById("task-metadata-strip");
  const stagesTableBody = document.getElementById("stages-table-body");
  const gcpResourcesGrid = document.getElementById("gcp-resources-grid");
  const datasetsCount = document.getElementById("datasets-count");
  const datasetsList = document.getElementById("datasets-list");
  const logsCount = document.getElementById("logs-count");
  const logsList = document.getElementById("logs-list");
  const openInferencesBtn = document.getElementById("open-inferences-btn");
  const evaluationContent = document.getElementById("evaluation-content");

  // Emerging Dialog Panel Elements
  const emergingPanel = document.getElementById("emerging-panel");
  const panelKicker = document.getElementById("panel-kicker");
  const panelTitle = document.getElementById("panel-title");
  const panelUri = document.getElementById("panel-uri");
  const panelCopyUriBtn = document.getElementById("panel-copy-uri-btn");
  const panelCloseBtn = document.getElementById("panel-close-btn");
  const panelToolbar = document.getElementById("panel-toolbar");
  const panelBody = document.getElementById("panel-body");

  function showError(message) {
    if (!message) {
      statusBanner.classList.add("hidden");
      statusBanner.textContent = "";
      return;
    }
    statusBanner.textContent = message;
    statusBanner.classList.remove("hidden");
  }

  function escapeHtml(value) {
    if (value === null || value === undefined) return "";
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function formatStatusBadge(status) {
    const clean = String(status || "PENDING").toUpperCase();
    return `<span class="status-badge status-${escapeHtml(clean)}">${escapeHtml(clean)}</span>`;
  }

  async function fetchJson(url) {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    return data;
  }

  function openEmergingPanel({ kicker, title, uri, toolbarHtml = "", bodyHtml = "" }) {
    state.currentPanelUri = uri || "";
    panelKicker.textContent = kicker || "Inspector";
    panelTitle.textContent = title || "";
    panelUri.textContent = uri || "";
    panelUri.classList.toggle("hidden", !uri);
    panelCopyUriBtn.classList.toggle("hidden", !uri);
    if (toolbarHtml) {
      panelToolbar.innerHTML = toolbarHtml;
      panelToolbar.classList.remove("hidden");
    } else {
      panelToolbar.innerHTML = "";
      panelToolbar.classList.add("hidden");
    }
    panelBody.innerHTML = bodyHtml;
    if (!emergingPanel.open) {
      emergingPanel.showModal();
    }
  }

  panelCloseBtn.addEventListener("click", () => {
    emergingPanel.close();
  });

  emergingPanel.addEventListener("click", (event) => {
    const rect = emergingPanel.getBoundingClientRect();
    const clickedInDialog =
      rect.top <= event.clientY &&
      event.clientY <= rect.top + rect.height &&
      rect.left <= event.clientX &&
      event.clientX <= rect.left + rect.width;
    if (!clickedInDialog) {
      emergingPanel.close();
    }
  });

  panelCopyUriBtn.addEventListener("click", async () => {
    if (!state.currentPanelUri) return;
    try {
      await navigator.clipboard.writeText(state.currentPanelUri);
      const prev = panelCopyUriBtn.textContent;
      panelCopyUriBtn.textContent = "Copied!";
      setTimeout(() => {
        panelCopyUriBtn.textContent = prev;
      }, 1200);
    } catch (_) {}
  });

  // 1. Load Tasks from GCS Root Path
  async function loadTasksFromRoot(autoSelectFirst = true) {
    const rootUri = gcsRootInput.value.trim();
    if (!rootUri) {
      showError("Please enter a GCS root path (e.g., gs://distillfw-storage/tasks).");
      return;
    }
    showError("");
    loadTasksBtn.disabled = true;
    loadTasksBtn.textContent = "Loading...";

    try {
      const payload = await fetchJson(`/api/tasks?root_uri=${encodeURIComponent(rootUri)}`);
      state.rootUri = payload.root_uri;
      state.tasks = payload.tasks || [];
      tasksCountBadge.textContent = String(state.tasks.length);

      taskSelect.innerHTML = `<option value="">Select a distillation task (${state.tasks.length} available)...</option>`;
      for (const t of state.tasks) {
        const opt = document.createElement("option");
        opt.value = t.task_uri;
        const stageInfo = t.current_stage ? ` • stage: ${t.current_stage}` : "";
        opt.textContent = `${t.task_id} [${t.status}]${stageInfo}`;
        taskSelect.appendChild(opt);
      }

      taskSelect.disabled = state.tasks.length === 0;
      if (state.tasks.length === 0) {
        emptyState.classList.remove("hidden");
        taskWorkspace.classList.add("hidden");
        refreshTaskBtn.disabled = true;
        return;
      }

      const targetUri =
        state.selectedTaskUri && state.tasks.some((t) => t.task_uri === state.selectedTaskUri)
          ? state.selectedTaskUri
          : autoSelectFirst
          ? state.tasks[0].task_uri
          : "";

      if (targetUri) {
        taskSelect.value = targetUri;
        await selectTask(targetUri);
      }
    } catch (err) {
      showError(`Failed to list tasks under '${rootUri}': ${err.message}`);
    } finally {
      loadTasksBtn.disabled = false;
      loadTasksBtn.textContent = "Load Tasks";
    }
  }

  // 2. Load Selected Task Details
  async function selectTask(taskUri) {
    if (!taskUri) {
      state.selectedTaskUri = "";
      state.taskDetails = null;
      emptyState.classList.remove("hidden");
      taskWorkspace.classList.add("hidden");
      refreshTaskBtn.disabled = true;
      return;
    }

    showError("");
    state.selectedTaskUri = taskUri;
    refreshTaskBtn.disabled = false;
    refreshTaskBtn.textContent = "Refreshing...";

    try {
      const details = await fetchJson(`/api/task/details?task_uri=${encodeURIComponent(taskUri)}&refresh=true`);
      state.taskDetails = details;
      renderTaskWorkspace(details);
    } catch (err) {
      showError(`Failed to load task details for '${taskUri}': ${err.message}`);
    } finally {
      refreshTaskBtn.textContent = "Refresh";
    }
  }

  function renderTaskWorkspace(details) {
    emptyState.classList.add("hidden");
    taskWorkspace.classList.remove("hidden");

    taskOverviewHeading.textContent = details.task_id;
    taskOverallStatus.className = `status-badge status-${details.status}`;
    taskOverallStatus.textContent = details.status;
    taskDescription.textContent =
      (details.config_summary && details.config_summary.description) ||
      `Task URI: ${details.task_uri}`;

    // Metadata strip
    const summary = details.config_summary || {};
    const metaItems = [
      { label: "GCS Task URI", value: details.task_uri, mono: true },
      { label: "Current Stage", value: details.current_stage || "None" },
      { label: "Teacher Model", value: summary.teacher_model || "-", mono: true },
      { label: "Student Model", value: summary.student_model || "-", mono: true },
      { label: "Algorithm / Paradigm", value: `${summary.algorithm || "-"} (${summary.paradigm || "-"})`, mono: true },
      { label: "Last Updated (UTC)", value: details.updated_at || "-", mono: true },
    ];
    taskMetadataStrip.innerHTML = metaItems
      .map(
        (m) => `
        <div class="meta-item">
          <span class="meta-label">${escapeHtml(m.label)}</span>
          <span class="meta-value ${m.mono ? "mono" : ""}">${escapeHtml(m.value)}</span>
        </div>`
      )
      .join("");

    // Stages Table
    stagesTableBody.innerHTML = (details.stages || [])
      .map((st) => {
        const cursorStr =
          st.progress_cursor && Object.keys(st.progress_cursor).length > 0
            ? JSON.stringify(st.progress_cursor)
            : "—";
        const errHtml = st.error_message
          ? `<div style="margin-top:6px;color:var(--status-failed-fg);font-weight:600;">Error: ${escapeHtml(
              st.error_message
            )}</div>`
          : "";
        return `
          <tr>
            <td class="mono" style="font-weight:600;">${escapeHtml(st.stage)}</td>
            <td>${formatStatusBadge(st.status)}</td>
            <td class="mono">${escapeHtml(st.started_at || "—")}</td>
            <td class="mono">${escapeHtml(st.completed_at || "—")}</td>
            <td>
              <div class="mono" style="font-size:12px;word-break:break-all;">${escapeHtml(cursorStr)}</div>
              ${errHtml}
            </td>
          </tr>`;
      })
      .join("");

    // GCP Resources Section
    renderGcpResources(details.gcp_resources || []);

    // Datasets List
    const datasets = details.datasets || [];
    datasetsCount.textContent = String(datasets.length);
    if (datasets.length === 0) {
      datasetsList.innerHTML = `<div style="color:var(--text-muted);padding:12px 0;">No dataset artifacts found yet in this task workspace.</div>`;
    } else {
      datasetsList.innerHTML = datasets
        .map(
          (ds) => `
          <button type="button" class="artifact-item" data-dataset-uri="${escapeHtml(ds.uri)}" data-dataset-name="${escapeHtml(ds.rel_path)}">
            <div class="artifact-info">
              <span class="artifact-name mono">${escapeHtml(ds.rel_path)}</span>
              <span class="artifact-sub">${escapeHtml(ds.stage_label)} &bull; ${escapeHtml(ds.uri)}</span>
            </div>
            <span class="artifact-badge mono">${escapeHtml(ds.format.toUpperCase())}</span>
          </button>`
        )
        .join("");

      datasetsList.querySelectorAll("[data-dataset-uri]").forEach((btn) => {
        btn.addEventListener("click", () => {
          openDatasetInPanel(btn.getAttribute("data-dataset-uri"), btn.getAttribute("data-dataset-name"));
        });
      });
    }

    // Log Files List
    const logs = details.logs || [];
    logsCount.textContent = String(logs.length);
    if (logs.length === 0) {
      logsList.innerHTML = `<div style="color:var(--text-muted);padding:12px 0;">No log files found under <code class="mono">logs/</code> yet.</div>`;
    } else {
      logsList.innerHTML = logs
        .map(
          (log) => `
          <button type="button" class="artifact-item" data-log-uri="${escapeHtml(log.uri)}" data-log-name="${escapeHtml(log.name)}">
            <div class="artifact-info">
              <span class="artifact-name mono">${escapeHtml(log.name)}</span>
              <span class="artifact-sub mono">${escapeHtml(log.uri)}</span>
            </div>
            <span class="artifact-badge">View Log</span>
          </button>`
        )
        .join("");

      logsList.querySelectorAll("[data-log-uri]").forEach((btn) => {
        btn.addEventListener("click", () => {
          openLogInPanel(btn.getAttribute("data-log-uri"), btn.getAttribute("data-log-name"));
        });
      });
    }

    // Evaluation Metrics & Side-by-Side Inferences Button
    openInferencesBtn.disabled = !details.has_predictions;
    renderEvaluationMetrics(details.scorecard, details.has_predictions);
  }

  function renderGcpResources(resources) {
    gcpResourcesGrid.innerHTML = resources
      .map((res) => {
        const consoleBtn = res.console_url
          ? `<a href="${escapeHtml(res.console_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary btn-sm link-btn">Open in GCP Console &nearr;</a>`
          : "";
        const logsBtn = res.logs_url
          ? `<a href="${escapeHtml(res.logs_url)}" target="_blank" rel="noopener noreferrer" class="btn btn-secondary btn-sm link-btn">Cloud Logs &nearr;</a>`
          : "";
        return `
          <div class="gcp-resource-card">
            <div>
              <div class="gcp-res-top">
                <span class="gcp-res-category">${escapeHtml(res.category)}</span>
                ${formatStatusBadge(res.status)}
              </div>
              <h3 class="gcp-res-label">${escapeHtml(res.label)}</h3>
            </div>
            <div class="gcp-res-uri mono">${escapeHtml(res.resource_id)}</div>
            <p class="gcp-res-desc">${escapeHtml(res.description || "")}</p>
            <div class="gcp-res-actions">
              <button type="button" class="btn btn-secondary btn-sm" data-copy-text="${escapeHtml(res.resource_id)}">Copy ID / URI</button>
              ${consoleBtn}
              ${logsBtn}
            </div>
          </div>`;
      })
      .join("");

    gcpResourcesGrid.querySelectorAll("[data-copy-text]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = btn.getAttribute("data-copy-text") || "";
        try {
          await navigator.clipboard.writeText(text);
          const orig = btn.textContent;
          btn.textContent = "Copied!";
          setTimeout(() => {
            btn.textContent = orig;
          }, 1200);
        } catch (_) {}
      });
    });
  }

  function buildSplitEvalTableHtml(splitTitle, splitInfo) {
    if (!splitInfo) return "";
    const beforeM = splitInfo.before_training || {};
    const afterM = splitInfo.after_training || {};
    const deltaM = splitInfo.improvement || {};
    const nSamples = splitInfo.num_samples || 0;

    const rows = [];
    const lexicalSpecs = [
      ["exact_match", "Exact Match", 4, true],
      ["rouge1", "ROUGE-1", 4, true],
      ["rouge2", "ROUGE-2", 4, true],
      ["bleu", "BLEU-4", 4, true],
    ];

    for (const [key, label, decimals, higherIsBetter] of lexicalSpecs) {
      const bVal = (beforeM.lexical_metrics || {})[key];
      const aVal = (afterM.lexical_metrics || {})[key];
      const dVal = (deltaM.lexical_metrics || {})[key];
      if (bVal !== undefined && aVal !== undefined) {
        rows.push({ label, bVal, aVal, dVal, decimals, higherIsBetter });
      }
    }

    if (beforeM.llm_judge && afterM.llm_judge) {
      const judgeSpecs = [
        ["mean_rubric_score", "LLM Judge Rubric Score (1–5)", 3, true],
        ["win_or_tie_rate_vs_teacher", "Win / Tie Rate vs. Teacher", 4, true],
      ];
      for (const [key, label, decimals, higherIsBetter] of judgeSpecs) {
        const bVal = beforeM.llm_judge[key];
        const aVal = afterM.llm_judge[key];
        const dVal = (deltaM.llm_judge || {})[key];
        if (bVal !== undefined && aVal !== undefined) {
          rows.push({ label, bVal, aVal, dVal, decimals, higherIsBetter });
        }
      }
    }

    if (beforeM.system_metrics && afterM.system_metrics) {
      const sysSpecs = [
        ["mean_output_tokens", "Output Length (Tokens)", 2, null],
        ["time_per_output_token_ms", "Time per Output Token (ms/tok)", 2, false],
        ["latency_mean_ms", "Mean Latency (ms)", 2, false],
        ["latency_p50_ms", "p50 Latency (ms)", 2, false],
        ["latency_p95_ms", "p95 Latency (ms)", 2, false],
      ];
      for (const [key, label, decimals, higherIsBetter] of sysSpecs) {
        const bVal = beforeM.system_metrics[key];
        const aVal = afterM.system_metrics[key];
        const dVal = (deltaM.system_metrics || {})[key];
        if (bVal !== undefined && aVal !== undefined) {
          rows.push({ label, bVal, aVal, dVal, decimals, higherIsBetter });
        }
      }
    }

    const trs = rows
      .map((r) => {
        const bFormatted = Number(r.bVal).toFixed(r.decimals);
        const aFormatted = Number(r.aVal).toFixed(r.decimals);
        const rawDelta = r.dVal !== undefined ? Number(r.dVal) : Number(r.aVal) - Number(r.bVal);
        const sign = rawDelta >= 0 ? "+" : "";
        const dFormatted = `${sign}${rawDelta.toFixed(r.decimals)}`;
        let pillClass = "";
        if (rawDelta !== 0 && r.higherIsBetter !== null && r.higherIsBetter !== undefined) {
          const isGood = r.higherIsBetter ? rawDelta >= 0 : rawDelta <= 0;
          pillClass = isGood ? "delta-pos" : "delta-neg";
        }
        return `
          <tr>
            <td style="font-weight:600;">${escapeHtml(r.label)}</td>
            <td class="num-col mono">${escapeHtml(bFormatted)}</td>
            <td class="num-col mono" style="font-weight:600;">${escapeHtml(aFormatted)}</td>
            <td class="num-col mono"><span class="delta-pill ${pillClass}">${escapeHtml(dFormatted)}</span></td>
          </tr>`;
      })
      .join("");

    return `
      <div class="eval-split-box">
        <h3>${escapeHtml(splitTitle)} <span class="count-pill">n = ${escapeHtml(nSamples)}</span></h3>
        <div class="table-wrapper">
          <table class="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th class="num-col">Before Training (Base)</th>
                <th class="num-col">After Training (Distilled)</th>
                <th class="num-col">Improvement (&Delta;)</th>
              </tr>
            </thead>
            <tbody>${trs}</tbody>
          </table>
        </div>
      </div>`;
  }

  function renderEvaluationMetrics(scorecard, hasPredictions) {
    if (!scorecard) {
      evaluationContent.innerHTML = `
        <div style="color:var(--text-muted);padding:12px 0;">
          Stage 4 (<code class="mono">model_evaluator</code>) has not produced <code class="mono">04_evaluation/scorecard.json</code> yet.
        </div>`;
      return;
    }

    const splits = scorecard.splits || {};
    const testHtml = buildSplitEvalTableHtml("Test Split (Held-Out Evaluation)", splits.test);
    const trainHtml = buildSplitEvalTableHtml("Train Split (Training Data Evaluation)", splits.train);

    evaluationContent.innerHTML = `
      <div class="eval-splits-grid">
        ${testHtml}
        ${trainHtml}
      </div>`;
  }

  // 3. Emerging Panel: Config YAML
  openConfigBtn.addEventListener("click", async () => {
    if (!state.selectedTaskUri) return;
    openEmergingPanel({
      kicker: "Task Configuration",
      title: "config.yaml",
      uri: `${state.selectedTaskUri}/config.yaml`,
      bodyHtml: `<div style="color:var(--text-muted);">Loading configuration...</div>`,
    });

    try {
      const data = await fetchJson(`/api/task/config?task_uri=${encodeURIComponent(state.selectedTaskUri)}`);
      panelBody.innerHTML = `<pre class="code-viewer">${escapeHtml(data.yaml_text)}</pre>`;
    } catch (err) {
      panelBody.innerHTML = `<div class="status-banner">${escapeHtml(err.message)}</div>`;
    }
  });

  // 4. Emerging Panel: Log File Viewer
  async function openLogInPanel(fileUri, logName) {
    openEmergingPanel({
      kicker: "Execution Log Viewer",
      title: logName,
      uri: fileUri,
      toolbarHtml: `
        <div style="display:flex;gap:10px;align-items:center;width:100%;">
          <input id="log-filter-input" type="text" class="text-input" placeholder="Filter log lines..." style="max-width:360px;" />
          <span id="log-meta-label" style="font-size:12px;color:var(--text-muted);"></span>
        </div>`,
      bodyHtml: `<div style="color:var(--text-muted);">Loading log file...</div>`,
    });

    try {
      const data = await fetchJson(`/api/task/log?file_uri=${encodeURIComponent(fileUri)}`);
      const allLines = (data.content || "").split("\n");
      const metaLabel = document.getElementById("log-meta-label");
      const filterInput = document.getElementById("log-filter-input");

      const renderLines = (query) => {
        const q = (query || "").trim().toLowerCase();
        const filtered = q ? allLines.filter((l) => l.toLowerCase().includes(q)) : allLines;
        if (metaLabel) {
          metaLabel.textContent = `Showing ${filtered.length} of ${allLines.length} lines (${data.size_bytes} bytes)`;
        }
        panelBody.innerHTML = `<pre class="code-viewer">${escapeHtml(filtered.join("\n"))}</pre>`;
      };

      renderLines("");
      if (filterInput) {
        filterInput.addEventListener("input", () => renderLines(filterInput.value));
      }
    } catch (err) {
      panelBody.innerHTML = `<div class="status-banner">${escapeHtml(err.message)}</div>`;
    }
  }

  // 5. Emerging Panel: Structured Dataset Viewer
  async function openDatasetInPanel(fileUri, displayName) {
    let currentOffset = 0;
    const limit = 50;
    let currentQuery = "";

    openEmergingPanel({
      kicker: "Structured Dataset Table",
      title: displayName,
      uri: fileUri,
      toolbarHtml: `
        <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:space-between;width:100%;">
          <div style="display:flex;gap:8px;align-items:center;">
            <input id="ds-search-input" type="text" class="text-input" placeholder="Search across all dataset fields..." style="width:300px;" />
            <button id="ds-search-btn" type="button" class="btn btn-secondary btn-sm">Filter</button>
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <span id="ds-page-info" class="mono" style="font-size:12px;color:var(--text-secondary);"></span>
            <button id="ds-prev-btn" type="button" class="btn btn-secondary btn-sm">Prev</button>
            <button id="ds-next-btn" type="button" class="btn btn-secondary btn-sm">Next</button>
          </div>
        </div>`,
      bodyHtml: `<div style="color:var(--text-muted);">Loading structured dataset table...</div>`,
    });

    const loadPage = async () => {
      try {
        const url = `/api/task/dataset?file_uri=${encodeURIComponent(
          fileUri
        )}&offset=${currentOffset}&limit=${limit}&q=${encodeURIComponent(currentQuery)}`;
        const data = await fetchJson(url);

        const pageInfo = document.getElementById("ds-page-info");
        const prevBtn = document.getElementById("ds-prev-btn");
        const nextBtn = document.getElementById("ds-next-btn");

        const startRow = data.total_rows === 0 ? 0 : data.offset + 1;
        const endRow = Math.min(data.offset + data.rows.length, data.total_rows);
        if (pageInfo) {
          pageInfo.textContent = `Rows ${startRow}–${endRow} of ${data.total_rows} (${data.columns.length} fields)`;
        }
        if (prevBtn) prevBtn.disabled = currentOffset <= 0;
        if (nextBtn) nextBtn.disabled = currentOffset + limit >= data.total_rows;

        if (data.rows.length === 0) {
          panelBody.innerHTML = `<div style="padding:16px;color:var(--text-muted);">No matching records found.</div>`;
          return;
        }

        const ths = [`<th>#</th>`, ...data.columns.map((c) => `<th class="mono">${escapeHtml(c)}</th>`)].join("");
        const trs = data.rows
          .map((row) => {
            const tds = data.columns
              .map((col) => {
                const val = row[col];
                const text = val === null || val === undefined ? "" : String(val);
                return `<td><div class="cell-truncate">${escapeHtml(text)}</div></td>`;
              })
              .join("");
            return `<tr><td class="mono" style="color:var(--text-muted);">${escapeHtml(row._row_num)}</td>${tds}</tr>`;
          })
          .join("");

        panelBody.innerHTML = `
          <div class="table-wrapper">
            <table class="data-table">
              <thead><tr>${ths}</tr></thead>
              <tbody>${trs}</tbody>
            </table>
          </div>`;
      } catch (err) {
        panelBody.innerHTML = `<div class="status-banner">${escapeHtml(err.message)}</div>`;
      }
    };

    const searchInput = document.getElementById("ds-search-input");
    const searchBtn = document.getElementById("ds-search-btn");
    const prevBtn = document.getElementById("ds-prev-btn");
    const nextBtn = document.getElementById("ds-next-btn");

    if (searchBtn && searchInput) {
      const triggerSearch = () => {
        currentQuery = searchInput.value;
        currentOffset = 0;
        loadPage();
      };
      searchBtn.addEventListener("click", triggerSearch);
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") triggerSearch();
      });
    }
    if (prevBtn) {
      prevBtn.addEventListener("click", () => {
        currentOffset = Math.max(0, currentOffset - limit);
        loadPage();
      });
    }
    if (nextBtn) {
      nextBtn.addEventListener("click", () => {
        currentOffset += limit;
        loadPage();
      });
    }

    await loadPage();
  }

  // 6. Emerging Panel: Side-by-Side Evaluation Inferences
  openInferencesBtn.addEventListener("click", async () => {
    if (!state.selectedTaskUri) return;
    let currentSplit = "all";
    let currentSearch = "";

    openEmergingPanel({
      kicker: "Side-by-Side Evaluation Comparison",
      title: "Student Before Training vs. Student After Training vs. Teacher Model",
      uri: `${state.selectedTaskUri}/04_evaluation/predictions.jsonl`,
      toolbarHtml: `
        <div style="display:flex;flex-wrap:wrap;gap:12px;align-items:center;justify-content:space-between;width:100%;">
          <div style="display:flex;gap:8px;align-items:center;">
            <label for="inf-split-select" style="font-size:12px;font-weight:600;">Split:</label>
            <select id="inf-split-select" class="select-input" style="width:160px;">
              <option value="all">All Splits</option>
              <option value="test">Test Split (Held-Out)</option>
              <option value="train">Train Split</option>
            </select>
            <input id="inf-search-input" type="text" class="text-input" placeholder="Search prompt or model generations..." style="width:320px;" />
          </div>
          <span id="inf-count-label" class="mono" style="font-size:12px;color:var(--text-secondary);"></span>
        </div>`,
      bodyHtml: `<div style="color:var(--text-muted);">Loading side-by-side inferences...</div>`,
    });

    const loadInferences = async () => {
      try {
        const url = `/api/task/inferences?task_uri=${encodeURIComponent(
          state.selectedTaskUri
        )}&split=${encodeURIComponent(currentSplit)}&q=${encodeURIComponent(currentSearch)}`;
        const data = await fetchJson(url);
        const countLabel = document.getElementById("inf-count-label");
        if (countLabel) {
          countLabel.textContent = `Showing ${data.filtered_count} of ${data.total_count} evaluation samples`;
        }

        if (!data.records || data.records.length === 0) {
          panelBody.innerHTML = `<div style="padding:16px;color:var(--text-muted);">No matching evaluation inferences found.</div>`;
          return;
        }

        const renderSampleMetricsPanel = (metricsObj, deltaObj, isAfter) => {
          if (!metricsObj || typeof metricsObj !== "object") return "";
          const metricDefs = [
            { key: "exact_match", label: "Exact Match", decimals: 4, higherBetter: true },
            { key: "rouge1", label: "ROUGE-1", decimals: 4, higherBetter: true },
            { key: "rouge2", label: "ROUGE-2", decimals: 4, higherBetter: true },
            { key: "bleu", label: "BLEU-4", decimals: 4, higherBetter: true },
            { key: "llm_judge_score", label: "LLM Judge", decimals: 1, suffix: " / 5", higherBetter: true },
            { key: "output_tokens", label: "Output Length", decimals: 0, suffix: " toks", higherBetter: null },
            { key: "ms_per_output_token", label: "Time / Output Token", decimals: 2, suffix: " ms/tok", higherBetter: false },
            { key: "latency_ms", label: "Latency", decimals: 1, suffix: " ms", higherBetter: false },
          ];

          const pills = metricDefs
            .filter((m) => metricsObj[m.key] !== undefined && metricsObj[m.key] !== null)
            .map((m) => {
              const rawVal = Number(metricsObj[m.key]);
              const formattedVal = Number.isFinite(rawVal)
                ? `${rawVal.toFixed(m.decimals)}${m.suffix || ""}`
                : String(metricsObj[m.key]);

              let deltaBadge = "";
              if (isAfter && deltaObj && deltaObj[m.key] !== undefined && deltaObj[m.key] !== null) {
                const dVal = Number(deltaObj[m.key]);
                if (Number.isFinite(dVal) && Math.abs(dVal) > 1e-6) {
                  const sign = dVal > 0 ? "+" : "";
                  let badgeClass = "";
                  if (m.higherBetter !== null && m.higherBetter !== undefined) {
                    const isGood = m.higherBetter ? dVal > 0 : dVal < 0;
                    badgeClass = isGood ? "metric-delta-pos" : "metric-delta-neg";
                  }
                  deltaBadge = `<span class="metric-delta-chip ${badgeClass}">${sign}${dVal.toFixed(
                    m.decimals
                  )}</span>`;
                }
              }

              return `
                <div class="sample-metric-chip">
                  <span class="sample-metric-label">${escapeHtml(m.label)}</span>
                  <span class="sample-metric-val mono">${escapeHtml(formattedVal)} ${deltaBadge}</span>
                </div>`;
            })
            .join("");

          const judgeReasonHtml = metricsObj.llm_judge_reason
            ? `<div class="sample-judge-reason"><strong>Judge Rationale:</strong> ${escapeHtml(
                String(metricsObj.llm_judge_reason)
              )}</div>`
            : "";

          if (!pills && !judgeReasonHtml) return "";
          return `
            <div class="sample-metrics-box">
              <div class="sample-metrics-title">Case Evaluation Metrics</div>
              <div class="sample-metrics-grid">${pills}</div>
              ${judgeReasonHtml}
            </div>`;
        };

        panelBody.innerHTML = data.records
          .map((rec) => {
            const bLat =
              rec.base_latency_ms !== null && rec.base_latency_ms !== undefined
                ? `${Number(rec.base_latency_ms).toFixed(1)} ms`
                : "";
            const dLat =
              rec.distilled_latency_ms !== null && rec.distilled_latency_ms !== undefined
                ? `${Number(rec.distilled_latency_ms).toFixed(1)} ms`
                : "";
            const beforeMetricsHtml = renderSampleMetricsPanel(rec.base_metrics, null, false);
            const afterMetricsHtml = renderSampleMetricsPanel(
              rec.distilled_metrics,
              rec.metrics_delta,
              true
            );
            return `
              <article class="inference-card">
                <div class="inference-prompt-bar">
                  <div class="inference-prompt-meta">
                    <span style="font-weight:700;font-size:12px;">Sample #${escapeHtml(rec.index)}</span>
                    <span class="status-badge status-${rec.split === "test" ? "COMPLETED" : "RUNNING"}">${escapeHtml(
              String(rec.split).toUpperCase()
            )} SPLIT</span>
                  </div>
                  <div class="inference-prompt-text">${escapeHtml(rec.prompt)}</div>
                </div>
                <div class="inference-columns">
                  <div class="inference-col col-before">
                    <div class="inference-col-header">
                      <span>1. Student Before Training (Base)</span>
                      <span class="mono" style="font-weight:500;">${escapeHtml(bLat)}</span>
                    </div>
                    ${beforeMetricsHtml}
                    <pre class="inference-output-box">${escapeHtml(rec.student_before_training)}</pre>
                  </div>
                  <div class="inference-col col-after">
                    <div class="inference-col-header">
                      <span>2. Student After Training (Distilled)</span>
                      <span class="mono" style="font-weight:500;">${escapeHtml(dLat)}</span>
                    </div>
                    ${afterMetricsHtml}
                    <pre class="inference-output-box">${escapeHtml(rec.student_after_training)}</pre>
                  </div>
                  <div class="inference-col col-teacher">
                    <div class="inference-col-header">
                      <span>3. Teacher Model (Gemini Reference)</span>
                      <span class="mono" style="font-weight:500;">Reference</span>
                    </div>
                    <pre class="inference-output-box">${escapeHtml(rec.teacher_model)}</pre>
                  </div>
                </div>
              </article>`;
          })
          .join("");
      } catch (err) {
        panelBody.innerHTML = `<div class="status-banner">${escapeHtml(err.message)}</div>`;
      }
    };

    const splitSelect = document.getElementById("inf-split-select");
    const searchInput = document.getElementById("inf-search-input");
    if (splitSelect) {
      splitSelect.addEventListener("change", () => {
        currentSplit = splitSelect.value;
        loadInferences();
      });
    }
    if (searchInput) {
      searchInput.addEventListener("input", () => {
        currentSearch = searchInput.value;
        loadInferences();
      });
    }

    await loadInferences();
  });

  function openAboutDistillationPanel() {
    const bodyHtml = `
      <div class="about-guide-container">
        <div class="about-intro-banner">
          <strong>Knowledge Distillation in <code class="mono">distillfw</code></strong> transfers task-specific capabilities from a large frontier <strong>Teacher model (Gemini)</strong> into a compact, open-weights <strong>Student model (Gemma)</strong>. Selecting the right algorithm depends on two fundamental design choices: <strong>(1) Off-Policy vs. On-Policy rollout collection</strong> and <strong>(2) Text-Only vs. Token-Level Logprob supervision (and tokenizer alignment)</strong>.
        </div>

        <!-- 1. Off-Policy vs. On-Policy Paradigm Comparison -->
        <div>
          <h3 class="subsection-title">1. Core Paradigm Comparison: Off-Policy vs. On-Policy (&amp; Hybrid)</h3>
          <div class="about-paradigm-grid">
            <article class="paradigm-card paradigm-card-off">
              <div class="paradigm-card-header">
                <h4 class="paradigm-card-title">Off-Policy Distillation (<code class="mono">off_policy</code>)</h4>
                <span class="paradigm-badge paradigm-badge-off">Fixed Teacher Dataset</span>
              </div>
              <ul class="paradigm-list">
                <li><strong>How it works:</strong> In Stage 1 (<code class="mono">dataset_generator</code>), the Gemini Teacher generates a fixed dataset of completions (and optionally top-<em>k</em> token logprobs) once. In Stage 3 (<code class="mono">model_trainer</code>), the Gemma Student trains strictly on those static trajectories (<code class="mono">y ~ p_teacher(y|x)</code>).</li>
                <li><strong>Advantages:</strong> Maximum GPU training speed, zero Teacher API cost during training, and 100% deterministic/reusable datasets cached on GCS.</li>
                <li><strong>Core Limitation (Exposure Bias):</strong> During training, the Student only ever conditions on the Teacher's gold prefix <code class="mono">y_{&lt;t} ~ p_teacher</code>. At deployment time, the Student conditions on its own generated prefix <code class="mono">y_{&lt;t} ~ q_student</code>; small early deviations can compound over long outputs.</li>
                <li><strong>Mitigations in Off-Policy:</strong> Using mode-seeking / bounded objectives (<code class="mono">reverse_kl</code>, <code class="mono">skew_kl</code>) or contrastive offline student pairs (<code class="mono">distillm2</code>, <code class="mono">dpo</code>, <code class="mono">simpo</code>, <code class="mono">orpo</code>).</li>
              </ul>
            </article>

            <article class="paradigm-card paradigm-card-on">
              <div class="paradigm-card-header">
                <h4 class="paradigm-card-title">On-Policy &amp; Hybrid Distillation (<code class="mono">on_policy</code> / <code class="mono">hybrid</code>)</h4>
                <span class="paradigm-badge paradigm-badge-on">Student-Generated Rollouts</span>
              </div>
              <ul class="paradigm-list">
                <li><strong>How it works:</strong> During Stage 3 training, the Student actively samples its own autoregressive sequences (<code class="mono">y ~ q_student(y|x)</code>, or <code class="mono">&lambda;</code>-mixed with Teacher trajectories in <code class="mono">gkd</code>). The Teacher or Reward Judge then scores the exact states the Student actually visits.</li>
                <li><strong>Advantages:</strong> Directly eliminates <strong>exposure bias</strong> (train–inference mismatch) and explicitly teaches the Student to recover from its own errors and avoid low-density Teacher regions.</li>
                <li><strong>Core Trade-off:</strong> Higher wall-clock training time and compute cost during Stage 3 due to online autoregressive rollout generation and Teacher logprob / reward scoring.</li>
                <li><strong>Supported Modes:</strong> Generalized KD (<code class="mono">gkd</code> with <code class="mono">&lambda; &in; (0, 1]</code>) for token-level logprob feedback, or Group Relative Policy Optimization (<code class="mono">grpo</code>) for scalar reward/verifier feedback.</li>
              </ul>
            </article>
          </div>
        </div>

        <!-- 2. Technical Requirements: Teacher Logprobs & Vocabulary Alignment -->
        <div>
          <h3 class="subsection-title">2. Vocabulary Alignment &amp; Teacher Logprobs (<code class="mono">response_logprobs</code>) Requirements</h3>
          <div class="about-paradigm-grid">
            <article class="paradigm-card">
              <div class="paradigm-card-header">
                <h4 class="paradigm-card-title">Sequence-Level / Text-Space Algorithms</h4>
                <span class="req-pill req-pill-none">Tokenizer-Agnostic &bull; No Logprobs</span>
              </div>
              <ul class="paradigm-list">
                <li><strong>Algorithms:</strong> <code class="mono">sft_seqkd</code>, <code class="mono">dpo</code>, <code class="mono">simpo</code>, <code class="mono">orpo</code>, <code class="mono">grpo</code>.</li>
                <li><strong>Teacher Logprobs (<code class="mono">teacher.response_logprobs</code>):</strong> Must be <code class="mono">false</code>. Only decoded surface text strings (and optional <code class="mono">thought</code> traces) are queried from Gemini.</li>
                <li><strong>Vocabulary Alignment:</strong> <strong>Not required.</strong> Because supervision happens in detokenized text space (or scalar sequence rewards), the Gemini Teacher and Gemma Student can have completely different tokenizers, subword splits, and vocabulary sizes. Stage 2 (<code class="mono">dataset_formatter</code>) simply tokenizes the text with the Student's tokenizer.</li>
              </ul>
            </article>

            <article class="paradigm-card">
              <div class="paradigm-card-header">
                <h4 class="paradigm-card-title">Token-Level / Gray-Box Logit KD Algorithms</h4>
                <span class="req-pill req-pill-required">Vocab Alignment &bull; Logprobs Required</span>
              </div>
              <ul class="paradigm-list">
                <li><strong>Algorithms:</strong> <code class="mono">forward_kl</code>, <code class="mono">reverse_kl</code>, <code class="mono">jsd</code>, <code class="mono">skew_kl</code>, <code class="mono">distillm2</code>, <code class="mono">gkd</code>.</li>
                <li><strong>Teacher Logprobs (<code class="mono">teacher.response_logprobs</code>):</strong> Must be <code class="mono">true</code> with <code class="mono">teacher.logprobs_top_k</code> (<code class="mono">1..20</code>). <code class="mono">distillfw init</code> runs a preflight probe to verify the Teacher endpoint returns at least <code class="mono">top_k</code> logprobs per token.</li>
                <li><strong>Vocabulary Alignment:</strong> <strong>Required.</strong> Computing token-level divergence <code class="mono">D(p_T(&middot;|y_{&lt;t}) || q_S(&middot;|y_{&lt;t}))</code> requires mapping each Teacher top-<em>k</em> token candidate onto the Student's vocabulary IDs (<code class="mono">dataset_representation: sparse_logprobs</code>), with residual probability mass <code class="mono">1 - &sum; p_T(top_k)</code> smoothed across the remaining Student vocabulary (<code class="mono">sparse_topk_kl_loss</code>).</li>
              </ul>
            </article>
          </div>
        </div>

        <!-- 3. Complete Algorithm Reference Table -->
        <div>
          <h3 class="subsection-title">3. Supported Distillation Algorithms &amp; Configuration Matrix</h3>
          <div class="table-wrapper">
            <table class="data-table">
              <thead>
                <tr>
                  <th>Algorithm (<code class="mono">training.algorithm</code>)</th>
                  <th>Family &amp; Objective</th>
                  <th>Paradigm (<code class="mono">training.paradigm</code>)</th>
                  <th>Teacher Logprobs (<code class="mono">response_logprobs</code>)</th>
                  <th>Tokenizer / Vocab Alignment</th>
                  <th>Dataset Representation</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="mono"><strong>sft_seqkd</strong></td>
                  <td>Sequence-Level KD / SFT (Cross-Entropy on Teacher text &amp; thoughts)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-none">No (false)</span></td>
                  <td><span class="req-pill req-pill-none">Not Required (Text Space)</span></td>
                  <td class="mono">text / tokens</td>
                </tr>
                <tr>
                  <td class="mono"><strong>forward_kl</strong></td>
                  <td>Word-Level Forward KLD <code class="mono">KL(p_T || q_S)</code> (Mode-Covering, Hinton et al.)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-required">Required (true, top_k)</span></td>
                  <td><span class="req-pill req-pill-required">Required (Token-Aligned)</span></td>
                  <td class="mono">sparse_logprobs</td>
                </tr>
                <tr>
                  <td class="mono"><strong>reverse_kl</strong></td>
                  <td>Reverse KLD <code class="mono">KL(q_S || p_T)</code> (Mode-Seeking, avoids low-density Teacher regions)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-required">Required (true, top_k)</span></td>
                  <td><span class="req-pill req-pill-required">Required (Token-Aligned)</span></td>
                  <td class="mono">sparse_logprobs</td>
                </tr>
                <tr>
                  <td class="mono"><strong>jsd</strong></td>
                  <td>Generalized Jensen-Shannon Divergence <code class="mono">JSD_&beta;(p_T || q_S)</code> (Bounded symmetric divergence)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-required">Required (true, top_k)</span></td>
                  <td><span class="req-pill req-pill-required">Required (Token-Aligned)</span></td>
                  <td class="mono">sparse_logprobs</td>
                </tr>
                <tr>
                  <td class="mono"><strong>skew_kl</strong></td>
                  <td><em>DistiLLM</em> Skew KLD <code class="mono">KL(p_T || &alpha;p_T + (1-&alpha;)q_S)</code> (Stabilized off-policy gradients)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-required">Required (true, top_k)</span></td>
                  <td><span class="req-pill req-pill-required">Required (Token-Aligned)</span></td>
                  <td class="mono">sparse_logprobs</td>
                </tr>
                <tr>
                  <td class="mono"><strong>distillm2</strong></td>
                  <td><em>DistiLLM-2</em> Asymmetric Contrastive KD (Skew KLD on Teacher + Reverse KLD on Student errors)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-required">Required (true, top_k)</span></td>
                  <td><span class="req-pill req-pill-required">Required (Token-Aligned)</span></td>
                  <td class="mono">sparse_logprobs</td>
                </tr>
                <tr>
                  <td class="mono"><strong>dpo / simpo / orpo</strong></td>
                  <td>Contrastive Preference Optimization (<code class="mono">chosen</code> Teacher vs. <code class="mono">rejected</code> base Student)</td>
                  <td><span class="paradigm-badge paradigm-badge-off">off_policy</span></td>
                  <td><span class="req-pill req-pill-none">No (false)</span></td>
                  <td><span class="req-pill req-pill-none">Not Required (Text Pairs)</span></td>
                  <td class="mono">preference</td>
                </tr>
                <tr>
                  <td class="mono"><strong>gkd</strong></td>
                  <td>Generalized Knowledge Distillation (<code class="mono">&lambda;</code>-mixed Student On-Policy rollouts + token KLD/JSD)</td>
                  <td><span class="paradigm-badge paradigm-badge-on">on_policy / hybrid</span></td>
                  <td><span class="req-pill req-pill-required">Required (true, top_k)</span></td>
                  <td><span class="req-pill req-pill-required">Required (Token-Aligned)</span></td>
                  <td class="mono">sparse_logprobs / tokens</td>
                </tr>
                <tr>
                  <td class="mono"><strong>grpo</strong></td>
                  <td>Group Relative Policy Optimization (Online Student rollout groups scored by Gemini Judge / Verifier)</td>
                  <td><span class="paradigm-badge paradigm-badge-on">on_policy</span></td>
                  <td><span class="req-pill req-pill-none">No (false)</span></td>
                  <td><span class="req-pill req-pill-none">Not Required (Reward on Text)</span></td>
                  <td class="mono">text</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;

    openEmergingPanel({
      kicker: "Distillation Reference Guide",
      title: "About Distillation: Off-Policy vs. On-Policy, Logprobs & Vocabulary Alignment",
      uri: "",
      toolbarHtml: "",
      bodyHtml,
    });
  }

  // Event Listeners for Top Bar
  if (aboutDistillationBtn) {
    aboutDistillationBtn.addEventListener("click", openAboutDistillationPanel);
  }
  loadTasksBtn.addEventListener("click", () => loadTasksFromRoot(true));
  gcsRootInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") loadTasksFromRoot(true);
  });
  taskSelect.addEventListener("change", () => {
    selectTask(taskSelect.value);
  });
  refreshTaskBtn.addEventListener("click", () => {
    if (state.selectedTaskUri) {
      selectTask(state.selectedTaskUri);
    }
  });

  // Initialize default GCS root path from backend
  (async function init() {
    try {
      const defaults = await fetchJson("/api/defaults");
      if (defaults.default_root_uri) {
        gcsRootInput.value = defaults.default_root_uri;
        await loadTasksFromRoot(true);
      }
    } catch (_) {}
  })();
})();
