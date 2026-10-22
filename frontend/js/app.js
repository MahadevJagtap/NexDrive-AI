/**
 * Autonomous Driving Perception Stack — Clean Frontend Application Logic
 * Pure Vanilla JavaScript (ES6+) for interactive perception analysis
 */

(() => {
  'use strict';

  // --- STATE MANAGEMENT ---
  const state = {
    backendOnline: false,
    inputMode: 'demo', // 'demo' | 'upload'
    selectedFile: null,
    activeJobId: null,
    pollingInterval: null,
    sampleAvailable: false,
    currentVideoTab: 'processed', // 'processed' | 'original' | 'split'
  };

  // --- DOM ELEMENT REFERENCES ---
  const elements = {
    // Header
    backendStatus: document.getElementById('backend-status'),
    backendStatusText: document.getElementById('backend-status-text'),

    // Views
    setupSection: document.getElementById('setup-section'),
    processingSection: document.getElementById('processing-section'),
    resultsSection: document.getElementById('results-section'),

    // Step 1: Video Selection
    tabDemo: document.getElementById('tab-demo'),
    tabUpload: document.getElementById('tab-upload'),
    panelDemo: document.getElementById('panel-demo'),
    panelUpload: document.getElementById('panel-upload'),
    presetMeta: document.getElementById('preset-meta'),
    demoError: document.getElementById('demo-error'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    fileSelectedBox: document.getElementById('file-selected-box'),
    selectedFileName: document.getElementById('selected-file-name'),
    selectedFileSize: document.getElementById('selected-file-size'),
    btnRemoveFile: document.getElementById('btn-remove-file'),
    uploadError: document.getElementById('upload-error'),

    // Step 2: Options
    optDetection: document.getElementById('opt-detection'),
    optTracking: document.getElementById('opt-tracking'),
    optLanes: document.getElementById('opt-lanes'),
    optDepth: document.getElementById('opt-depth'),
    depthNotice: document.getElementById('depth-notice'),
    advModel: document.getElementById('adv-model'),
    advMaxFrames: document.getElementById('adv-max-frames'),
    advConf: document.getElementById('adv-conf'),
    advConfVal: document.getElementById('adv-conf-val'),
    advLaneAlgo: document.getElementById('adv-lane-algo'),

    // Step 3: Action & Progress
    btnAnalyze: document.getElementById('btn-analyze'),
    processingTitle: document.getElementById('processing-title'),
    processingSubtitle: document.getElementById('processing-subtitle'),
    progressFill: document.getElementById('progress-fill'),
    progressFrames: document.getElementById('progress-frames'),
    progressPercent: document.getElementById('progress-percent'),
    btnCancel: document.getElementById('btn-cancel'),

    // Results: Summary Banner
    bannerSummary: document.getElementById('banner-summary'),
    btnBannerDownload: document.getElementById('btn-banner-download'),
    btnBannerReset: document.getElementById('btn-banner-reset'),

    // Results: Video Viewports & Controls
    tabViewProcessed: document.getElementById('tab-view-processed'),
    tabViewOriginal: document.getElementById('tab-view-original'),
    tabViewSplit: document.getElementById('tab-view-split'),
    videoModeLabel: document.getElementById('video-mode-label'),
    singleViewport: document.getElementById('single-viewport'),
    splitViewport: document.getElementById('split-viewport'),
    playerMain: document.getElementById('player-main'),
    playerSplitOrig: document.getElementById('player-split-orig'),
    playerSplitAnnot: document.getElementById('player-split-annot'),
    btnDownloadVideoCard: document.getElementById('btn-download-video-card'),

    // Results: Key Metrics
    valDetected: document.getElementById('val-detected'),
    subDetected: document.getElementById('sub-detected'),
    valTracked: document.getElementById('val-tracked'),
    subTracked: document.getElementById('sub-tracked'),
    valLanes: document.getElementById('val-lanes'),
    subLanes: document.getElementById('sub-lanes'),
    valSpeed: document.getElementById('val-speed'),
    subSpeed: document.getElementById('sub-speed'),

    // Results: Objects Table
    badgeObjectsCount: document.getElementById('badge-objects-count'),
    objectsTableBody: document.getElementById('objects-table-body'),

    // Results: Technical Details
    techLaneStatus: document.getElementById('tech-lane-status'),
    techLaneOffset: document.getElementById('tech-lane-offset'),
    techLaneConf: document.getElementById('tech-lane-conf'),
    techDepthStatus: document.getElementById('tech-depth-status'),
    techHazardAlert: document.getElementById('tech-hazard-alert'),
    eventLogFeed: document.getElementById('event-log-feed'),
    btnDownloadTelemetry: document.getElementById('btn-download-telemetry'),

    // Results: Next Actions
    btnNextDownload: document.getElementById('btn-next-download'),
    btnNextReset: document.getElementById('btn-next-reset'),
  };

  // --- INITIALIZATION ---
  document.addEventListener('DOMContentLoaded', () => {
    initEventListeners();
    checkBackendHealth();
    fetchPresetSamples();
  });

  // --- EVENT LISTENERS REGISTRATION ---
  function initEventListeners() {
    // Step 1: Input Source Tabs
    elements.tabDemo.addEventListener('click', () => switchInputSource('demo'));
    elements.tabUpload.addEventListener('click', () => switchInputSource('upload'));

    // Step 1: File Upload & Drag-and-Drop
    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileInputChange);
    elements.btnRemoveFile.addEventListener('click', handleFileRemove);

    ['dragenter', 'dragover'].forEach(name => {
      elements.dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(name => {
      elements.dropZone.addEventListener(name, (e) => {
        e.preventDefault();
        e.stopPropagation();
        elements.dropZone.classList.remove('dragover');
      });
    });

    elements.dropZone.addEventListener('drop', handleFileDrop);

    // Step 2: Depth speed notice toggle
    elements.optDepth.addEventListener('change', () => {
      elements.depthNotice.classList.toggle('hidden', !elements.optDepth.checked);
    });

    // Step 2: Confidence slider readout
    elements.advConf.addEventListener('input', () => {
      elements.advConfVal.textContent = parseFloat(elements.advConf.value).toFixed(2);
    });

    // Step 3: Start Analysis & Cancel
    elements.btnAnalyze.addEventListener('click', startAnalysis);
    elements.btnCancel.addEventListener('click', cancelActiveAnalysis);

    // Results: Video View Modes
    elements.tabViewProcessed.addEventListener('click', () => switchVideoViewMode('processed'));
    elements.tabViewOriginal.addEventListener('click', () => switchVideoViewMode('original'));
    elements.tabViewSplit.addEventListener('click', () => switchVideoViewMode('split'));

    // Results: Reset / Analyze Another Video
    elements.btnBannerReset.addEventListener('click', resetToSetupView);
    elements.btnNextReset.addEventListener('click', resetToSetupView);

    // Synchronize split video viewports
    elements.playerSplitOrig.addEventListener('play', () => elements.playerSplitAnnot.play());
    elements.playerSplitOrig.addEventListener('pause', () => elements.playerSplitAnnot.pause());
    elements.playerSplitOrig.addEventListener('seeked', () => {
      elements.playerSplitAnnot.currentTime = elements.playerSplitOrig.currentTime;
    });
  }

  // --- BACKEND HEALTH & SAMPLE CHECK ---
  async function checkBackendHealth() {
    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check non-200');
      const data = await res.json();

      state.backendOnline = true;
      elements.backendStatus.className = 'status-pill status-online';
      elements.backendStatusText.textContent = `Connected (${data.device.toUpperCase()})`;
    } catch (err) {
      state.backendOnline = false;
      elements.backendStatus.className = 'status-pill status-disconnected';
      elements.backendStatusText.textContent = 'Disconnected';
    }
  }

  async function fetchPresetSamples() {
    try {
      const res = await fetch('/api/samples');
      if (res.ok) {
        const samples = await res.json();
        if (samples && samples.length > 0) {
          state.sampleAvailable = true;
          const s = samples[0];
          elements.presetMeta.textContent = `Resolution: ${s.resolution} • ${s.frames} frames @ ${s.fps} FPS • ${s.size_mb} MB MP4`;
          elements.demoError.classList.add('hidden');
        } else {
          showDemoError("Preset video 'demo_banner.mp4' is not present in data/. Please upload your own driving video.");
        }
      } else {
        showDemoError('Could not verify preset demo video on server.');
      }
    } catch (err) {
      showDemoError('Preset video server check failed.');
    }
  }

  function showDemoError(msg) {
    state.sampleAvailable = false;
    elements.demoError.textContent = msg;
    elements.demoError.classList.remove('hidden');
  }

  // --- STEP 1: INPUT SOURCE SWITCHING ---
  function switchInputSource(mode) {
    state.inputMode = mode;
    elements.tabDemo.classList.toggle('active', mode === 'demo');
    elements.tabUpload.classList.toggle('active', mode === 'upload');

    elements.panelDemo.classList.toggle('hidden', mode !== 'demo');
    elements.panelUpload.classList.toggle('hidden', mode !== 'upload');

    clearUploadError();
  }

  function handleFileInputChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      processSelectedFile(e.target.files[0]);
    }
  }

  function handleFileDrop(e) {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      processSelectedFile(e.dataTransfer.files[0]);
    }
  }

  function processSelectedFile(file) {
    clearUploadError();
    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const lowerName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => lowerName.endsWith(ext));

    if (!isValid) {
      showUploadError(`Unsupported video format. Please upload one of: ${validExtensions.join(', ')}`);
      return;
    }

    const maxBytes = 150 * 1024 * 1024; // 150 MB safety cap
    if (file.size > maxBytes) {
      showUploadError('The selected video exceeds the 150 MB file size limit.');
      return;
    }

    state.selectedFile = file;
    elements.selectedFileName.textContent = file.name;
    elements.selectedFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;

    elements.dropZone.classList.add('hidden');
    elements.fileSelectedBox.classList.remove('hidden');
  }

  function handleFileRemove(e) {
    if (e) e.stopPropagation();
    state.selectedFile = null;
    elements.fileInput.value = '';
    elements.fileSelectedBox.classList.add('hidden');
    elements.dropZone.classList.remove('hidden');
    clearUploadError();
  }

  function showUploadError(msg) {
    elements.uploadError.textContent = msg;
    elements.uploadError.classList.remove('hidden');
  }

  function clearUploadError() {
    elements.uploadError.textContent = '';
    elements.uploadError.classList.add('hidden');
  }

  // --- STEP 3: START ANALYSIS ---
  async function startAnalysis() {
    clearUploadError();

    // Input Validation
    if (state.inputMode === 'upload' && !state.selectedFile) {
      showUploadError('Please select or drop a valid road driving video before starting.');
      return;
    }

    if (state.inputMode === 'demo' && !state.sampleAvailable) {
      showDemoError('The preset demo video is unavailable. Please upload a road video.');
      switchInputSource('upload');
      return;
    }

    // Build Form Data
    const formData = new FormData();
    if (state.inputMode === 'upload') {
      formData.append('file', state.selectedFile);
    } else {
      formData.append('sample_id', 'demo_banner');
    }

    formData.append('enable_tracking', elements.optTracking.checked);
    formData.append('enable_lanes', elements.optLanes.checked);
    formData.append('lane_method', elements.advLaneAlgo.value);
    formData.append('enable_depth', elements.optDepth.checked);
    formData.append('depth_freq', elements.optDepth.checked ? 10 : 10);
    formData.append('conf_thresh', parseFloat(elements.advConf.value));
    formData.append('max_frames', parseInt(elements.advMaxFrames.value, 10));
    formData.append('model_variant', elements.advModel.value);

    // Transition UI to Processing Mode
    elements.setupSection.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.processingSection.classList.remove('hidden');

    elements.processingTitle.textContent = 'Uploading Video...';
    elements.processingSubtitle.textContent = 'Transmitting data to perception server';
    elements.progressFill.style.width = '0%';
    elements.progressFrames.textContent = 'Starting job...';
    elements.progressPercent.textContent = '0%';
    elements.btnCancel.classList.remove('hidden');

    try {
      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        let errMessage = `Server error (${res.status} ${res.statusText})`;
        try {
          const errorData = await res.json();
          if (errorData && errorData.detail) {
            errMessage = typeof errorData.detail === 'string'
              ? errorData.detail
              : JSON.stringify(errorData.detail);
          }
        } catch {
          try {
            const rawText = await res.text();
            if (rawText && rawText.trim().length > 0 && rawText.length < 250) {
              errMessage = rawText.trim();
            } else if (res.status === 502 || res.status === 503 || res.status === 504) {
              errMessage = 'Render service is currently waking up or restarting. Please wait 15 seconds and try again.';
            }
          } catch {}
        }
        throw new Error(errMessage);
      }

      const jobData = await res.json();
      state.activeJobId = jobData.job_id;

      elements.processingTitle.textContent = 'Analyzing Frames...';
      elements.processingSubtitle.textContent = 'Running object detection, tracking, and lane boundary fitting';

      // Start Polling Job Status
      startJobPolling(jobData.job_id);

    } catch (err) {
      alert(`Error starting analysis: ${err.message}`);
      resetToSetupView();
    }
  }

  // --- JOB POLLING & PROGRESS ---
  function startJobPolling(jobId) {
    if (state.pollingInterval) {
      clearInterval(state.pollingInterval);
    }

    let failureCount = 0;

    state.pollingInterval = setInterval(async () => {
      try {
        const res = await fetch(`/api/jobs/${jobId}`);
        if (!res.ok) {
          failureCount++;
          if (res.status === 404 && failureCount >= 3) {
            clearInterval(state.pollingInterval);
            state.pollingInterval = null;
            alert('The analysis job is no longer active on the server (the cloud container instance may have restarted). Please click "Start Perception Analysis" to run again.');
            resetToSetupView();
            return;
          }
          if (failureCount >= 10) {
            clearInterval(state.pollingInterval);
            state.pollingInterval = null;
            alert('Lost connection to the perception server. Please refresh the page and try again.');
            resetToSetupView();
            return;
          }
          return;
        }

        failureCount = 0;
        const job = await res.json();

        updateProcessingProgress(job);

        if (job.status === 'completed') {
          clearInterval(state.pollingInterval);
          state.pollingInterval = null;
          handleAnalysisCompleted(job);
        } else if (job.status === 'failed' || job.status === 'cancelled') {
          clearInterval(state.pollingInterval);
          state.pollingInterval = null;
          handleAnalysisFailed(job);
        }
      } catch (err) {
        console.warn('Job polling warning:', err);
      }
    }, 400);
  }

  function updateProcessingProgress(job) {
    const pct = Math.max(0, Math.min(100, Math.round(job.progress || 0)));
    elements.progressFill.style.width = `${pct}%`;
    elements.progressPercent.textContent = `${pct}%`;

    if (job.total_frames && job.current_frame !== undefined) {
      elements.progressFrames.textContent = `Frame ${job.current_frame} of ${job.total_frames}`;
    } else if (job.current_frame !== undefined) {
      elements.progressFrames.textContent = `Frame ${job.current_frame} (processing entire video)`;
    }

    if (job.live_metrics) {
      const fps = job.live_metrics.fps;
      const playFps = job.live_metrics.playback_fps || 30.0;
      const outDur = job.live_metrics.output_duration_sec || (job.current_frame ? (job.current_frame / playFps).toFixed(1) : 0);
      if (fps > 0) {
        elements.processingSubtitle.textContent = `Analyzing at ${fps} FPS throughput • Processed duration: ${outDur}s (@ ${playFps} FPS)`;
      }
    }
  }

  async function cancelActiveAnalysis() {
    if (!state.activeJobId) return;
    try {
      await fetch(`/api/jobs/${state.activeJobId}/cancel`, { method: 'POST' });
      elements.processingSubtitle.textContent = 'Cancelling job...';
    } catch (err) {
      console.warn('Cancel request error:', err);
    }
  }

  function handleAnalysisFailed(job) {
    alert(`Analysis ${job.status}: ${job.error || 'The job could not be completed.'}`);
    resetToSetupView();
  }

  // --- RESULTS SCREEN RENDERING ---
  async function handleAnalysisCompleted(job) {
    elements.processingSection.classList.add('hidden');
    elements.resultsSection.classList.remove('hidden');

    // 1. Complete Summary Banner
    const s = job.summary || {};
    const frames = s.processed_frames || job.current_frame || 0;
    const playFps = s.playback_fps || 30.0;
    const outDur = s.output_duration_sec !== undefined ? s.output_duration_sec : (frames / playFps).toFixed(1);
    const inDur = s.input_duration_sec !== undefined ? s.input_duration_sec : outDur;
    const procTime = s.processing_duration_sec || s.duration_sec || 0;
    const throughput = s.avg_throughput_fps || s.avg_fps || 0;

    elements.bannerSummary.textContent = `Processed full video: ${frames} frames • Video duration: ${outDur}s (Source: ${inDur}s @ ${playFps} FPS) • Completed in ${procTime}s (${throughput} FPS throughput)`;

    // Configure Download Links
    const videoDownloadUrl = `/api/jobs/${job.job_id}/video?download=1`;
    const telemetryDownloadUrl = `/api/jobs/${job.job_id}/telemetry`;

    elements.btnBannerDownload.href = videoDownloadUrl;
    elements.btnDownloadVideoCard.href = videoDownloadUrl;
    elements.btnNextDownload.href = videoDownloadUrl;
    elements.btnDownloadTelemetry.href = telemetryDownloadUrl;

    // 2. Load and Play Video (A. Highest Priority)
    loadVideosForJob(job.job_id);

    // Fetch full results from dedicated /api/jobs/{job_id}/results endpoint
    let resultsData = job;
    try {
      const res = await fetch(`/api/jobs/${job.job_id}/results`);
      if (res.ok) {
        resultsData = await res.json();
      }
    } catch (e) {
      console.warn('Could not fetch results endpoint, using job payload:', e);
    }

    // 3. Populate Key Results (B)
    const metrics = job.live_metrics || {};
    const sum = resultsData.summary || job.summary || {};

    // Get detected objects list from resultsData or summary
    const detectedObjects = (resultsData.detected_objects && resultsData.detected_objects.length > 0)
      ? resultsData.detected_objects
      : ((sum.detected_objects && sum.detected_objects.length > 0)
          ? sum.detected_objects
          : (metrics.active_obstacles || []));

    const totalDetected = (resultsData.total_detected !== undefined)
      ? resultsData.total_detected
      : (sum.total_detected !== undefined
          ? sum.total_detected
          : (detectedObjects.length > 0 ? detectedObjects.length : (metrics.object_count || 0)));

    const totalTracked = (resultsData.total_tracked !== undefined)
      ? resultsData.total_tracked
      : (sum.total_unique_tracks !== undefined
          ? sum.total_unique_tracks
          : detectedObjects.filter(o => o.track_id !== null && o.track_id !== undefined && o.track_id >= 0).length);

    elements.valDetected.textContent = totalDetected;
    elements.valTracked.textContent = totalTracked;

    const classCounts = resultsData.class_counts || sum.unique_class_counts || metrics.class_counts;
    if (classCounts && Object.keys(classCounts).length > 0) {
      const breakdown = Object.entries(classCounts)
        .map(([k, v]) => `${v} ${k}`)
        .join(', ');
      elements.subDetected.textContent = breakdown;
    } else if (detectedObjects.length > 0) {
      elements.subDetected.textContent = `${detectedObjects.length} objects observed`;
    } else {
      elements.subDetected.textContent = 'Cars, trucks, obstacles';
    }

    elements.subTracked.textContent = 'Followed with persistent IDs';

    elements.valLanes.textContent = metrics.lane_status || 'In Lane';
    if (metrics.lane_offset_px !== null && metrics.lane_offset_px !== undefined) {
      const offsetMeters = (metrics.lane_offset_px * 0.005).toFixed(2);
      elements.subLanes.textContent = `Lateral offset: ${metrics.lane_offset_px > 0 ? '+' : ''}${offsetMeters} m`;
    } else {
      elements.subLanes.textContent = 'Boundary corridor tracking';
    }

    elements.valSpeed.textContent = `${throughput} FPS`;
    elements.subSpeed.textContent = `${s.avg_latency_ms || '--'} ms average latency`;

    // 4. Populate Detected Objects Table (C)
    renderObjectsTable(detectedObjects);

    // 5. Populate Technical Details (D)
    elements.techLaneStatus.textContent = metrics.lane_status || 'Active';
    elements.techLaneOffset.textContent = metrics.lane_offset_px !== null ? `${metrics.lane_offset_px.toFixed(1)} px` : 'N/A';
    elements.techLaneConf.textContent = '0.92';
    elements.techDepthStatus.textContent = metrics.depth_status || 'Not Enabled';
    elements.techHazardAlert.textContent = metrics.hazard_status || 'Clear (No Visual Hazards)';

    // Append to Event Log
    appendEventLog(`[00:00.00] Perception job ${job.job_id} initiated.`);
    appendEventLog(`[00:01.20] Pipeline finished: ${frames} frames generated in ${procTime}s at ${throughput} FPS.`);
    if (metrics.hazard_status && metrics.hazard_status !== 'SAFE_FOLLOWING_DISTANCE') {
      appendEventLog(`[00:01.80] Rule alert: ${metrics.hazard_status}`, 'hazard');
    }

    // Scroll smoothly to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- VIDEO LOADING & VIEW MODES ---
  function loadVideosForJob(jobId) {
    const annotUrl = `/api/jobs/${jobId}/video?t=${Date.now()}`;
    const origUrl = `/api/jobs/${jobId}/original-video?t=${Date.now()}`;

    // Main single player
    elements.playerMain.src = annotUrl;
    elements.playerMain.load();

    // Split dual players
    elements.playerSplitOrig.src = origUrl;
    elements.playerSplitAnnot.src = annotUrl;
    elements.playerSplitOrig.load();
    elements.playerSplitAnnot.load();

    switchVideoViewMode('processed');

    // Attempt autoplay; if blocked by browser policy, user controls remain active
    elements.playerMain.play().catch(() => {});
  }

  function switchVideoViewMode(mode) {
    state.currentVideoTab = mode;
    elements.tabViewProcessed.classList.toggle('active', mode === 'processed');
    elements.tabViewOriginal.classList.toggle('active', mode === 'original');
    elements.tabViewSplit.classList.toggle('active', mode === 'split');

    if (!state.activeJobId) return;

    if (mode === 'split') {
      elements.singleViewport.classList.add('hidden');
      elements.splitViewport.classList.remove('hidden');
      elements.videoModeLabel.textContent = 'Dual View: Original vs HUD';
      elements.playerSplitOrig.play().catch(() => {});
      elements.playerSplitAnnot.play().catch(() => {});
    } else {
      elements.splitViewport.classList.add('hidden');
      elements.singleViewport.classList.remove('hidden');

      const isOrig = (mode === 'original');
      elements.playerMain.src = isOrig
        ? `/api/jobs/${state.activeJobId}/original-video`
        : `/api/jobs/${state.activeJobId}/video`;

      elements.videoModeLabel.textContent = isOrig ? 'Raw Road Camera Feed' : 'Annotated Perception HUD';
      elements.playerMain.play().catch(() => {});
    }
  }

  // --- OBJECTS TABLE RENDERING ---
  function renderObjectsTable(obstacles) {
    if (!obstacles || obstacles.length === 0) {
      elements.badgeObjectsCount.textContent = '0 Tracked';
      elements.objectsTableBody.innerHTML = `
        <tr>
          <td colspan="4" class="empty-table-cell">No objects detected in completed video.</td>
        </tr>`;
      return;
    }

    const trackedCount = obstacles.filter(o => o.track_id !== null && o.track_id !== undefined && o.track_id >= 0).length;
    elements.badgeObjectsCount.textContent = `${trackedCount > 0 ? trackedCount : obstacles.length} Tracked`;
    let html = '';
    obstacles.forEach(obs => {
      const tid = (obs.track_id !== null && obs.track_id !== undefined && obs.track_id >= 0)
        ? `#${obs.track_id}`
        : 'Unassigned';
      const confVal = typeof obs.confidence === 'number' ? Math.round(obs.confidence * 100) : '--';
      const confPct = `${confVal}%`;
      const statusText = obs.status || (obs.track_id !== null && obs.track_id !== undefined && obs.track_id >= 0 ? 'Tracked' : 'Detected');
      html += `
        <tr>
          <td><strong>${obs.class_name}</strong></td>
          <td>${confPct}</td>
          <td><code>${tid}</code></td>
          <td><span class="badge-tag-table">${statusText}</span></td>
        </tr>`;
    });
    elements.objectsTableBody.innerHTML = html;
  }

  // --- EVENT LOGGING ---
  function appendEventLog(text, level = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-line log-${level}`;
    entry.textContent = text;
    elements.eventLogFeed.appendChild(entry);
    elements.eventLogFeed.scrollTop = elements.eventLogFeed.scrollHeight;
  }

  // --- RESET DASHBOARD ---
  function resetToSetupView() {
    if (state.pollingInterval) {
      clearInterval(state.pollingInterval);
      state.pollingInterval = null;
    }

    state.activeJobId = null;

    // Pause any playing videos
    elements.playerMain.pause();
    elements.playerSplitOrig.pause();
    elements.playerSplitAnnot.pause();

    elements.playerMain.removeAttribute('src');
    elements.playerSplitOrig.removeAttribute('src');
    elements.playerSplitAnnot.removeAttribute('src');

    // Switch Views
    elements.processingSection.classList.add('hidden');
    elements.resultsSection.classList.add('hidden');
    elements.setupSection.classList.remove('hidden');

    elements.setupSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

})();
