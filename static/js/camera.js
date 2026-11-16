/**
 * camera.js
 * ---------
 * Manages live video stream, RTSP camera connections, multi-camera switching,
 * image upload identification, and camera controls.
 */

document.addEventListener('DOMContentLoaded', () => {
  initCameraControls();
  initUploadIdentify();
});

function initCameraControls() {
  const streamImg = document.getElementById('cameraStream');
  const startBtn = document.getElementById('btnStartCamera');
  const stopBtn = document.getElementById('btnStopCamera');
  const disconnectBtn = document.getElementById('btnDisconnectCamera');
  const fullscreenBtn = document.getElementById('btnFullscreen');
  const refreshBtn = document.getElementById('btnRefresh');
  const fallbackUi = document.getElementById('cameraFallback');

  if (!streamImg) return;

  // Stream error handler
  streamImg.onerror = () => {
    if (fallbackUi) fallbackUi.style.display = 'flex';
    const statusBadge = document.getElementById('cameraStatusBadge');
    if (statusBadge) {
      statusBadge.textContent = 'OFFLINE';
      statusBadge.className = 'badge badge-unknown';
    }
  };

  streamImg.onload = () => {
    if (fallbackUi) fallbackUi.style.display = 'none';
    const statusBadge = document.getElementById('cameraStatusBadge');
    if (statusBadge) {
      statusBadge.textContent = 'STREAMING';
      statusBadge.className = 'badge badge-known';
    }
  };

  // 1. Start Camera
  if (startBtn) {
    startBtn.addEventListener('click', async () => {
      try {
        startBtn.disabled = true;
        const res = await fetch('/api/camera/start', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          streamImg.src = '/video_feed?t=' + Date.now();
          if (fallbackUi) fallbackUi.style.display = 'none';
          showToast('Camera stream active', 'success');
        }
      } catch (e) {
        showToast('Error starting camera', 'danger');
      } finally {
        startBtn.disabled = false;
      }
    });
  }

  // 2. Stop Camera
  if (stopBtn) {
    stopBtn.addEventListener('click', async () => {
      try {
        stopBtn.disabled = true;
        const res = await fetch('/api/camera/stop', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          streamImg.src = '';
          if (fallbackUi) fallbackUi.style.display = 'flex';
          showToast('Camera feed paused', 'info');
        }
      } catch (e) {
        showToast('Error stopping camera', 'danger');
      } finally {
        stopBtn.disabled = false;
      }
    });
  }

  // 3. Disconnect Camera
  if (disconnectBtn) {
    disconnectBtn.addEventListener('click', async () => {
      if (!confirm('Disconnect and release camera stream resources?')) return;
      try {
        disconnectBtn.disabled = true;
        const res = await fetch('/api/camera/disconnect', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          streamImg.src = '';
          if (fallbackUi) fallbackUi.style.display = 'flex';
          const statusBadge = document.getElementById('cameraStatusBadge');
          if (statusBadge) {
            statusBadge.textContent = 'DISCONNECTED';
            statusBadge.className = 'badge badge-secondary';
          }
          showToast('Camera stream disconnected and resources released', 'info');
        }
      } catch (e) {
        showToast('Error disconnecting camera', 'danger');
      } finally {
        disconnectBtn.disabled = false;
      }
    });
  }

  // 4. Refresh Stream
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      streamImg.src = '/video_feed?t=' + Date.now();
      showToast('Video stream refreshed', 'info');
    });
  }

  // 5. Fullscreen Toggle
  if (fullscreenBtn) {
    fullscreenBtn.addEventListener('click', () => {
      const wrap = document.getElementById('cameraScreenWrapper');
      if (!wrap) return;
      if (!document.fullscreenElement) {
        wrap.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });
  }

  // Periodically update camera telemetry
  setInterval(async () => {
    try {
      const res = await fetch('/api/camera/status');
      if (!res.ok) return;
      const data = await res.json();
      if (data.success && data.camera) {
        const c = data.camera;
        const activeCam = c.camera || {};
        const fpsEl = document.getElementById('liveFps');
        if (fpsEl && c.fps !== undefined) {
          fpsEl.textContent = c.fps.toFixed(1);
        }

        const badge = document.getElementById('cameraStatusBadge');
        if (badge && activeCam.connection_status) {
          badge.textContent = activeCam.connection_status;
          if (activeCam.connection_status.includes('CONNECTED')) {
            badge.className = 'badge badge-known';
          } else if (activeCam.connection_status === 'RECONNECTING') {
            badge.className = 'badge badge-secondary';
          } else {
            badge.className = 'badge badge-unknown';
          }
        }
      }
    } catch (e) {}
  }, 3000);
}

// ---------------------------------------------------------
// Multi-Camera Switching
// ---------------------------------------------------------
async function switchCCTVCamera(cameraId) {
  try {
    const res = await fetch('/api/camera/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: cameraId })
    });
    const data = await res.json();
    if (data.success) {
      const streamImg = document.getElementById('cameraStream');
      if (streamImg) {
        streamImg.src = `/video_feed/${cameraId}?t=` + Date.now();
      }

      const activeCam = (data.camera && data.camera.camera) || {};
      const nameEl = document.getElementById('cameraDisplayName');
      if (nameEl) {
        nameEl.textContent = `${activeCam.name || cameraId} (${activeCam.location || 'Campus'})`;
      }
      showToast(`Switched to camera: ${cameraId}`, 'success');
    } else {
      showToast(data.error || 'Failed to switch camera', 'danger');
    }
  } catch (e) {
    showToast('Camera switch request failed', 'danger');
  }
}

// ---------------------------------------------------------
// RTSP Connection Modal Functions (Module 2)
// ---------------------------------------------------------
async function testRtspConnection() {
  const urlInput = document.getElementById('rtspUrlInput');
  const resEl = document.getElementById('rtspTestResult');
  const btn = document.getElementById('btnTestRtsp');
  if (!urlInput || !resEl) return;

  const url = urlInput.value.trim();
  if (!url) {
    resEl.innerHTML = '<span style="color: #EF4444;">Please enter an RTSP URL or camera index first.</span>';
    return;
  }

  btn.disabled = true;
  btn.textContent = '⏳ Testing Stream...';
  resEl.innerHTML = '<span style="color: #2563EB;">Connecting to RTSP stream and decoding initial frames...</span>';

  try {
    const res = await fetch('/api/camera/verify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ rtsp_url: url })
    });
    const data = await res.json();

    if (data.success) {
      const info = data.stream_info || {};
      resEl.innerHTML = `
        <div style="padding: 8px 12px; background: #ECFDF5; border-radius: 8px; border: 1px solid #A7F3D0; color: #065F46;">
          <strong>✓ Connection Verified!</strong><br>
          Resolution: <strong>${info.width}x${info.height}</strong> • FPS: <strong>${info.fps}</strong>
        </div>
      `;
    } else {
      resEl.innerHTML = `
        <div style="padding: 8px 12px; background: #FEF2F2; border-radius: 8px; border: 1px solid #FECACA; color: #991B1B;">
          <strong>✕ Connection Failed:</strong> ${data.message || data.error}
        </div>
      `;
    }
  } catch (e) {
    resEl.innerHTML = `<span style="color: #EF4444;">Network error while verifying camera: ${e.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '⚡ Test Connection';
  }
}

async function handleRtspConnect(event) {
  event.preventDefault();
  const camId = document.getElementById('rtspCamId').value.trim();
  const camName = document.getElementById('rtspCamName').value.trim();
  const camLoc = document.getElementById('rtspCamLocation').value.trim();
  const rtspUrl = document.getElementById('rtspUrlInput').value.trim();
  const btn = document.getElementById('btnSubmitRtsp');
  const resEl = document.getElementById('rtspTestResult');

  if (!camId || !rtspUrl) {
    alert('Camera ID and RTSP URL are required');
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Connecting...';

  try {
    const res = await fetch('/api/camera/connect', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        camera_id: camId,
        name: camName,
        location: camLoc,
        rtsp_url: rtspUrl
      })
    });
    const data = await res.json();

    if (data.success) {
      showToast(data.message, 'success');
      // Update dropdown option or add if new
      const dropdown = document.getElementById('cameraSelectDropdown');
      if (dropdown) {
        let exists = false;
        for (let opt of dropdown.options) {
          if (opt.value === camId) {
            exists = true;
            opt.selected = true;
            break;
          }
        }
        if (!exists) {
          const opt = document.createElement('option');
          opt.value = camId;
          opt.textContent = `${camName} (${camId})`;
          opt.selected = true;
          dropdown.appendChild(opt);
        }
      }

      // Update stream img
      const streamImg = document.getElementById('cameraStream');
      if (streamImg) {
        streamImg.src = `/video_feed/${camId}?t=` + Date.now();
      }

      const nameEl = document.getElementById('cameraDisplayName');
      if (nameEl) {
        nameEl.textContent = `${camName} (${camLoc})`;
      }

      closeRtspModal();
    } else {
      resEl.innerHTML = `<span style="color: #EF4444;">${data.error || 'Connection failed'}</span>`;
    }
  } catch (e) {
    resEl.innerHTML = `<span style="color: #EF4444;">Error: ${e.message}</span>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔗 Connect & Stream';
  }
}

// ---------------------------------------------------------
// Image Upload & Identify (Module 3 & 4)
// ---------------------------------------------------------
function initUploadIdentify() {
  const uploadBtn = document.getElementById('btnUploadIdentify');
  const fileInput = document.getElementById('inputUploadIdentify');
  if (!uploadBtn || !fileInput) return;

  uploadBtn.addEventListener('click', () => {
    fileInput.click();
  });

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file extension on client-side first
    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (!['.jpg', '.jpeg', '.png', '.webp'].includes(ext)) {
      showToast(`Invalid file type '${ext}'. Please upload a JPG, JPEG, PNG, or WEBP image.`, 'danger');
      fileInput.value = '';
      return;
    }

    const formData = new FormData();
    formData.append('image', file);

    uploadBtn.disabled = true;
    showToast('Analyzing uploaded image with YOLOv8 & Deep Face Embeddings...', 'info');

    try {
      const res = await fetch('/api/identify_image', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (!data.success) {
        showToast(data.error || 'Failed to analyze uploaded image', 'danger');
        return;
      }

      if (data.results && data.results.length > 0) {
        const modal = document.getElementById('detectionDetailModal');
        const body = document.getElementById('detModalBody');
        const title = document.getElementById('detModalTitle');
        const subtitle = document.getElementById('detModalSubtitle');

        title.textContent = `Upload Analysis: ${data.entities_count} Entity Detected`;
        subtitle.textContent = `Processed with Ultralytics YOLOv8 & Deep Feature Cosine Embeddings • ${data.timestamp || ''}`;

        const annUrl = data.annotated_image_url || data.original_image_url;
        const origUrl = data.original_image_url;

        body.innerHTML = `
          <!-- Image Views: Annotated & Original -->
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
            <div>
              <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; margin-bottom: 4px; text-transform: uppercase;">Annotated Result Image</div>
              <a href="${annUrl}" target="_blank" title="Click to view full resolution">
                <img src="${annUrl}" style="width: 100%; height: 180px; object-fit: contain; background: #0F172A; border-radius: 10px; border: 2px solid #3B82F6;">
              </a>
            </div>
            <div>
              <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; margin-bottom: 4px; text-transform: uppercase;">Original Uploaded Image</div>
              <a href="${origUrl}" target="_blank" title="Click to view full resolution">
                <img src="${origUrl}" style="width: 100%; height: 180px; object-fit: contain; background: #0F172A; border-radius: 10px; border: 1px solid #E2E8F0;">
              </a>
            </div>
          </div>

          <!-- Detected Entities List -->
          <h4 style="font-size: 0.9rem; font-weight: 700; color: #0F172A; margin-bottom: 8px;">Detections & Identification Summary</h4>
          <div style="display: flex; flex-direction: column; gap: 10px; max-height: 240px; overflow-y: auto; padding-right: 4px; margin-bottom: 16px;">
            ${data.results.map((ent, idx) => {
              const isPerson = ent.class_name === 'person';
              const isKnown = ent.status === 'known';
              const isPhone = ent.class_name === 'mobile phone' || ent.class_name === 'cell phone';
              const isObject = ent.status === 'object';
              const name = ent.name || ent.class_name.toUpperCase();

              let badgeClass = 'badge-unknown';
              let badgeLabel = 'UNKNOWN VISITOR';
              let borderStyle = 'border-left: 4px solid #EF4444;';

              if (isKnown) {
                badgeClass = 'badge-known';
                badgeLabel = `KNOWN | ${ent.person_id || 'REGISTERED'}`;
                borderStyle = 'border-left: 4px solid #10B981;';
              } else if (isPhone) {
                badgeClass = 'badge-secondary';
                badgeLabel = 'MOBILE PHONE DETECTED';
                borderStyle = 'border-left: 4px solid #F59E0B;';
              } else if (isObject) {
                badgeClass = 'badge-secondary';
                badgeLabel = (ent.class_name || 'OBJECT').toUpperCase();
                borderStyle = 'border-left: 4px solid #8B5CF6;';
              }

              const simScore = ent.similarity_score !== undefined ? `${Math.round(ent.similarity_score * 100)}%` : 'N/A';
              const yoloConf = ent.yolo_confidence !== undefined ? `${Math.round(ent.yolo_confidence * 100)}%` : `${Math.round((ent.confidence || 0.8) * 100)}%`;
              const faceConf = ent.face_confidence !== undefined && ent.face_confidence > 0 ? `${Math.round(ent.face_confidence * 100)}%` : 'N/A';

              return `
                <div style="padding: 12px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; ${borderStyle}">
                  <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                    <div>
                      <strong style="color: #0F172A; font-size: 0.95rem;">${idx + 1}. ${escapeHtml(name)}</strong>
                      <div style="font-size: 0.78rem; color: #64748B;">
                        Role: <strong>${escapeHtml(ent.role || 'Visitor')}</strong> • Dept: <strong>${escapeHtml(ent.department || 'General')}</strong>
                      </div>
                    </div>
                    <span class="badge ${badgeClass}" style="font-size: 0.75rem;">${badgeLabel}</span>
                  </div>

                  <!-- Confidence metrics bar -->
                  <div style="display: flex; gap: 12px; background: #FFFFFF; padding: 6px 10px; border-radius: 6px; border: 1px solid #E2E8F0; font-size: 0.74rem;">
                    <span>YOLO Object Conf: <strong style="color: #D97706;">${yoloConf}</strong></span>
                    ${isPerson ? `<span>Face Detect Conf: <strong style="color: #2563EB;">${faceConf}</strong></span>` : ''}
                    ${isPerson ? `<span>Cosine Similarity: <strong style="color: ${isKnown ? '#16A34A' : '#DC2626'};">${simScore}</strong></span>` : ''}
                  </div>
                </div>
              `;
            }).join('')}
          </div>

          <div style="display: flex; justify-content: flex-end;">
            <button class="btn btn-secondary" onclick="closeDetectionDetailModal()">Close</button>
          </div>
        `;

        modal.style.display = 'flex';
        showToast(`Analysis complete! Identified ${data.entities_count} entities`, 'success');

        // Trigger dashboard polling refresh
        if (typeof fetchStats === 'function') fetchStats();
      } else {
        showToast('No recognizable persons or objects found in uploaded photo', 'info');
      }
    } catch (err) {
      showToast('Error analyzing uploaded image: ' + err.message, 'danger');
    } finally {
      uploadBtn.disabled = false;
      fileInput.value = '';
    }
  });
}
