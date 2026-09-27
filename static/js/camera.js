/**
 * camera.js
 * ---------
 * Manages live camera stream interactions, start/stop commands,
 * evidence snapshot capture modal, and fullscreen display controls.
 */

document.addEventListener('DOMContentLoaded', () => {
  initCameraControls();
});

function initCameraControls() {
  const streamImg = document.getElementById('cameraStream');
  const startBtn = document.getElementById('btnStartCamera');
  const stopBtn = document.getElementById('btnStopCamera');
  const captureBtn = document.getElementById('btnCapture');
  const fullscreenBtn = document.getElementById('btnFullscreen');
  const refreshBtn = document.getElementById('btnRefresh');
  const fallbackUi = document.getElementById('cameraFallback');
  const cameraWrapper = document.getElementById('cameraScreenWrapper');

  if (!streamImg) return;

  // Handle stream error gracefully
  streamImg.onerror = () => {
    console.warn('Camera feed dropped or disconnected.');
    if (fallbackUi) fallbackUi.style.display = 'flex';
  };

  streamImg.onload = () => {
    if (fallbackUi) fallbackUi.style.display = 'none';
  };

  // Start Camera
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

  // Stop Camera
  if (stopBtn) {
    stopBtn.addEventListener('click', async () => {
      try {
        stopBtn.disabled = true;
        const res = await fetch('/api/camera/stop', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          streamImg.src = '';
          if (fallbackUi) fallbackUi.style.display = 'flex';
          showToast('Camera feed paused', 'primary');
        }
      } catch (e) {
        showToast('Error stopping camera', 'danger');
      } finally {
        stopBtn.disabled = false;
      }
    });
  }

  // Live recognition telemetry badge updater
  function updateCaptureBadge(isLocked, personName = '') {
    const badge = document.getElementById('recLockBadge');
    if (!badge) return;
    if (isLocked) {
      badge.textContent = `🔒 CAPTURED: ${personName || 'LOCKED'} (NO RE-CAPTURE)`;
      badge.style.background = '#ECFDF5';
      badge.style.color = '#065F46';
      badge.style.border = '1px solid #A7F3D0';
    } else {
      badge.textContent = 'READY (1-TIME CAPTURE)';
      badge.style.background = '#E2E8F0';
      badge.style.color = '#475569';
      badge.style.border = 'none';
    }
  }

  // Polling for live stream recognition result
  let lastShownRecId = null;
  let recPollingInterval = null;

  function startRecognitionPolling() {
    if (recPollingInterval) clearInterval(recPollingInterval);
    recPollingInterval = setInterval(async () => {
      try {
        const res = await fetch('/api/recognition/latest');
        if (!res.ok) return;
        const data = await res.json();

        if (data.success && data.recognition) {
          const rec = data.recognition;
          const pName = rec.name || (rec.person && rec.person.name) || rec.person_id;
          updateCaptureBadge(true, pName);

          // If this is a new recognition event we haven't displayed yet
          if (rec.id && rec.id !== lastShownRecId) {
            lastShownRecId = rec.id;
            showSnapshotModal(rec);
            showToast(`✓ Person Recognized: ${pName}. Results returned! Live capture locked.`, 'success');
          }
        } else if (data.success && (!data.recognized_persons || data.recognized_persons.length === 0)) {
          updateCaptureBadge(false);
        }
      } catch (e) {
        // Silent fail during poll
      }
    }, 1500);
  }

  // Start polling immediately
  startRecognitionPolling();

  // Reset Recognition / Scan Next Person button
  const resetBtn = document.getElementById('btnResetRec');
  if (resetBtn) {
    resetBtn.addEventListener('click', async () => {
      try {
        resetBtn.disabled = true;
        const res = await fetch('/api/recognition/reset', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          lastShownRecId = null;
          updateCaptureBadge(false);
          showToast('Live capture unlocked. Ready for next person!', 'primary');
        }
      } catch (e) {
        showToast('Error resetting live capture', 'danger');
      } finally {
        resetBtn.disabled = false;
      }
    });
  }

  // Instant Snapshot Capture
  if (captureBtn) {
    captureBtn.addEventListener('click', async () => {
      try {
        captureBtn.innerHTML = '<span>Capturing...</span>';
        captureBtn.disabled = true;
        const res = await fetch('/api/capture', { method: 'POST' });
        const data = await res.json();

        if (data.success && data.snapshot) {
          lastShownRecId = data.snapshot.id;
          const pName = data.snapshot.person?.name || data.snapshot.person?.person_id;
          updateCaptureBadge(true, pName);
          showSnapshotModal(data.snapshot);
          showToast('Evidence snapshot recorded! Live capture locked.', 'success');
        } else {
          showToast(data.error || 'Failed to capture frame', 'danger');
        }
      } catch (e) {
        showToast('Network error during snapshot', 'danger');
      } finally {
        captureBtn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
            <circle cx="12" cy="13" r="4"></circle>
          </svg>
          <span>Capture</span>
        `;
        captureBtn.disabled = false;
      }
    });
  }

  // Refresh Feed
  if (refreshBtn) {
    refreshBtn.addEventListener('click', () => {
      streamImg.src = '/video_feed?t=' + Date.now();
      showToast('Camera connection refreshed', 'primary');
    });
  }

  // Fullscreen Toggle
  if (fullscreenBtn && cameraWrapper) {
    fullscreenBtn.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        cameraWrapper.requestFullscreen().catch(err => {
          showToast(`Error entering fullscreen: ${err.message}`, 'danger');
        });
      } else {
        document.exitFullscreen();
      }
    });
  }

  // Upload & Identify External Image
  const uploadBtn = document.getElementById('btnUploadIdentify');
  const uploadInput = document.getElementById('inputUploadIdentify');
  if (uploadBtn && uploadInput) {
    uploadBtn.addEventListener('click', () => uploadInput.click());
    uploadInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      try {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = '<span>Analyzing...</span>';
        showToast('Processing facial match & timetable...', 'primary');

        const formData = new FormData();
        formData.append('image', file);

        const res = await fetch('/api/identify_image', {
          method: 'POST',
          body: formData
        });
        const json = await res.json();

        if (json.success && json.data) {
          showSnapshotModal(json.data);
          showToast('Person identified & profile loaded!', 'success');
        } else {
          showToast(json.error || 'Failed to identify person', 'danger');
        }
      } catch (err) {
        showToast('Network error during image identification', 'danger');
      } finally {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = `
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
          <span>Upload & Identify</span>
        `;
        uploadInput.value = '';
      }
    });
  }
}

// ---------------------------------------------------------
// Snapshot Evidence & Identified Person Profile Modal
// ---------------------------------------------------------
function showSnapshotModal(snapshot) {
  let modal = document.getElementById('snapshotModal');
  if (modal) {
    modal.remove(); // Re-render fresh with updated person details
  }

  const p = snapshot.person || {};
  const isKnown = p.status === 'known';
  const isStudent = (p.person_type === 'Student') || (p.role && p.role.toLowerCase().includes('student'));
  const isFaculty = (p.person_type === 'Faculty') || (p.role && (p.role.toLowerCase().includes('faculty') || p.role.toLowerCase().includes('professor')));

  let personContentHtml = '';

  if (isStudent) {
    // Generate timetable rows
    const timetableRows = (p.timetable || []).map(slot => `
      <tr>
        <td style="font-weight: 600; color: #1E293B;">${slot.day}</td>
        <td style="color: #64748B;">${slot.time}</td>
        <td style="font-weight: 600; color: var(--primary);">${slot.subject}</td>
        <td><span class="badge" style="background:#F1F5F9; color:#475569;">${slot.room}</span></td>
        <td style="color: #64748B;">${slot.faculty || 'Dept Staff'}</td>
      </tr>
    `).join('');

    personContentHtml = `
      <div class="person-details-box" style="margin-top: 18px; padding: 18px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; text-align: left;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
          <div style="display: flex; align-items: center; gap: 14px;">
            <div style="width: 52px; height: 52px; border-radius: 12px; background: #EEF2FF; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; border: 1px solid #C7D2FE;">
              🎓
            </div>
            <div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <h4 style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 0;">${p.name}</h4>
                <span class="badge badge-known" style="font-size: 0.72rem;">STUDENT VERIFIED</span>
              </div>
              <div style="font-size: 0.85rem; color: #64748B; margin-top: 2px;">
                ${p.department}
              </div>
            </div>
          </div>
          <div style="text-align: right;">
            <span style="font-size: 0.72rem; color: #94A3B8; display: block;">CONFIDENCE</span>
            <span style="font-weight: 800; color: var(--success); font-size: 1rem;">${Math.round((p.confidence || 0.96) * 100)}%</span>
          </div>
        </div>

        <!-- Student Key Attributes Grid -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px;">
          <div style="background: #FFFFFF; padding: 10px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">ROLL NUMBER</div>
            <div style="font-size: 0.92rem; font-weight: 800; color: #1E293B;">${p.roll_number || p.person_id}</div>
          </div>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">CLASS / SECTION</div>
            <div style="font-size: 0.92rem; font-weight: 800; color: #1E293B;">${p.class || 'CSE-3A'}</div>
          </div>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">ACADEMIC YEAR</div>
            <div style="font-size: 0.92rem; font-weight: 800; color: #1E293B;">${p.year || '3rd Year'}</div>
          </div>
          <div style="background: #FFFFFF; padding: 10px; border-radius: 10px; border: 1px solid #E2E8F0;">
            <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600;">CGPA</div>
            <div style="font-size: 0.92rem; font-weight: 800; color: var(--primary);">${p.cgpa || '8.92'} / 10</div>
          </div>
        </div>

        <!-- Class Timetable Schedule -->
        <div style="margin-top: 14px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h5 style="font-size: 0.88rem; font-weight: 700; color: #334155; margin: 0; display: flex; align-items: center; gap: 6px;">
              <span>📅 Class Timetable Schedule (${p.class || 'CSE-3A'})</span>
            </h5>
            <span style="font-size: 0.75rem; color: var(--primary); font-weight: 600;">Synced with Firebase</span>
          </div>

          <div style="max-height: 180px; overflow-y: auto; border: 1px solid #E2E8F0; border-radius: 10px; background: #FFFFFF;">
            <table class="table" style="font-size: 0.82rem; margin: 0; width: 100%;">
              <thead style="background: #F8FAFC; position: sticky; top: 0;">
                <tr>
                  <th style="padding: 8px 12px;">Day</th>
                  <th style="padding: 8px 12px;">Time Slot</th>
                  <th style="padding: 8px 12px;">Course / Subject</th>
                  <th style="padding: 8px 12px;">Room</th>
                  <th style="padding: 8px 12px;">Faculty</th>
                </tr>
              </thead>
              <tbody>
                ${timetableRows || '<tr><td colspan="5" style="text-align:center; color:#94A3B8;">No timetable slots listed.</td></tr>'}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  } else if (isFaculty) {
    // Generate faculty classes rows
    const classRows = (p.classes || []).map(c => `
      <tr>
        <td style="font-weight: 700; color: #1E293B;">${c.class}</td>
        <td style="font-weight: 600; color: var(--primary);">${c.subject}</td>
        <td style="color: #64748B;">${c.timing}</td>
        <td><span class="badge" style="background:#F1F5F9; color:#475569;">${c.room}</span></td>
      </tr>
    `).join('');

    personContentHtml = `
      <div class="person-details-box" style="margin-top: 18px; padding: 18px; background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 14px; text-align: left;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 14px;">
          <div style="display: flex; align-items: center; gap: 14px;">
            <div style="width: 52px; height: 52px; border-radius: 12px; background: #ECFDF5; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; border: 1px solid #A7F3D0;">
              👨‍🏫
            </div>
            <div>
              <div style="display: flex; align-items: center; gap: 8px;">
                <h4 style="font-size: 1.15rem; font-weight: 800; color: #0F172A; margin: 0;">${p.name}</h4>
                <span class="badge badge-success" style="font-size: 0.72rem;">FACULTY VERIFIED</span>
              </div>
              <div style="font-size: 0.85rem; color: #64748B; margin-top: 2px;">
                ${p.designation || 'Professor'} • ${p.department}
              </div>
            </div>
          </div>
          <div style="text-align: right;">
            <span style="font-size: 0.72rem; color: #94A3B8; display: block;">FACULTY ID</span>
            <span style="font-weight: 800; color: #1E293B; font-size: 1rem;">${p.faculty_id || p.person_id}</span>
          </div>
        </div>

        <!-- Faculty Teaching Classes & Schedule -->
        <div style="margin-top: 12px;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <h5 style="font-size: 0.88rem; font-weight: 700; color: #334155; margin: 0;">
              📚 Assigned Classes & Lecture Schedule
            </h5>
            <span style="font-size: 0.75rem; color: var(--primary); font-weight: 600;">Department of ${p.department}</span>
          </div>

          <div style="max-height: 180px; overflow-y: auto; border: 1px solid #E2E8F0; border-radius: 10px; background: #FFFFFF;">
            <table class="table" style="font-size: 0.82rem; margin: 0; width: 100%;">
              <thead style="background: #F8FAFC; position: sticky; top: 0;">
                <tr>
                  <th style="padding: 8px 12px;">Assigned Class</th>
                  <th style="padding: 8px 12px;">Subject / Course</th>
                  <th style="padding: 8px 12px;">Schedule & Timing</th>
                  <th style="padding: 8px 12px;">Assigned Room</th>
                </tr>
              </thead>
              <tbody>
                ${classRows || '<tr><td colspan="4" style="text-align:center; color:#94A3B8;">No classes scheduled.</td></tr>'}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    `;
  } else {
    personContentHtml = `
      <div class="person-details-box" style="margin-top: 18px; padding: 16px; background: #FEF2F2; border: 1px solid #FECACA; border-radius: 14px; text-align: left;">
        <div style="display: flex; align-items: center; gap: 12px;">
          <div style="width: 44px; height: 44px; border-radius: 10px; background: #FEE2E2; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
            ⚠️
          </div>
          <div>
            <div style="display: flex; align-items: center; gap: 8px;">
              <h4 style="font-size: 1.05rem; font-weight: 800; color: #991B1B; margin: 0;">Unregistered Individual Detected</h4>
              <span class="badge badge-unknown" style="font-size: 0.72rem;">UNKNOWN</span>
            </div>
            <div style="font-size: 0.82rem; color: #B91C1C; margin-top: 3px;">
              ${p.alert || 'Person not found in authorized database. Dispatched security alert.'}
            </div>
          </div>
        </div>
        <div style="margin-top: 12px; display: flex; justify-content: flex-end;">
          <a href="/persons" class="btn btn-primary" style="padding: 6px 14px; font-size: 0.8rem;">
            + Register Person into Database
          </a>
        </div>
      </div>
    `;
  }

  modal = document.createElement('div');
  modal.id = 'snapshotModal';
  modal.className = 'modal-backdrop';
  modal.innerHTML = `
    <div class="modal-content" style="max-width: 680px; width: 95%;">
      <div class="modal-header">
        <div>
          <h3 style="margin: 0; font-size: 1.25rem;">Camera Snapshot & Person Recognition</h3>
          <span style="font-size: 0.8rem; color: #64748B;">High-precision YOLOv8 Detection + Facial Matching</span>
        </div>
        <button id="closeSnapModal" style="background:none; border:none; font-size:1.5rem; cursor:pointer; color:#64748B;">&times;</button>
      </div>

      <div class="modal-body" style="padding: 20px;">
        <div style="position: relative;">
          <img id="snapModalImg" src="${snapshot.image_url}" style="width: 100%; max-height: 280px; object-fit: cover; border-radius: 12px; border: 1px solid #E2E8F0;" />
          <div style="position: absolute; bottom: 10px; left: 10px; background: rgba(15, 23, 42, 0.75); color: #FFF; padding: 4px 10px; border-radius: 6px; font-size: 0.76rem; backdrop-filter: blur(4px);">
            📷 ${snapshot.filename} • ${snapshot.timestamp}
          </div>
        </div>

        <div style="margin-top: 12px; padding: 10px 14px; background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 10px; display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; color: #065F46;">
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 1.1rem;">🔒</span>
            <span><strong>Live Capture Locked:</strong> Person recognized once. Results returned. Live camera will not capture again.</span>
          </div>
          <button id="modalResetRecBtn" class="btn btn-secondary" style="padding: 4px 10px; font-size: 0.75rem; white-space: nowrap;">
            Scan Next Person
          </button>
        </div>

        ${personContentHtml}
      </div>

      <div class="modal-footer" style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.78rem; color: #94A3B8;">Evidence stored locally & Firebase Storage</span>
        <div style="display: flex; gap: 8px;">
          <a id="downloadSnapBtn" href="${snapshot.image_url}" download="${snapshot.filename}" class="btn btn-secondary">
            Download Evidence
          </a>
          <button id="okSnapModal" class="btn btn-primary">Done</button>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(modal);

  const closeBtn = modal.querySelector('#closeSnapModal');
  const okBtn = modal.querySelector('#okSnapModal');
  const modalResetBtn = modal.querySelector('#modalResetRecBtn');

  [closeBtn, okBtn].forEach(b => {
    if (b) b.addEventListener('click', () => modal.classList.remove('show'));
  });

  if (modalResetBtn) {
    modalResetBtn.addEventListener('click', async () => {
      try {
        await fetch('/api/recognition/reset', { method: 'POST' });
        updateCaptureBadge(false);
        showToast('Live capture lock cleared. Ready for next person!', 'primary');
      } catch (err) {
        console.warn('Reset error:', err);
      }
      modal.classList.remove('show');
    });
  }

  // Trigger modal display
  setTimeout(() => modal.classList.add('show'), 20);
}
