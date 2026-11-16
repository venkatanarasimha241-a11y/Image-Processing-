/**
 * dashboard.js
 * ------------
 * Core JavaScript logic for AI Camera Monitoring Dashboard.
 * Handles real-time KPI telemetry, real dynamic charts, live activity stream,
 * detection evidence modal inspection, and modal dialog controllers.
 */

let cachedDetections = [];

document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initRecentDetectionsPolling();
  initKPIPolling();
  initMobileNav();
  initFirebaseStatus();
  initUiControls();
});

// ---------------------------------------------------------
// Live Header Clock
// ---------------------------------------------------------
function initClock() {
  const clockEl = document.getElementById('liveClock');
  if (!clockEl) return;

  function update() {
    const now = new Date();
    const options = {
      weekday: 'short',
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    };
    clockEl.textContent = now.toLocaleDateString('en-US', options);
  }

  update();
  setInterval(update, 1000);
}

// ---------------------------------------------------------
// Real-time KPI & Analytics Polling
// ---------------------------------------------------------
function initKPIPolling() {
  async function fetchStats() {
    try {
      const res = await fetch('/api/stats');
      if (!res.ok) return;
      const data = await res.json();
      if (!data.success || !data.stats) return;

      const s = data.stats;
      const cam = data.camera || {};

      // 1. Update KPI Values
      const elReg = document.getElementById('kpiRegisteredPersons');
      if (elReg && s.registered_persons_count !== undefined) {
        elReg.textContent = Number(s.registered_persons_count).toLocaleString();
      }

      const elKnown = document.getElementById('kpiKnownPersons');
      if (elKnown && s.known_persons !== undefined) {
        elKnown.textContent = Number(s.known_persons).toLocaleString();
      }

      const elUnknown = document.getElementById('kpiUnknownPersons');
      if (elUnknown && s.unknown_persons !== undefined) {
        elUnknown.textContent = Number(s.unknown_persons).toLocaleString();
      }

      const elObj = document.getElementById('kpiDetectedObjects');
      if (elObj && s.total_detected_objects !== undefined) {
        elObj.textContent = Number(s.total_detected_objects).toLocaleString();
      }

      const elActiveCam = document.getElementById('kpiActiveCameras');
      if (elActiveCam && s.active_cameras) {
        elActiveCam.textContent = s.active_cameras;
      }

      // Update objects badge with specific breakdown
      const elObjBadge = document.getElementById('kpiObjectsBadge');
      if (elObjBadge && s.objects_breakdown) {
        const topObjs = Object.entries(s.objects_breakdown)
          .filter(([k, v]) => k !== 'person' && v > 0)
          .map(([k, v]) => `${k}: ${v}`);
        if (topObjs.length > 0) {
          elObjBadge.textContent = topObjs.slice(0, 2).join(' | ');
        }
      }

      // 2. Update Charts with REAL backend data
      if (s.chart_labels && s.chart_data) {
        renderActivityChart(s.chart_labels, s.chart_data);
      }

      const knownVal = Number(s.known_persons) || 0;
      const unkVal = Number(s.unknown_persons) || 0;
      renderDistributionChart(knownVal, unkVal);

      // Update badges
      const ratioKnown = document.getElementById('ratioKnownBadge');
      const ratioUnknown = document.getElementById('ratioUnknownBadge');
      const totalPersons = knownVal + unkVal;
      if (ratioKnown && totalPersons > 0) {
        const pctKnown = Math.round((knownVal / totalPersons) * 100);
        ratioKnown.textContent = `Known ${pctKnown}%`;
      }
      if (ratioUnknown && totalPersons > 0) {
        const pctUnk = Math.round((unkVal / totalPersons) * 100);
        ratioUnknown.textContent = `Alerts ${pctUnk}%`;
      }

    } catch (e) {
      console.warn('Failed to fetch KPI stats:', e);
    }
  }

  fetchStats();
  setInterval(fetchStats, 3000);
}

// ---------------------------------------------------------
// Real-time Detection Feed Polling & Rendering
// ---------------------------------------------------------
function initRecentDetectionsPolling() {
  const activityList = document.getElementById('recentDetectionsList');

  async function fetchLatest() {
    try {
      const res = await fetch('/api/detections?limit=8');
      if (!res.ok) return;
      const data = await res.json();
      if (!data.success || !data.detections) return;

      cachedDetections = data.detections;
      renderDetections(data.detections);
    } catch (e) {
      console.warn('Failed to fetch detections stream:', e);
    }
  }

  function renderDetections(records) {
    if (!activityList) return;
    if (!records || records.length === 0) {
      activityList.innerHTML = `
        <div style="padding: 32px 16px; text-align: center; color: #94A3B8;">
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom: 8px;">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <div style="font-size: 0.88rem; font-weight: 600;">No surveillance detections recorded yet</div>
          <div style="font-size: 0.78rem; margin-top: 4px;">Camera events will stream here live as persons or objects are identified.</div>
        </div>
      `;
      return;
    }

    activityList.innerHTML = records.map((r, idx) => {
      const status = (r.recognition_status || 'unknown').toLowerCase();
      const isKnown = status === 'known';
      const isObject = status === 'object' || (r.object_class && r.object_class !== 'person');
      const isNoFace = status === 'face_unavailable' || status === 'no_face';

      let badgeClass = 'badge-unknown';
      let badgeLabel = 'SECURITY ALERT';
      let icon = '⚠️';

      if (isKnown) {
        badgeClass = 'badge-known';
        badgeLabel = 'VERIFIED';
        icon = '✓';
      } else if (isObject) {
        badgeClass = 'badge-secondary';
        badgeLabel = (r.object_class || 'OBJECT').toUpperCase();
        icon = '📦';
      } else if (isNoFace) {
        badgeClass = 'badge-secondary';
        badgeLabel = 'FACE UNAVAIL';
        icon = '👤';
      }

      const imgUrl = r.thumbnail_url || r.image_url || '/static/images/default_avatar.svg';
      const score = (r.similarity_score !== undefined && isKnown) ?
        Math.round(r.similarity_score * 100) :
        Math.round((r.yolo_confidence || r.confidence || 0.80) * 100);

      const metricLabel = isKnown ? `Sim: ${score}%` : `Conf: ${score}%`;
      const timeStr = r.timestamp || 'Just now';
      const camName = r.camera_name || r.camera_id || 'Gate Camera';

      return `
        <div class="activity-item" onclick="openDetectionDetailById('${r.id}')" style="cursor: pointer; transition: background 0.15s ease;" title="Click to inspect detection evidence">
          <div class="activity-avatar-wrap">
            <img src="${imgUrl}" alt="${r.person_name || 'Detection'}" onerror="this.src='/static/images/default_avatar.svg'">
            <span class="activity-badge-mini ${isKnown ? 'mini-known' : (isObject ? 'mini-object' : 'mini-unknown')}">${icon}</span>
          </div>
          <div class="activity-content">
            <div class="activity-title-row">
              <span class="activity-name">${escapeHtml(r.person_name || 'Unknown Subject')}</span>
              <span class="badge ${badgeClass}" style="font-size: 0.68rem; padding: 2px 7px;">${badgeLabel}</span>
            </div>
            <div class="activity-meta-row" style="font-size: 0.74rem; color: #64748B; margin-top: 3px; display: flex; justify-content: space-between;">
              <span>📹 ${escapeHtml(camName)} • <strong style="color: #1E293B;">${metricLabel}</strong></span>
              <span>${escapeHtml(timeStr)}</span>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  fetchLatest();
  setInterval(fetchLatest, 2500);
}

// ---------------------------------------------------------
// Detection Evidence Modal Inspection
// ---------------------------------------------------------
function openDetectionDetailById(recordId) {
  const record = cachedDetections.find(d => d.id === recordId);
  if (!record) return;

  const modal = document.getElementById('detectionDetailModal');
  const body = document.getElementById('detModalBody');
  const title = document.getElementById('detModalTitle');
  const subtitle = document.getElementById('detModalSubtitle');
  if (!modal || !body) return;

  const status = (record.recognition_status || 'unknown').toLowerCase();
  const isKnown = status === 'known';
  const isObject = status === 'object' || (record.object_class && record.object_class !== 'person');
  const isNoFace = status === 'face_unavailable' || status === 'no_face';

  title.textContent = record.person_name || 'Detection Event';
  subtitle.textContent = `${record.camera_name || record.camera_id || 'Camera'} • ${record.timestamp || 'Recorded'}`;

  const imgUrl = record.image_url || '/static/images/default_avatar.svg';

  // Distinct separated metrics
  const yoloConf = record.yolo_confidence !== undefined ? `${Math.round(record.yolo_confidence * 100)}%` : 'N/A';
  const faceConf = record.face_confidence !== undefined ? `${Math.round(record.face_confidence * 100)}%` : 'N/A';
  const simScore = (record.similarity_score !== undefined && !isNoFace && !isObject) ? `${Math.round(record.similarity_score * 100)}%` : 'N/A';
  const threshold = record.decision_threshold !== undefined ? `${record.decision_threshold}` : '0.85';

  let timetableHtml = '';
  if (record.timetable && record.timetable.length > 0) {
    timetableHtml = `
      <div style="margin-top: 16px;">
        <h4 style="font-size: 0.88rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">Academic Schedule / Timetable</h4>
        <div style="display: flex; flex-direction: column; gap: 6px;">
          ${record.timetable.map(t => `
            <div style="padding: 8px 12px; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0; font-size: 0.78rem;">
              <strong>${t.day} (${t.time}):</strong> ${t.subject} - <em>${t.room || 'Room TBA'}</em>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  body.innerHTML = `
    <div style="display: flex; gap: 16px; margin-bottom: 18px; align-items: flex-start;">
      <img src="${imgUrl}" style="width: 140px; height: 140px; object-fit: cover; border-radius: 12px; border: 2px solid #E2E8F0; box-shadow: 0 4px 10px rgba(0,0,0,0.06);">
      <div style="flex: 1;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
          <h3 style="margin: 0; font-size: 1.2rem; color: #0F172A;">${escapeHtml(record.person_name || 'Unknown')}</h3>
          <span class="badge ${isKnown ? 'badge-known' : (isObject || isNoFace ? 'badge-secondary' : 'badge-unknown')}">${isKnown ? 'KNOWN VERIFIED' : (isObject ? 'OBJECT DETECTED' : (isNoFace ? 'FACE UNAVAILABLE' : 'UNKNOWN VISITOR'))}</span>
        </div>
        <div style="font-size: 0.82rem; color: #64748B; line-height: 1.6;">
          <div><strong>ID:</strong> ${escapeHtml(record.person_id || 'N/A')}</div>
          <div><strong>Department:</strong> ${escapeHtml(record.department || 'Visitor')}</div>
          <div><strong>Role:</strong> ${escapeHtml(record.role || 'Visitor')}</div>
          <div><strong>Camera:</strong> ${escapeHtml(record.camera_name || record.camera_id)} (${escapeHtml(record.camera_location || 'Campus')})</div>
        </div>
      </div>
    </div>

    <!-- SEPARATED METRICS GRID -->
    <div style="background: #F1F5F9; border-radius: 10px; padding: 12px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; text-align: center;">
      <div>
        <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">YOLO Object Conf</span>
        <div style="font-size: 1.05rem; font-weight: 800; color: #D97706;">${yoloConf}</div>
      </div>
      <div>
        <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Face Detector Conf</span>
        <div style="font-size: 1.05rem; font-weight: 800; color: #2563EB;">${faceConf}</div>
      </div>
      <div>
        <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Cosine Similarity</span>
        <div style="font-size: 1.05rem; font-weight: 800; color: ${isKnown ? '#16A34A' : (isNoFace ? '#64748B' : '#DC2626')};">${simScore}</div>
      </div>
      <div>
        <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Decision Threshold</span>
        <div style="font-size: 1.05rem; font-weight: 800; color: #475569;">≥ ${threshold}</div>
      </div>
    </div>

    ${timetableHtml}

    <div style="margin-top: 18px; display: flex; justify-content: flex-end;">
      <button class="btn btn-secondary" onclick="closeDetectionDetailModal()">Close</button>
    </div>
  `;

  modal.style.display = 'flex';
}

function closeDetectionDetailModal() {
  const modal = document.getElementById('detectionDetailModal');
  if (modal) modal.style.display = 'none';
}

// ---------------------------------------------------------
// Real Dynamic Canvas Charts
// ---------------------------------------------------------
function renderActivityChart(labels, values) {
  const canvas = document.getElementById('activityTrendChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const maxVal = Math.max(5, ...values) * 1.25;
  const w = rect.width;
  const h = rect.height;
  const padBottom = 35;
  const padLeft = 35;
  const padTop = 20;
  const padRight = 20;

  ctx.clearRect(0, 0, w, h);

  // Grid lines
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (h - padTop - padBottom) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
  }

  if (values.length < 2) return;

  const points = values.map((val, idx) => {
    const x = padLeft + (w - padLeft - padRight) * (idx / (values.length - 1));
    const y = h - padBottom - ((val / maxVal) * (h - padTop - padBottom));
    return { x, y };
  });

  // Gradient area
  const grad = ctx.createLinearGradient(0, padTop, 0, h - padBottom);
  grad.addColorStop(0, 'rgba(79, 70, 229, 0.28)');
  grad.addColorStop(1, 'rgba(79, 70, 229, 0.01)');

  ctx.beginPath();
  ctx.moveTo(points[0].x, h - padBottom);
  points.forEach(p => ctx.lineTo(p.x, p.y));
  ctx.lineTo(points[points.length - 1].x, h - padBottom);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Primary line
  ctx.beginPath();
  ctx.strokeStyle = '#4F46E5';
  ctx.lineWidth = 3;
  points.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();

  // Dots & Labels
  points.forEach((p, idx) => {
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#4F46E5';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.fillStyle = '#64748B';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(labels[idx] || '', p.x, h - 12);
  });
}

function renderDistributionChart(knownCount, unknownCount) {
  const canvas = document.getElementById('distributionDonutChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const radius = Math.min(cx, cy) - 20;
  const innerRadius = radius * 0.65;

  const total = (knownCount + unknownCount) || 1;
  const knownPct = Math.round((knownCount / total) * 100);
  const unkPct = 100 - knownPct;

  const data = [
    { label: 'Known', value: (knownCount / total), color: '#22C55E' },
    { label: 'Unknown', value: (unknownCount / total), color: '#EF4444' }
  ];

  let startAngle = -Math.PI / 2;
  ctx.clearRect(0, 0, rect.width, rect.height);

  data.forEach(slice => {
    const sliceAngle = slice.value * (Math.PI * 2);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
    ctx.arc(cx, cy, innerRadius, startAngle + sliceAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = slice.color;
    ctx.fill();
    startAngle += sliceAngle;
  });

  // Center percentage
  ctx.fillStyle = '#1E293B';
  ctx.font = 'bold 22px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${knownPct}%`, cx, cy - 6);

  ctx.fillStyle = '#64748B';
  ctx.font = '11px sans-serif';
  ctx.fillText('Known Match', cx, cy + 14);
}

// ---------------------------------------------------------
// Modal Open / Close Handlers
// ---------------------------------------------------------
function openRtspModal() {
  const modal = document.getElementById('rtspModal');
  if (modal) modal.style.display = 'flex';
}

function closeRtspModal() {
  const modal = document.getElementById('rtspModal');
  if (modal) modal.style.display = 'none';
}

function openThresholdsModal() {
  const modal = document.getElementById('thresholdsModal');
  if (modal) {
    // Load active settings from API
    fetch('/api/settings/thresholds')
      .then(res => res.json())
      .then(data => {
        if (data.success) {
          const recIn = document.getElementById('inputRecThreshold');
          const yoloIn = document.getElementById('inputYoloConf');
          if (recIn) {
            recIn.value = data.recognition_similarity_threshold;
            document.getElementById('labelRecThreshold').textContent = parseFloat(data.recognition_similarity_threshold).toFixed(2);
          }
          if (yoloIn) {
            yoloIn.value = data.yolo_confidence;
            document.getElementById('labelYoloConf').textContent = parseFloat(data.yolo_confidence).toFixed(2);
          }
        }
      })
      .catch(() => {});
    modal.style.display = 'flex';
  }
}

function closeThresholdsModal() {
  const modal = document.getElementById('thresholdsModal');
  if (modal) modal.style.display = 'none';
}

function openFirebaseModal() {
  const modal = document.getElementById('firebaseModal');
  if (modal) modal.style.display = 'flex';
}

function closeFirebaseModal() {
  const modal = document.getElementById('firebaseModal');
  if (modal) modal.style.display = 'none';
}

// ---------------------------------------------------------
// Thresholds Save Handler
// ---------------------------------------------------------
async function saveThresholds(event) {
  event.preventDefault();
  const recVal = parseFloat(document.getElementById('inputRecThreshold').value);
  const yoloVal = parseFloat(document.getElementById('inputYoloConf').value);

  try {
    const res = await fetch('/api/settings/thresholds', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        recognition_similarity: recVal,
        yolo_confidence: yoloVal
      })
    });
    const data = await res.json();
    if (data.success) {
      showToast(`Thresholds updated! Face: ${recVal.toFixed(2)}, YOLO: ${yoloVal.toFixed(2)}`, 'success');
      closeThresholdsModal();
    }
  } catch (e) {
    showToast('Failed to update thresholds', 'danger');
  }
}

// ---------------------------------------------------------
// Mobile Nav & Utilities
// ---------------------------------------------------------
function initMobileNav() {
  const toggleBtn = document.getElementById('mobileNavToggle');
  const sidebar = document.querySelector('.sidebar');
  if (!toggleBtn || !sidebar) return;

  toggleBtn.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });
}

function initFirebaseStatus() {
  fetch('/api/firebase/status')
    .then(res => res.json())
    .then(data => {
      const dot = document.getElementById('firebaseDot');
      const txt = document.getElementById('firebaseStatusText');
      if (data.success && data.status && data.status.connected) {
        if (dot) dot.style.background = '#10B981';
        if (txt) txt.textContent = `🔥 Cloud Firestore: Connected`;
      } else {
        if (dot) dot.style.background = '#F59E0B';
        if (txt) txt.textContent = `🔥 Firebase: Local Cache`;
      }
    })
    .catch(() => {});
}

function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.style.cssText = `
    padding: 12px 18px;
    background: #1E293B;
    color: #FFF;
    border-radius: 10px;
    font-size: 0.85rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.18);
    margin-bottom: 8px;
    animation: fadeIn 0.3s ease;
    border-left: 4px solid ${type === 'success' ? '#10B981' : (type === 'danger' ? '#EF4444' : '#3B82F6')};
  `;
  toast.textContent = message;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function initUiControls() {
  // 1. Profile Dropdown
  const profileBtn = document.getElementById('profileBtn');
  const profileMenu = document.getElementById('profileMenu');
  if (profileBtn && profileMenu) {
    profileBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      profileMenu.classList.toggle('open');
      const expanded = profileMenu.classList.contains('open');
      profileBtn.setAttribute('aria-expanded', expanded);
    });
    document.addEventListener('click', (e) => {
      if (!profileBtn.contains(e.target) && !profileMenu.contains(e.target)) {
        profileMenu.classList.remove('open');
        profileBtn.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // 2. Mobile Sidebar Toggle
  const sidebarBtn = document.querySelector('button[data-sidebar]');
  const sidebar = document.getElementById('sidebar');
  if (sidebarBtn && sidebar) {
    sidebarBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      sidebar.classList.toggle('open');
    });
    document.addEventListener('click', (e) => {
      if (!sidebar.contains(e.target) && !sidebarBtn.contains(e.target)) {
        sidebar.classList.remove('open');
      }
    });
  }

  // 3. Global Search Keyboard Shortcut (Cmd+K / Ctrl+K)
  const searchInput = document.getElementById('globalSearch');
  if (searchInput) {
    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        searchInput.focus();
      }
    });
  }

  // 4. Animate Dashboard mini bars
  document.querySelectorAll('.bars i[data-height]').forEach(bar => {
    const h = bar.getAttribute('data-height');
    if (h) {
      setTimeout(() => {
        bar.style.height = h + '%';
      }, 100);
    }
  });

  // 5. Animate Analytics page progress bars
  document.querySelectorAll('.analytics-bar[data-width]').forEach(bar => {
    const w = bar.getAttribute('data-width');
    if (w) {
      setTimeout(() => {
        bar.style.width = w + '%';
      }, 100);
    }
  });

  // 6. Theme Toggle (subtle ambient shift)
  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('alt-theme');
      themeToggle.textContent = document.body.classList.contains('alt-theme') ? '☾' : '☼';
    });
  }
}
