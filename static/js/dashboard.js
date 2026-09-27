/**
 * dashboard.js
 * ------------
 * Core JavaScript logic for AI Camera Monitoring Dashboard.
 * Handles live KPI telemetry, animated counters, activity feed updates,
 * alerts toast system, and responsive chart visualizers.
 */

document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initAnimatedCounters();
  initRecentDetectionsPolling();
  initAlertsPolling();
  initCharts();
  initMobileNav();
  initFirebaseStatus();
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
// Animated Number Counters
// ---------------------------------------------------------
function initAnimatedCounters() {
  const counterElements = document.querySelectorAll('.animate-counter');

  counterElements.forEach(el => {
    const target = parseInt(el.getAttribute('data-target') || el.innerText.replace(/,/g, ''), 10);
    if (isNaN(target)) return;

    let current = 0;
    const duration = 1200; // ms
    const stepTime = 20;
    const totalSteps = duration / stepTime;
    const increment = target / totalSteps;

    const timer = setInterval(() => {
      current += increment;
      if (current >= target) {
        current = target;
        clearInterval(timer);
      }
      el.textContent = Math.round(current).toLocaleString();
    }, stepTime);
  });
}

// ---------------------------------------------------------
// Real-time KPI & Detections Polling
// ---------------------------------------------------------
function initRecentDetectionsPolling() {
  const activityList = document.getElementById('recentDetectionsList');

  async function fetchLatest() {
    try {
      const res = await fetch('/api/detections?limit=6');
      if (!res.ok) return;
      const data = await res.json();

      if (data.success && data.detections && activityList) {
        renderRecentDetections(data.detections, activityList);
      }

      // Also refresh KPI counters
      const statsRes = await fetch('/api/stats');
      if (statsRes.ok) {
        const statsData = await statsRes.json();
        if (statsData.success && statsData.stats) {
          updateKpis(statsData.stats);
        }
      }
    } catch (e) {
      console.warn('Telemetry polling error:', e);
    }
  }

  // Poll every 4 seconds
  setInterval(fetchLatest, 4000);
}

function updateKpis(stats) {
  const totalEl = document.getElementById('kpiTotalDetections');
  const knownEl = document.getElementById('kpiKnownPersons');
  const unknownEl = document.getElementById('kpiUnknownPersons');
  const cameraEl = document.getElementById('kpiActiveCameras');

  if (totalEl) totalEl.textContent = Number(stats.total_detections).toLocaleString();
  if (knownEl) knownEl.textContent = Number(stats.known_persons).toLocaleString();
  if (unknownEl) unknownEl.textContent = Number(stats.unknown_persons).toLocaleString();
  if (cameraEl && stats.active_cameras) cameraEl.textContent = stats.active_cameras;
}

function renderRecentDetections(detections, container) {
  if (detections.length === 0) {
    container.innerHTML = '<div style="padding: 24px; text-align: center; color: #94A3B8;">No recent detections recorded</div>';
    return;
  }

  const html = detections.map(det => {
    const isKnown = det.recognition_status === 'known';
    const badgeClass = isKnown ? 'badge-known' : 'badge-unknown';
    const badgeText = isKnown ? 'KNOWN' : 'UNKNOWN';
    const confPercent = Math.round(det.confidence * 100);

    return `
      <div class="activity-item animate-fade-in">
        <img class="activity-thumb" src="${det.image_url || '/static/images/default_avatar.svg'}"
             alt="${det.person_name}" onerror="this.src='/static/images/default_avatar.svg'">
        <div class="activity-info">
          <div class="activity-title-row">
            <span class="activity-name">${escapeHtml(det.person_name)}</span>
            <span class="badge ${badgeClass}">${badgeText}</span>
          </div>
          <div class="activity-meta">
            <span><i class="camera-icon"></i> ${escapeHtml(det.camera_id)}</span>
            <span>•</span>
            <span>${confPercent}% match</span>
            <span>•</span>
            <span>${escapeHtml(det.timestamp)}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');

  container.innerHTML = html;
}

// ---------------------------------------------------------
// Unknown Person Alert Notifications
// ---------------------------------------------------------
let notifiedAlertIds = new Set();

function initAlertsPolling() {
  async function checkAlerts() {
    try {
      const res = await fetch('/api/alerts');
      if (!res.ok) return;
      const data = await res.json();

      if (data.success && data.alerts && data.alerts.length > 0) {
        const latest = data.alerts[0];
        if (!notifiedAlertIds.has(latest.id)) {
          notifiedAlertIds.add(latest.id);
          showToast(`⚠️ ${latest.title}`, 'danger');
        }
      }
    } catch (e) {
      console.warn('Alert polling error:', e);
    }
  }

  setInterval(checkAlerts, 5000);
}

// ---------------------------------------------------------
// Toast System
// ---------------------------------------------------------
function showToast(message, type = 'primary') {
  let container = document.getElementById('toastContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toastContainer';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <div style="font-weight: 600; font-size: 0.9rem; color: #1E293B;">${escapeHtml(message)}</div>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(-10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ---------------------------------------------------------
// Lightweight Responsive Charts (Canvas-based)
// ---------------------------------------------------------
function initCharts() {
  renderActivityChart();
  renderDistributionChart();
}

function renderActivityChart() {
  const canvas = document.getElementById('activityTrendChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const labels = ['08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00'];
  const values = [45, 120, 210, 185, 290, 230, 95];
  const maxVal = Math.max(...values) * 1.2;

  const w = rect.width;
  const h = rect.height;
  const padBottom = 35;
  const padLeft = 40;
  const padTop = 20;
  const padRight = 20;

  ctx.clearRect(0, 0, w, h);

  // Draw grid lines
  ctx.strokeStyle = '#E2E8F0';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padTop + (h - padTop - padBottom) * (i / 4);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(w - padRight, y);
    ctx.stroke();
  }

  // Draw Gradient Area & Line
  const points = values.map((val, idx) => {
    const x = padLeft + (w - padLeft - padRight) * (idx / (values.length - 1));
    const y = h - padBottom - ((val / maxVal) * (h - padTop - padBottom));
    return { x, y };
  });

  // Area
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

  // Line
  ctx.beginPath();
  ctx.strokeStyle = '#4F46E5';
  ctx.lineWidth = 3;
  points.forEach((p, idx) => {
    if (idx === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();

  // Draw points & Labels
  points.forEach((p, idx) => {
    ctx.fillStyle = '#FFFFFF';
    ctx.beginPath();
    ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#4F46E5';
    ctx.lineWidth = 2.5;
    ctx.stroke();

    // Label
    ctx.fillStyle = '#64748B';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(labels[idx], p.x, h - 12);
  });
}

function renderDistributionChart() {
  const canvas = document.getElementById('distributionDonutChart');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const radius = Math.min(cx, cy) - 25;
  const innerRadius = radius * 0.65;

  const data = [
    { label: 'Known Persons', value: 72, color: '#22C55E' },
    { label: 'Unknown Persons', value: 28, color: '#EF4444' }
  ];

  let startAngle = -Math.PI / 2;

  data.forEach(slice => {
    const sliceAngle = (slice.value / 100) * (Math.PI * 2);
    ctx.beginPath();
    ctx.arc(cx, cy, radius, startAngle, startAngle + sliceAngle);
    ctx.arc(cx, cy, innerRadius, startAngle + sliceAngle, startAngle, true);
    ctx.closePath();
    ctx.fillStyle = slice.color;
    ctx.fill();
    startAngle += sliceAngle;
  });

  // Center text
  ctx.fillStyle = '#1E293B';
  ctx.font = 'bold 20px sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('72%', cx, cy - 8);

  ctx.fillStyle = '#64748B';
  ctx.font = '11px sans-serif';
  ctx.fillText('Known Match', cx, cy + 14);
}

// ---------------------------------------------------------
// Mobile Navigation Drawer
// ---------------------------------------------------------
function initMobileNav() {
  const toggleBtn = document.getElementById('mobileNavToggle');
  const sidebar = document.querySelector('.sidebar');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
    });
  }
}

function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------
// Firebase Firestore Status & Configuration
// ---------------------------------------------------------
function initFirebaseStatus() {
  fetchFirebaseStatus();
  setInterval(fetchFirebaseStatus, 12000);
}

async function fetchFirebaseStatus() {
  try {
    const res = await fetch('/api/firebase/status');
    if (!res.ok) return;
    const data = await res.json();
    if (data.success && data.status) {
      updateFirebaseUI(data.status);
    }
  } catch (e) {
    console.warn('Firebase status fetch failed:', e);
  }
}

function updateFirebaseUI(status) {
  const badge = document.getElementById('firebaseStatusText');
  const dot = document.getElementById('firebaseDot');
  const modalMode = document.getElementById('fbModalMode');
  const modalProject = document.getElementById('fbModalProject');
  const modalStorage = document.getElementById('fbModalStorage');

  if (status.connected) {
    if (badge) badge.textContent = `🔥 Firestore: ${status.project_id}`;
    if (dot) dot.style.background = '#22C55E';
    if (modalMode) {
      modalMode.innerHTML = `<span class="badge badge-known">CONNECTED (LIVE CLOUD)</span>`;
    }
  } else {
    if (badge) badge.textContent = '🔥 Firebase: Demo Mode';
    if (dot) dot.style.background = '#F59E0B';
    if (modalMode) {
      modalMode.innerHTML = `<span class="badge badge-unknown">DEMO MODE (Local)</span>`;
    }
  }

  if (modalProject) modalProject.textContent = status.project_id || 'Not configured';
  if (modalStorage) modalStorage.textContent = status.image_storage || "Cloud Firestore ('images' collection)";
}

window.openFirebaseModal = function() {
  const modal = document.getElementById('firebaseModal');
  if (modal) {
    modal.classList.add('show');
    fetchFirebaseStatus();
  }
};

window.closeFirebaseModal = function() {
  const modal = document.getElementById('firebaseModal');
  if (modal) modal.classList.remove('show');
};

window.testFirebaseConnection = async function() {
  const testBtn = document.getElementById('btnTestFirestore');
  const statusMsg = document.getElementById('fbTestResult');
  if (testBtn) testBtn.disabled = true;
  if (statusMsg) statusMsg.innerHTML = '<span style="color:#64748B;">Testing connection to Cloud Firestore...</span>';

  try {
    const res = await fetch('/api/firebase/test', { method: 'POST' });
    const data = await res.json();
    if (data.success) {
      if (statusMsg) statusMsg.innerHTML = `<span style="color:#16A34A; font-weight:600;">✓ ${data.message}</span>`;
      fetchFirebaseStatus();
    } else {
      if (statusMsg) statusMsg.innerHTML = `<span style="color:#DC2626; font-weight:600;">✗ ${data.error || 'Connection failed'}</span>`;
    }
  } catch (err) {
    if (statusMsg) statusMsg.innerHTML = `<span style="color:#DC2626;">Network error testing Firestore: ${err}</span>`;
  } finally {
    if (testBtn) testBtn.disabled = false;
  }
};

window.saveFirebaseConfig = async function(e) {
  if (e) e.preventDefault();
  const jsonInput = document.getElementById('fbJsonInput');
  const jsonText = jsonInput ? jsonInput.value.trim() : '';
  const fileInput = document.getElementById('fbFileInput');
  const statusMsg = document.getElementById('fbTestResult');

  const formData = new FormData();
  if (fileInput && fileInput.files && fileInput.files.length > 0) {
    formData.append('file', fileInput.files[0]);
  } else if (jsonText) {
    formData.append('credentials_json', jsonText);
  } else {
    alert('Please paste service account JSON or select a file');
    return;
  }

  if (statusMsg) statusMsg.innerHTML = '<span style="color:#64748B;">Connecting & verifying credentials...</span>';

  try {
    const res = await fetch('/api/firebase/configure', {
      method: 'POST',
      body: formData
    });
    const data = await res.json();
    if (data.success) {
      if (statusMsg) statusMsg.innerHTML = `<span style="color:#16A34A; font-weight:600;">✓ Connected to Cloud Firestore successfully!</span>`;
      fetchFirebaseStatus();
      setTimeout(() => {
        closeFirebaseModal();
        location.reload();
      }, 1500);
    } else {
      if (statusMsg) statusMsg.innerHTML = `<span style="color:#DC2626; font-weight:600;">✗ ${data.error || 'Configuration failed'}</span>`;
    }
  } catch (err) {
    if (statusMsg) statusMsg.innerHTML = `<span style="color:#DC2626;">Error: ${err}</span>`;
  }
};
