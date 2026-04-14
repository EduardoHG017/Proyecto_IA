/* ============================================================
   script.js — Mesa de Ayuda · Clasificador de Tickets
   Universidad Rafael Landívar · IA 2026
   ============================================================ */

'use strict';

// ---- Estado de la sesión ----
const history = [];

// ---- Generar Ticket ID al cargar ----
document.addEventListener('DOMContentLoaded', () => {
  refreshTicketId();
  updateCharCount();
  document.getElementById('description').addEventListener('input', updateCharCount);
});

function refreshTicketId() {
  const id = 'TKT-' + Date.now().toString(36).toUpperCase().slice(-6);
  document.getElementById('ticketId').textContent = id;
  return id;
}

function updateCharCount() {
  const desc = document.getElementById('description');
  document.getElementById('charCount').textContent =
    `${desc.value.length} / 2000`;
}

// ============================================================
// CLASIFICAR TICKET
// ============================================================
async function classifyTicket() {
  const subject     = document.getElementById('subject').value.trim();
  const description = document.getElementById('description').value.trim();
  const ticketId    = document.getElementById('ticketId').textContent;

  hideError();

  if (!subject && !description) {
    showError('Por favor, ingresa al menos el asunto del ticket.');
    return;
  }

  setLoading(true);

  try {
    const response = await fetch('/classify', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ subject, description }),
    });

    const data = await response.json();

    if (!response.ok) {
      showError(data.error || 'Error al clasificar el ticket.');
      return;
    }

    renderResult(data, subject, description, ticketId);
    addToHistory(ticketId, subject || description.slice(0, 50), data.prediction, data.meta);
    refreshTicketId();

  } catch (err) {
    showError('No se pudo conectar con el servidor. Verifica que Flask esté corriendo.');
  } finally {
    setLoading(false);
  }
}

// ============================================================
// RENDERIZAR RESULTADO
// ============================================================
function renderResult(data, subject, description, ticketId) {
  const { prediction, probabilities, meta, token_count } = data;

  // Header
  const header = document.getElementById('resultHeader');
  header.style.background = meta.bg  || '#f0f9ff';
  header.style.color      = meta.color || '#1e293b';

  document.getElementById('resultIcon').className =
    `fa-solid ${meta.icon || 'fa-tag'} fa-2x mb-2`;
  document.getElementById('resultCategory').textContent = prediction;
  document.getElementById('resultTicketInfo').textContent =
    `${ticketId}  ·  ${new Date().toLocaleTimeString('es-GT')}`;

  // Barras de probabilidad
  const container = document.getElementById('probBars');
  container.innerHTML = '';

  const sorted = Object.entries(probabilities).sort((a, b) => b[1].probability - a[1].probability);

  sorted.forEach(([cls, info]) => {
    const isTop = cls === prediction;
    const div = document.createElement('div');
    div.className = 'prob-item';
    div.innerHTML = `
      <div class="prob-label">
        <span class="name">
          <i class="fa-solid ${info.icon}" style="color:${info.color}; width:16px"></i>
          ${cls}
          ${isTop ? '<span class="badge bg-primary ms-1" style="font-size:.65rem">predicho</span>' : ''}
        </span>
        <span class="pct" style="color:${info.color}">${info.probability.toFixed(1)}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" data-width="${info.probability}"
             style="background:${info.color}; width:0%"></div>
      </div>`;
    container.appendChild(div);
  });

  // Token info
  document.getElementById('tokenInfo').textContent =
    `${token_count} tokens procesados`;

  // Mostrar card
  document.getElementById('placeholder').classList.add('d-none');
  document.getElementById('loadingCard').classList.add('d-none');
  const card = document.getElementById('resultCard');
  card.classList.remove('d-none');
  card.classList.add('fade-in');

  // Animar barras con pequeño delay
  requestAnimationFrame(() => {
    setTimeout(() => {
      document.querySelectorAll('.prob-bar-fill').forEach(el => {
        el.style.width = el.dataset.width + '%';
      });
    }, 80);
  });
}

// ============================================================
// HISTORIAL
// ============================================================
function addToHistory(ticketId, subject, category, meta) {
  history.unshift({ ticketId, subject, category, meta, time: new Date() });

  const list  = document.getElementById('historyList');
  const empty = document.getElementById('historyEmpty');

  empty.classList.add('d-none');
  list.classList.remove('d-none');

  const item = document.createElement('div');
  item.className = 'history-item list-group-item fade-in';
  item.innerHTML = `
    <i class="fa-solid ${meta.icon || 'fa-tag'}" style="color:${meta.color}; width:18px"></i>
    <span class="history-subject" title="${subject}">${subject || '(sin asunto)'}</span>
    <span class="history-badge" style="background:${meta.bg}; color:${meta.color}">
      ${category}
    </span>
    <span class="history-time">${new Date().toLocaleTimeString('es-GT')}</span>`;

  list.insertBefore(item, list.firstChild);
}

function clearHistory() {
  history.length = 0;
  document.getElementById('historyList').innerHTML = '';
  document.getElementById('historyList').classList.add('d-none');
  document.getElementById('historyEmpty').classList.remove('d-none');
}

// ============================================================
// UTILIDADES
// ============================================================
function clearForm() {
  document.getElementById('subject').value     = '';
  document.getElementById('description').value = '';
  updateCharCount();
  hideError();
  document.getElementById('resultCard').classList.add('d-none');
  document.getElementById('placeholder').classList.remove('d-none');
  refreshTicketId();
}

function setLoading(on) {
  const btn     = document.getElementById('btnClassify');
  const loading = document.getElementById('loadingCard');
  const result  = document.getElementById('resultCard');
  const placeholder = document.getElementById('placeholder');

  btn.disabled = on;
  btn.innerHTML = on
    ? '<span class="spinner-border spinner-border-sm me-2"></span>Clasificando…'
    : '<i class="fa-solid fa-magnifying-glass-chart me-2"></i>Clasificar Ticket';

  if (on) {
    placeholder.classList.add('d-none');
    result.classList.add('d-none');
    loading.classList.remove('d-none');
    loading.classList.add('d-flex');
  } else {
    loading.classList.add('d-none');
    loading.classList.remove('d-flex');
  }
}

function showError(msg) {
  document.getElementById('errorMsg').textContent = msg;
  document.getElementById('alertError').classList.remove('d-none');
}

function hideError() {
  document.getElementById('alertError').classList.add('d-none');
}

// Permitir clasificar con Enter desde el campo de asunto
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('subject').addEventListener('keydown', e => {
    if (e.key === 'Enter') classifyTicket();
  });
});
