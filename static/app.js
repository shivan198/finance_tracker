const form = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const dropzone = document.getElementById('dropzone');
const fileChosen = document.getElementById('file-chosen');
const analyzeBtn = document.getElementById('analyze-btn');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loading-text');
const errorBox = document.getElementById('error-box');
const uploadView = document.getElementById('upload-view');
const resultsView = document.getElementById('results-view');
const newUploadBtn = document.getElementById('new-upload-btn');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatLog = document.getElementById('chat-log');

let chartInstances = {};
let currentCurrencySymbol = '$';
const REFUSAL_TEXT = "I can only answer questions about this uploaded statement.";

// ---------------- Upload interactions ----------------

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    fileChosen.textContent = fileInput.files[0].name;
    analyzeBtn.disabled = false;
  }
});

['dragover', 'dragenter'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
});

['dragleave', 'drop'].forEach(evt => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
  });
});

dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files.length > 0) {
    fileInput.files = files;
    fileChosen.textContent = files[0].name;
    analyzeBtn.disabled = false;
  }
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!fileInput.files.length) return;

  setLoading(true);
  hideError();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const res = await fetch('/api/analyze', { method: 'POST', body: formData });
    const data = await res.json();

    if (!data.ok) {
      showError(data.error || 'Something went wrong while analyzing this statement.');
      setLoading(false);
      return;
    }

    renderResults(data);
    uploadView.classList.add('hidden');
    resultsView.classList.remove('hidden');
  } catch (err) {
    showError('Could not reach the server. Is it running?');
  } finally {
    setLoading(false);
  }
});

newUploadBtn.addEventListener('click', () => {
  resultsView.classList.add('hidden');
  uploadView.classList.remove('hidden');
  fileInput.value = '';
  fileChosen.textContent = '';
  analyzeBtn.disabled = true;
  chatLog.innerHTML = '';
  hideError();
});

function setLoading(isLoading) {
  loading.classList.toggle('hidden', !isLoading);
  analyzeBtn.disabled = isLoading || !fileInput.files.length;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove('hidden');
}

function hideError() {
  errorBox.classList.add('hidden');
  errorBox.textContent = '';
}

// ---------------- Rendering ----------------

const currency = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
};

const money = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return currentCurrencySymbol + currency(v);
};

function renderResults(data) {
  currentCurrencySymbol = (data.currency && data.currency.symbol) || '$';

  document.getElementById('result-filename').textContent = data.filename;

  const s = data.summary;
  const period = (s.period_start && s.period_end) ? `${s.period_start} to ${s.period_end}` : '';
  document.getElementById('result-period').textContent = period;

  document.getElementById('stat-income').textContent = money(s.total_income);
  document.getElementById('stat-expense').textContent = money(s.total_expense);
  const net = s.net_change;
  const netEl = document.getElementById('stat-net');
  netEl.textContent = (net >= 0 ? '+' : '') + money(net);
  netEl.className = 'stat-value ' + (net >= 0 ? 'income' : 'expense');
  document.getElementById('stat-count').textContent = s.total_transactions;

  const flagBanner = document.getElementById('flag-banner');
  if (data.flagged_count > 0) {
    flagBanner.textContent = `${data.flagged_count} transaction(s) didn't pass balance validation — worth a manual check.`;
    flagBanner.classList.remove('hidden');
  } else {
    flagBanner.classList.add('hidden');
  }

  const insightsCard = document.getElementById('insights-card');
  const insightsError = document.getElementById('insights-error');
  if (data.insights) {
    document.getElementById('insights-text').textContent = data.insights;
    insightsCard.classList.remove('hidden');
    insightsError.classList.add('hidden');
  } else {
    insightsCard.classList.add('hidden');
    insightsError.textContent = data.insights_error || '';
    insightsError.classList.toggle('hidden', !data.insights_error);
  }

  renderCharts(data.chart_data);
  renderTransactions(data.transactions);

  const categoryCard = document.getElementById('category-chart-card');
  categoryCard.classList.toggle('hidden', !(data.chart_data.by_category && data.chart_data.by_category.length));
}

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

function renderCharts(chartData) {
  const inkSoft = '#4a5468';
  const income = '#1f6f5c';
  const expense = '#a6432c';
  const gold = '#c9a227';

  Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
  Chart.defaults.color = inkSoft;

  destroyChart('monthly');
  destroyChart('balance');
  destroyChart('topExpenses');
  destroyChart('dailyNet');
  destroyChart('category');

  chartInstances.monthly = new Chart(document.getElementById('chart-monthly'), {
    type: 'bar',
    data: {
      labels: chartData.monthly.map(m => m.month),
      datasets: [
        { label: 'Income', data: chartData.monthly.map(m => m.income), backgroundColor: income },
        { label: 'Expenses', data: chartData.monthly.map(m => m.expense), backgroundColor: expense },
      ],
    },
    options: { responsive: true, plugins: { legend: { position: 'bottom' } } },
  });

  chartInstances.balance = new Chart(document.getElementById('chart-balance'), {
    type: 'line',
    data: {
      labels: chartData.balance.map(b => b.date),
      datasets: [{
        label: 'Balance',
        data: chartData.balance.map(b => b.balance),
        borderColor: gold,
        backgroundColor: 'rgba(201,162,39,0.12)',
        fill: true,
        tension: 0.25,
        pointRadius: 0,
      }],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });

  chartInstances.topExpenses = new Chart(document.getElementById('chart-top-expenses'), {
    type: 'bar',
    data: {
      labels: chartData.top_expenses.map(t => t.description),
      datasets: [{ label: 'Amount', data: chartData.top_expenses.map(t => t.amount), backgroundColor: expense }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });

  chartInstances.dailyNet = new Chart(document.getElementById('chart-daily-net'), {
    type: 'bar',
    data: {
      labels: chartData.daily_net.map(d => d.date),
      datasets: [{
        label: 'Net',
        data: chartData.daily_net.map(d => d.net),
        backgroundColor: chartData.daily_net.map(d => d.net >= 0 ? income : expense),
      }],
    },
    options: { responsive: true, plugins: { legend: { display: false } } },
  });

  if (chartData.by_category && chartData.by_category.length) {
    const palette = ['#a6432c', '#c9a227', '#1f6f5c', '#4a5468', '#7a5c3e', '#8a6d3b', '#5c7a6f', '#9c5b48', '#3f5d54', '#b08a3e'];
    chartInstances.category = new Chart(document.getElementById('chart-category'), {
      type: 'doughnut',
      data: {
        labels: chartData.by_category.map(c => c.category),
        datasets: [{
          data: chartData.by_category.map(c => c.total),
          backgroundColor: chartData.by_category.map((_, i) => palette[i % palette.length]),
        }],
      },
      options: { responsive: true, plugins: { legend: { position: 'right' } } },
    });
  }
}

function renderTransactions(transactions) {
  const tbody = document.getElementById('transactions-body');
  tbody.innerHTML = '';

  for (const tx of transactions) {
    const tr = document.createElement('tr');
    if (tx.flagged) tr.classList.add('flagged');

    tr.innerHTML = `
      <td>${tx.date ?? '—'}</td>
      <td>${escapeHtml(tx.description)}${tx.flagged ? '<span class="flag-tag">needs review</span>' : ''}</td>
      <td class="num">${tx.withdrawal !== null ? money(tx.withdrawal) : ''}</td>
      <td class="num">${tx.deposit !== null ? money(tx.deposit) : ''}</td>
      <td class="num">${tx.balance !== null ? money(tx.balance) : ''}</td>
    `;
    tbody.appendChild(tr);
  }
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ---------------- Chat ----------------

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  addChatBubble(question, 'user');
  chatInput.value = '';
  setChatBusy(true);
  const thinkingEl = addThinkingBubble();

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();

    thinkingEl.remove();

    if (!data.ok) {
      addChatBubble(data.error || 'Something went wrong.', 'error');
    } else if (data.answer === REFUSAL_TEXT) {
      addChatBubble(data.answer, 'assistant declined');
    } else {
      addChatBubble(data.answer, 'assistant', data.sources);
    }
  } catch (err) {
    thinkingEl.remove();
    addChatBubble('Could not reach the server.', 'error');
  } finally {
    setChatBusy(false);
  }
});

function addChatBubble(text, kind, sources) {
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${kind}`;
  bubble.textContent = text;

  if (sources && sources.length) {
    const sourceLine = document.createElement('div');
    sourceLine.className = 'chat-sources';
    sourceLine.textContent = `Based on: ${sources.join(', ')}`;
    bubble.appendChild(sourceLine);
  }

  chatLog.appendChild(bubble);
  bubble.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return bubble;
}

function addThinkingBubble() {
  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble thinking';
  bubble.innerHTML = '<span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>';
  chatLog.appendChild(bubble);
  bubble.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return bubble;
}

function setChatBusy(isBusy) {
  chatInput.disabled = isBusy;
  chatForm.querySelector('button').disabled = isBusy;
}
