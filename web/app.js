const examples = [
  "我预约了胃镜检查，现在想取消重新约时间，应该在哪里操作？",
  "网上复诊开了药，但是药品邮寄一直显示加载失败怎么办？",
  "电子发票发不到邮箱，提示今天超过限制，能帮我处理吗？",
  "孩子发烧很严重，现在去急诊需要先挂号吗？",
  "我想投诉今天窗口工作人员态度不好，排队等了很久没人解释。",
];

const state = {
  status: null,
  history: JSON.parse(localStorage.getItem("classifier-history") || "[]"),
};

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  renderExamples();
  renderHistory();
  fetchStatus();
});

function bindEvents() {
  document.getElementById("classifyBtn").addEventListener("click", classifyCurrentText);
  document.getElementById("clearBtn").addEventListener("click", () => {
    document.getElementById("queryText").value = "";
    renderEmptyResult();
  });
  document.getElementById("clearHistoryBtn").addEventListener("click", () => {
    state.history = [];
    localStorage.removeItem("classifier-history");
    renderHistory();
  });
  document.getElementById("queryText").addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      classifyCurrentText();
    }
  });
}

async function fetchStatus() {
  try {
    const response = await fetch("/api/status");
    state.status = await response.json();
    renderStatus();
    renderLabelGrid();
  } catch (error) {
    renderConnectionError(error);
  }
}

function renderExamples() {
  const container = document.getElementById("examples");
  container.innerHTML = examples.map((item, index) => (
    `<button type="button" class="example-btn" data-example="${index}">样例${index + 1}</button>`
  )).join("");
  container.querySelectorAll("[data-example]").forEach((button) => {
    button.addEventListener("click", () => {
      document.getElementById("queryText").value = examples[Number(button.dataset.example)];
    });
  });
}

function renderStatus() {
  const dot = document.querySelector(".status-dot");
  const engineName = document.getElementById("engineName");
  const engineNote = document.getElementById("engineNote");
  const isBert = state.status.engine === "bert";
  dot.className = `status-dot ${isBert ? "is-ready" : "is-fallback"}`;
  engineName.textContent = isBert ? "BERT推理已启用" : "规则兜底模式";
  engineNote.textContent = state.status.message || "本地服务已连接";
}

function renderConnectionError(error) {
  const dot = document.querySelector(".status-dot");
  dot.className = "status-dot is-error";
  document.getElementById("engineName").textContent = "未连接服务";
  document.getElementById("engineNote").textContent = "请先运行 classification_platform/server.py";
  document.getElementById("resultPanel").innerHTML = `
    <div class="notice">
      无法连接后端服务：${escapeHTML(String(error.message || error))}
    </div>
  `;
}

function renderLabelGrid() {
  const container = document.getElementById("labelGrid");
  const labels = state.status?.labels || [];
  container.innerHTML = labels.map((item) => `
    <button class="label-chip" type="button" title="${escapeHTML(item.definition)}">
      <strong>${item.label}</strong>
      <span>${item.route}</span>
    </button>
  `).join("");
}

async function classifyCurrentText() {
  const text = document.getElementById("queryText").value.trim();
  const topK = Number(document.getElementById("topK").value || 5);
  if (!text) {
    document.getElementById("resultPanel").innerHTML = `<div class="notice">请输入需要分类的患者客服文本。</div>`;
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, top_k: topK }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "分类失败");
    renderResult(result);
    renderTopK(result.top_k || []);
    addHistory(result);
  } catch (error) {
    document.getElementById("resultPanel").innerHTML = `<div class="notice">${escapeHTML(String(error.message || error))}</div>`;
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  const button = document.getElementById("classifyBtn");
  button.disabled = isLoading;
  button.textContent = isLoading ? "分类中..." : "开始分类";
}

function renderEmptyResult() {
  document.getElementById("resultPanel").innerHTML = `
    <div class="empty-state">
      <h2>等待输入</h2>
      <p>分类后这里会显示一级标签、置信度、门诊二分类和建议流转方向。</p>
    </div>
  `;
  document.getElementById("topKList").innerHTML = "";
}

function renderResult(result) {
  const primary = result.multi_class;
  const binary = result.binary;
  const decision = result.decision;
  const keywords = result.matched_keywords || [];
  const riskClass = riskClassName(primary.risk);
  document.getElementById("resultPanel").innerHTML = `
    <div class="result-stack">
      <div class="primary-result">
        <div class="result-label">
          <strong>${primary.label}</strong>
          <span class="confidence">${formatPercent(primary.confidence)}</span>
        </div>
        <p>${primary.definition}</p>
      </div>
      <div class="result-meta">
        <div class="result-card">
          <span>门诊二分类</span>
          <strong>${binary.label}</strong>
          <span>置信度 ${formatPercent(binary.confidence)}</span>
        </div>
        <div class="result-card">
          <span>风险等级</span>
          <strong class="${riskClass}">${decision.risk_level}</strong>
          <span>${decision.recommended_action}</span>
        </div>
        <div class="result-card">
          <span>建议流转</span>
          <strong>${decision.route}</strong>
        </div>
        <div class="result-card">
          <span>推理引擎</span>
          <strong>${result.engine === "bert" ? "BERT" : "规则兜底"}</strong>
          <span>${result.engine_note}</span>
        </div>
      </div>
      <div class="result-card">
        <span>触发关键词</span>
        <strong>${keywords.length ? keywords.join("、") : "未命中显式关键词"}</strong>
      </div>
    </div>
  `;
}

function renderTopK(items) {
  const max = Math.max(...items.map((item) => item.confidence), 0.01);
  document.getElementById("topKList").innerHTML = items.map((item, index) => {
    const width = Math.max(2, item.confidence / max * 100);
    const color = ["#0f766e", "#4f46e5", "#d97706", "#e11d48", "#2563eb"][index % 5];
    return `
      <div class="bar-row">
        <span class="bar-name">${item.label}</span>
        <span class="bar-track"><span class="bar-fill" style="width:${width}%; background:${color}"></span></span>
        <span class="bar-score">${formatPercent(item.confidence)}</span>
      </div>
    `;
  }).join("");
}

function addHistory(result) {
  state.history.unshift({
    text: result.text,
    label: result.multi_class.label,
    score: result.multi_class.confidence,
    engine: result.engine,
  });
  state.history = state.history.slice(0, 12);
  localStorage.setItem("classifier-history", JSON.stringify(state.history));
  renderHistory();
}

function renderHistory() {
  const container = document.getElementById("historyList");
  if (!state.history.length) {
    container.innerHTML = `<p class="subtitle">暂无分类记录。</p>`;
    return;
  }
  container.innerHTML = state.history.map((item) => `
    <div class="history-item">
      <span class="history-text">${escapeHTML(item.text)}</span>
      <span class="history-label">${item.label}</span>
      <span class="history-score">${formatPercent(item.score)} · ${item.engine}</span>
    </div>
  `).join("");
}

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function riskClassName(risk) {
  if (risk === "high") return "risk-high";
  if (risk === "medium") return "risk-medium";
  if (risk === "low") return "risk-low";
  return "risk-review";
}

function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#039;",
  }[char]));
}
