const lastResults = { diabetes: null, heart: null };

const levelLabels = { low: "Low Risk", moderate: "Moderate Risk", high: "High Risk" };

function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  const sections = document.querySelectorAll(".section");

  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.target;
      navButtons.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      sections.forEach((sec) => sec.classList.toggle("is-active", sec.id === target));

      if (target === "section-combined") renderCombined();
    });
  });
}

function animateGauge(path, percent) {
  const length = path.getTotalLength();
  path.style.strokeDasharray = `${length}`;
  path.style.strokeDashoffset = `${length}`;
  // double rAF so the browser paints the empty state first, otherwise no transition
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      path.style.strokeDashoffset = `${length * (1 - percent / 100)}`;
    });
  });
}

function riskMessage(level) {
  if (level === "low") return "Indicators look mostly within a healthy range. Keep up regular checkups.";
  if (level === "moderate") return "Some indicators are borderline. Consider consulting a doctor for a proper checkup.";
  return "Several indicators suggest elevated risk. Please consult a medical professional soon.";
}

function renderGauge(card, result) {
  card.innerHTML = `
    <div class="result-content">
      <div class="gauge-wrap">
        <svg viewBox="0 0 200 120" class="gauge">
          <path class="gauge-track" d="M10,110 A90,90 0 0 1 190,110"></path>
          <path class="gauge-fill ${result.risk_level}" d="M10,110 A90,90 0 0 1 190,110"></path>
        </svg>
        <div class="gauge-center">
          <span class="gauge-number">${result.probability.toFixed(1)}%</span>
          <span class="gauge-caption">estimated risk</span>
        </div>
      </div>
      <p class="pill ${result.risk_level}">${result.risk_label}</p>
      <p class="result-message">${riskMessage(result.risk_level)}</p>
      <p class="fine-print">This is a machine learning estimate for an academic project, not a medical diagnosis.</p>
    </div>
  `;
  animateGauge(card.querySelector(".gauge-fill"), result.probability);
}

function setupPredictForm(key, { numeric, buttonText }) {
  const form = document.getElementById(`${key}-form`);
  const card = document.getElementById(`${key}-result-card`);

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const btn = form.querySelector("button[type=submit]");
    btn.disabled = true;
    btn.textContent = "Predicting...";

    const payload = {};
    new FormData(form).forEach((value, name) => {
      payload[name] = numeric ? Number(value) : value;
    });

    try {
      const res = await fetch(`/api/predict/${key}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!data.success) throw new Error(data.error || "Prediction failed");

      lastResults[key] = data;
      renderGauge(card, data);
    } catch (err) {
      card.innerHTML = `<p class="result-placeholder">Something went wrong: ${err.message}</p>`;
    } finally {
      btn.disabled = false;
      btn.textContent = buttonText;
    }
  });
}

function setPill(el, label, level) {
  el.textContent = label;
  el.className = `pill ${level}`;
}

function renderCombined() {
  const placeholder = document.getElementById("combined-placeholder");
  const grid = document.getElementById("combined-grid");
  const overall = document.getElementById("combined-overall");
  const { diabetes, heart } = lastResults;

  const ready = diabetes && heart;
  placeholder.hidden = ready;
  grid.hidden = !ready;
  overall.hidden = !ready;
  if (!ready) return;

  document.getElementById("combined-diabetes-value").textContent = `${diabetes.probability.toFixed(1)}%`;
  setPill(document.getElementById("combined-diabetes-pill"), diabetes.risk_label, diabetes.risk_level);

  document.getElementById("combined-heart-value").textContent = `${heart.probability.toFixed(1)}%`;
  setPill(document.getElementById("combined-heart-pill"), heart.risk_label, heart.risk_level);

  const score = (diabetes.probability + heart.probability) / 2;
  let level = "low";
  if (score >= 60) level = "high";
  else if (score >= 30) level = "moderate";

  const path = document.querySelector("#combined-gauge .gauge-fill");
  path.classList.remove("low", "moderate", "high");
  path.classList.add(level);
  animateGauge(path, score);

  document.getElementById("combined-gauge-number").textContent = `${score.toFixed(1)}%`;
  setPill(document.getElementById("combined-overall-pill"), levelLabels[level], level);

  const messages = {
    high: "Both risk areas need attention. A full medical checkup is strongly recommended.",
    moderate: "Some risk indicators present. Consider lifestyle changes (diet, exercise) and a routine checkup.",
    low: "Overall indicators look healthy. Keep maintaining a good lifestyle!",
  };
  document.getElementById("combined-message-text").textContent = messages[level];
}

function initInsightsTabs() {
  const buttons = document.querySelectorAll(".insights-tab-btn");
  const panels = document.querySelectorAll(".insights-panel");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("is-active"));
      btn.classList.add("is-active");
      panels.forEach((p) => p.classList.toggle("is-active", p.id === `insights-${btn.dataset.insights}`));
    });
  });
}

function buildMetricsTable(table, results, bestModel) {
  const cols = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "CV Accuracy (mean)"];
  const head = cols.map((c) => `<th>${c}</th>`).join("");

  const body = results.map((row) => {
    const cells = cols
      .map((c) => `<td>${c === "Model" ? row[c] : Number(row[c]).toFixed(3)}</td>`)
      .join("");
    return `<tr class="${row.Model === bestModel ? "is-best" : ""}">${cells}</tr>`;
  }).join("");

  table.innerHTML = `<thead><tr>${head}</tr></thead><tbody>${body}</tbody>`;
}

async function loadModelInsights() {
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    buildMetricsTable(document.getElementById("table-diabetes"), data.diabetes.all_results, data.diabetes.best_model_name);
    buildMetricsTable(document.getElementById("table-heart"), data.heart.all_results, data.heart.best_model_name);
  } catch (err) {
    console.error("couldn't load model info", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  setupPredictForm("diabetes", { numeric: true, buttonText: "Predict Diabetes Risk" });
  setupPredictForm("heart", { numeric: false, buttonText: "Predict Heart Disease Risk" });
  initInsightsTabs();
  loadModelInsights();
});
