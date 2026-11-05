const SAMPLE_MESSAGES = {
  ham_casual: "Hey! Are we still meeting at the library around 4pm today to finish our AI lab report? Let me know if you need me to bring my laptop.",
  ham_delivery: "Hi Mum, I reached college safely. My project presentation is scheduled for 2:30 PM. Will call you once it finishes.",
  spam_prize: "CONGRATULATIONS! You have won a guaranteed £1000 cash prize in our weekly draw! Call 09061701461 immediately or visit www.prize-claim-now.com to claim your reward.",
  spam_phish: "URGENT: Your bank account has been suspended due to incomplete KYC verification. Click http://bit.ly/verify-kyc-alert within 24 hours and enter your OTP and PIN to restore access."
};

let dashboardState = null;
let chartInstances = {};

function switchTab(tabId) {
  document.querySelectorAll(".nav-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${tabId}`);
  });
}

document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => switchTab(btn.dataset.tab));
});

function destroyChart(key) {
  if (chartInstances[key]) {
    chartInstances[key].destroy();
    delete chartInstances[key];
  }
}

async function initDashboard() {
  const alertEl = document.getElementById("global-alert");
  try {
    const response = await fetch("/api/dashboard");
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Failed to load dashboard data.");
    }
    dashboardState = data;
    populateOverview(data.eda, data.metrics);
    populateModelSelectors(data.metrics);
    renderAnalyticsCharts(data.eda);
    renderComparisonSection(data.metrics);
  } catch (err) {
    alertEl.textContent = `Backend Alert: ${err.message}`;
    alertEl.classList.remove("hidden");
  }
}

function populateOverview(eda, metrics) {
  const winner = metrics.models[metrics.default_model_id];
  document.getElementById("kpi-total-rows").textContent = eda.cleaned_rows.toLocaleString();
  document.getElementById("kpi-duplicates-removed").textContent = `${eda.duplicate_rows_removed} exact duplicates removed`;
  document.getElementById("kpi-ham-count").textContent = eda.ham_count.toLocaleString();
  document.getElementById("kpi-ham-pct").textContent = `${eda.ham_percentage}% of cleaned dataset`;
  document.getElementById("kpi-spam-count").textContent = eda.spam_count.toLocaleString();
  document.getElementById("kpi-spam-pct").textContent = `${eda.spam_percentage}% of cleaned dataset`;
  document.getElementById("kpi-default-model").textContent = metrics.default_model_name;
  document.getElementById("kpi-test-f1").textContent = winner.test.f1_score.toFixed(4);
  document.getElementById("kpi-test-prec-rec").textContent = `Precision: ${(winner.test.precision * 100).toFixed(1)}% | Recall: ${(winner.test.recall * 100).toFixed(1)}%`;
  document.getElementById("kpi-test-roc").textContent = winner.test.roc_auc.toFixed(4);
  document.getElementById("kpi-test-acc").textContent = `Test Accuracy: ${(winner.test.accuracy * 100).toFixed(2)}%`;
}

function populateModelSelectors(metrics) {
  const detectorSelect = document.getElementById("model-select");
  const inspectorSelect = document.getElementById("inspector-model-select");
  detectorSelect.innerHTML = "";
  inspectorSelect.innerHTML = "";

  Object.values(metrics.models).forEach((m) => {
    const isDefault = m.id === metrics.default_model_id;
    const label = `${m.name} (${m.type})${isDefault ? " ★ Default" : ""}`;
    detectorSelect.add(new Option(label, m.id, isDefault, isDefault));
    inspectorSelect.add(new Option(label, m.id, isDefault, isDefault));
  });

  inspectorSelect.addEventListener("change", (e) => updateModelInspector(e.target.value));
}

function renderAnalyticsCharts(eda) {
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.borderColor = "#1f2937";

  destroyChart("classDist");
  chartInstances["classDist"] = new Chart(document.getElementById("chart-class-dist"), {
    type: "doughnut",
    data: {
      labels: [`Ham (${eda.ham_percentage}%)`, `Spam (${eda.spam_percentage}%)`],
      datasets: [{ data: [eda.ham_count, eda.spam_count], backgroundColor: ["#14b8a6", "#f43f5e"], borderWidth: 0 }]
    }
  });

  destroyChart("charHist");
  chartInstances["charHist"] = new Chart(document.getElementById("chart-char-hist"), {
    type: "bar",
    data: {
      labels: eda.char_length_histogram.labels,
      datasets: [
        { label: "Ham", data: eda.char_length_histogram.ham, backgroundColor: "rgba(20, 184, 166, 0.7)" },
        { label: "Spam", data: eda.char_length_histogram.spam, backgroundColor: "rgba(244, 63, 94, 0.85)" }
      ]
    }
  });

  destroyChart("spamTerms");
  chartInstances["spamTerms"] = new Chart(document.getElementById("chart-spam-terms"), {
    type: "bar",
    data: {
      labels: eda.top_spam_terms.map((t) => t.term),
      datasets: [{ label: "Spam Occurrences", data: eda.top_spam_terms.map((t) => t.count), backgroundColor: "#f43f5e" }]
    },
    options: { indexAxis: "y" }
  });

  destroyChart("hamTerms");
  chartInstances["hamTerms"] = new Chart(document.getElementById("chart-ham-terms"), {
    type: "bar",
    data: {
      labels: eda.top_ham_terms.map((t) => t.term),
      datasets: [{ label: "Ham Occurrences", data: eda.top_ham_terms.map((t) => t.count), backgroundColor: "#14b8a6" }]
    },
    options: { indexAxis: "y" }
  });
}

function renderComparisonSection(metrics) {
  const tbody = document.querySelector("#comparison-table tbody");
  tbody.innerHTML = "";

  metrics.comparison_table.forEach((row) => {
    const tr = document.createElement("tr");
    const isWinner = row.model_id === metrics.default_model_id;
    if (isWinner) tr.classList.add("winner-row");
    tr.innerHTML = `
      <td><strong>${row.model_name}</strong> ${isWinner ? '<span class="badge">Default</span>' : ""}</td>
      <td>${row.model_type}</td>
      <td>${row.val_f1.toFixed(4)}</td>
      <td>${(row.test_accuracy * 100).toFixed(2)}%</td>
      <td>${(row.test_precision * 100).toFixed(2)}%</td>
      <td>${(row.test_recall * 100).toFixed(2)}%</td>
      <td><strong>${row.test_f1.toFixed(4)}</strong></td>
      <td>${row.test_roc_auc.toFixed(4)}</td>
      <td>${row.test_pr_auc.toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });

  destroyChart("modelComp");
  chartInstances["modelComp"] = new Chart(document.getElementById("chart-model-comparison"), {
    type: "bar",
    data: {
      labels: metrics.comparison_table.map((r) => r.model_name),
      datasets: [
        { label: "Spam Precision", data: metrics.comparison_table.map((r) => r.test_precision), backgroundColor: "#14b8a6" },
        { label: "Spam Recall", data: metrics.comparison_table.map((r) => r.test_recall), backgroundColor: "#f59e0b" },
        { label: "Spam F1-Score", data: metrics.comparison_table.map((r) => r.test_f1), backgroundColor: "#f43f5e" },
        { label: "PR-AUC", data: metrics.comparison_table.map((r) => r.test_pr_auc), backgroundColor: "#38bdf8" }
      ]
    },
    options: { scales: { y: { min: 0.65, max: 1.0 } } }
  });

  updateModelInspector(metrics.default_model_id);
}

function updateModelInspector(modelId) {
  if (!dashboardState) return;
  const modelObj = dashboardState.metrics.models[modelId];
  if (!modelObj) return;

  document.getElementById("inspector-description").textContent = `${modelObj.name} (${modelObj.type}): ${modelObj.description}`;

  const cm = modelObj.test.confusion_matrix;
  document.getElementById("confusion-matrix-grid").innerHTML = `
    <div class="cm-cell"><span>True Negative (Ham → Ham)</span><strong class="text-teal">${cm[0][0]}</strong></div>
    <div class="cm-cell"><span>False Positive (Ham → Spam)</span><strong class="text-rose">${cm[0][1]}</strong></div>
    <div class="cm-cell"><span>False Negative (Spam → Ham)</span><strong class="text-rose">${cm[1][0]}</strong></div>
    <div class="cm-cell"><span>True Positive (Spam → Spam)</span><strong class="text-teal">${cm[1][1]}</strong></div>
  `;

  destroyChart("inspectorCurves");
  const rocPts = modelObj.test.roc_curve.fpr.map((x, i) => ({ x, y: modelObj.test.roc_curve.tpr[i] }));
  const prPts = modelObj.test.pr_curve.recall.map((x, i) => ({ x, y: modelObj.test.pr_curve.precision[i] }));

  chartInstances["inspectorCurves"] = new Chart(document.getElementById("chart-inspector-curves"), {
    type: "scatter",
    data: {
      datasets: [
        { label: `ROC (AUC=${modelObj.test.roc_auc})`, data: rocPts, showLine: true, borderColor: "#38bdf8", pointRadius: 0 },
        { label: `PR (AUC=${modelObj.test.pr_auc})`, data: prPts, showLine: true, borderColor: "#f43f5e", pointRadius: 0 }
      ]
    }
  });

  destroyChart("featureImp");
  const topFeats = (modelObj.top_features || []).slice(0, 10);
  chartInstances["featureImp"] = new Chart(document.getElementById("chart-feature-importance"), {
    type: "bar",
    data: {
      labels: topFeats.map((f) => f.feature),
      datasets: [{ label: "Importance", data: topFeats.map((f) => f.importance), backgroundColor: "#38bdf8" }]
    },
    options: { indexAxis: "y" }
  });
}

// Live SMS Prediction Form
const smsInput = document.getElementById("sms-input");
const charCounter = document.getElementById("char-counter");
const errorBox = document.getElementById("detector-error");

smsInput.addEventListener("input", () => {
  charCounter.textContent = `${smsInput.value.length} / 2000 characters`;
});

document.getElementById("clear-btn").addEventListener("click", () => {
  smsInput.value = "";
  charCounter.textContent = "0 / 2000 characters";
  errorBox.classList.add("hidden");
});

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    smsInput.value = SAMPLE_MESSAGES[chip.dataset.sample] || "";
    charCounter.textContent = `${smsInput.value.length} / 2000 characters`;
    errorBox.classList.add("hidden");
  });
});

document.getElementById("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.classList.add("hidden");

  const message = smsInput.value.trim();
  const modelId = document.getElementById("model-select").value;

  if (!message) {
    errorBox.textContent = "Please enter an SMS message before clicking Analyze.";
    errorBox.classList.remove("hidden");
    return;
  }

  document.getElementById("result-empty").classList.add("hidden");
  document.getElementById("result-content").classList.add("hidden");
  document.getElementById("result-loading").classList.remove("hidden");

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, model_id: modelId })
    });
    const res = await response.json();
    document.getElementById("result-loading").classList.add("hidden");

    if (!response.ok) {
      errorBox.textContent = res.error || "Prediction failed.";
      errorBox.classList.remove("hidden");
      return;
    }

    const bannerEl = document.getElementById("verdict-banner");
    const titleEl = document.getElementById("verdict-title");
    const scoreBarEl = document.getElementById("score-bar");
    const warningListEl = document.getElementById("warning-list");

    bannerEl.className = `verdict-banner ${res.is_spam ? "verdict-spam" : "verdict-ham"}`;
    titleEl.textContent = res.is_spam ? "SPAM (Suspicious Message)" : "HAM (Predicted Legitimate)";
    titleEl.className = res.is_spam ? "text-rose" : "text-teal";
    document.getElementById("verdict-model-badge").textContent = `Model: ${res.model_name}`;

    document.getElementById("score-percentage").textContent = `${res.estimated_spam_percentage}%`;
    scoreBarEl.style.width = `${Math.min(100, Math.max(2, res.estimated_spam_percentage))}%`;
    scoreBarEl.style.backgroundColor = res.estimated_spam_percentage >= 50 ? "#f43f5e" : "#14b8a6";

    warningListEl.innerHTML = "";
    if (!res.warning_signs || res.warning_signs.length === 0) {
      warningListEl.innerHTML = `<li class="warning-none">No rule-based warning patterns (links, OTP requests, prize claims, or urgent phrases) triggered.</li>`;
    } else {
      res.warning_signs.forEach((w) => {
        const li = document.createElement("li");
        li.className = "warning-item";
        li.innerHTML = `<strong>${w.title}</strong> (Matched: <code>${w.matched_text}</code>)<br/><span>${w.explanation}</span>`;
        warningListEl.appendChild(li);
      });
    }

    document.getElementById("result-disclaimer").textContent = res.disclaimer;
    document.getElementById("result-content").classList.remove("hidden");
  } catch (err) {
    document.getElementById("result-loading").classList.add("hidden");
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.classList.remove("hidden");
  }
});

document.addEventListener("DOMContentLoaded", initDashboard);