/* ==========================================================================
   Fake News Detection - Frontend Interactions
   NOTE: This file contains ONLY UI logic. All Machine Learning prediction
   happens on the Flask/Python backend (see app.py / model.py).
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initClearButton();
  initCharCounter();
  initFormSubmitLoading();
  initPipelineStages();
  initStatsChart();
});

/* ---------------------------------------------------------------------
   CLEAR BUTTON
--------------------------------------------------------------------- */
function initClearButton() {
  const clearBtn = document.getElementById("clearBtn");
  if (!clearBtn) return;

  clearBtn.addEventListener("click", () => {
    const headline = document.getElementById("headline");
    const article = document.getElementById("article");
    if (headline) headline.value = "";
    if (article) article.value = "";
    updateCharCounter();
  });
}

/* ---------------------------------------------------------------------
   CHARACTER COUNTER + MAX LENGTH VALIDATION
--------------------------------------------------------------------- */
const MAX_CHARS = 10000;

function updateCharCounter() {
  const article = document.getElementById("article");
  const headline = document.getElementById("headline");
  const counter = document.getElementById("charCounter");
  const warning = document.getElementById("charWarning");
  if (!article || !counter) return;

  const total = (headline ? headline.value.length : 0) + article.value.length;
  counter.textContent = `${total} / ${MAX_CHARS} characters`;

  if (warning) {
    warning.textContent = total > MAX_CHARS ? "Maximum length exceeded." : "";
  }

  const checkBtn = document.getElementById("checkBtn");
  if (checkBtn && total > MAX_CHARS) {
    checkBtn.setAttribute("disabled", "disabled");
  } else if (checkBtn && !checkBtn.dataset.modelMissing) {
    checkBtn.removeAttribute("disabled");
  }
}

function initCharCounter() {
  const article = document.getElementById("article");
  const headline = document.getElementById("headline");
  if (!article) return;

  article.addEventListener("input", updateCharCounter);
  if (headline) headline.addEventListener("input", updateCharCounter);
  updateCharCounter();
}

/* ---------------------------------------------------------------------
   FORM SUBMISSION: VALIDATION + LOADING STATE
--------------------------------------------------------------------- */
function initFormSubmitLoading() {
  const form = document.getElementById("detectForm");
  const checkBtn = document.getElementById("checkBtn");
  if (!form || !checkBtn) return;

  form.addEventListener("submit", (event) => {
    const headline = document.getElementById("headline").value.trim();
    const article = document.getElementById("article").value.trim();

    if (!headline && !article) {
      event.preventDefault();
      alert("Please enter a news headline or article before checking.");
      return;
    }

    const total = headline.length + article.length;
    if (total > MAX_CHARS) {
      event.preventDefault();
      alert(`Please limit your text to ${MAX_CHARS.toLocaleString()} characters.`);
      return;
    }

    // Show loading state and disable the button to prevent duplicate submits.
    const btnText = checkBtn.querySelector(".btn-text");
    const btnLoading = checkBtn.querySelector(".btn-loading");
    if (btnText && btnLoading) {
      btnText.classList.add("d-none");
      btnLoading.classList.remove("d-none");
    }
    checkBtn.setAttribute("disabled", "disabled");
  });
}

/* ---------------------------------------------------------------------
   PIPELINE STAGE EXPLANATIONS
--------------------------------------------------------------------- */
const STAGE_EXPLANATIONS = {
  collection:
    "News articles are collected from an appropriate dataset containing labelled examples.",
  cleaning:
    "Text is normalized (lowercased, HTML/URLs/punctuation removed) and unnecessary content is stripped out.",
  extraction:
    "TF-IDF converts the cleaned text into numerical features that reflect how distinctive each word is.",
  ml:
    "Logistic Regression learns patterns that separate REAL and FAKE examples from labelled training data.",
  prediction:
    "The trained model classifies new input as REAL or FAKE, with a confidence score when available.",
};

function initPipelineStages() {
  const stages = document.querySelectorAll(".pipeline-stage");
  const explanationBox = document.getElementById("stageExplanation");
  const explanationText = document.getElementById("stageText");
  if (!stages.length || !explanationBox || !explanationText) return;

  stages.forEach((stage) => {
    stage.addEventListener("click", () => {
      stages.forEach((s) => s.classList.remove("active"));
      stage.classList.add("active");

      const key = stage.getAttribute("data-stage");
      explanationText.textContent = STAGE_EXPLANATIONS[key] || "";
      explanationBox.classList.remove("d-none");
    });
  });
}

/* ---------------------------------------------------------------------
   DASHBOARD CHART (current-session stats only - no fabricated history)
--------------------------------------------------------------------- */
function initStatsChart() {
  const canvas = document.getElementById("statsChart");
  if (!canvas || typeof Chart === "undefined") return;

  const real = parseInt(canvas.dataset.real || "0", 10);
  const fake = parseInt(canvas.dataset.fake || "0", 10);

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["Real News", "Fake News"],
      datasets: [
        {
          label: "Current Session Predictions",
          data: [real, fake],
          backgroundColor: ["#1a9e5e", "#d64545"],
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } },
      },
    },
  });
}
