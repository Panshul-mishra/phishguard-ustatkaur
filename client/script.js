const LEVEL_COLORS = {
  "Safe": "#22c55e",
  "Low Risk": "#84cc16",
  "Medium Risk": "#eab308",
  "High Risk": "#f97316",
  "Phishing": "#ef4444",
};

const urlInput = document.getElementById("urlInput");
const checkBtn = document.getElementById("checkBtn");
const errorBox = document.getElementById("error");
const resultBox = document.getElementById("result");

async function checkUrl() {
  const url = urlInput.value.trim();
  errorBox.style.display = "none";
  if (!url) return;

  checkBtn.disabled = true;
  checkBtn.textContent = "Checking...";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `request failed (${res.status})`);

    const badge = document.getElementById("levelBadge");
    badge.textContent = data.level;
    badge.style.background = LEVEL_COLORS[data.level] || "#93a0c4";

    const pct = Math.round(data.score * 100);
    document.getElementById("scoreVal").textContent = pct + "%";
    document.getElementById("meterFill").style.width = pct + "%";

    const reasonsEl = document.getElementById("reasons");
    reasonsEl.innerHTML = "";
    data.explanation.forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      reasonsEl.appendChild(li);
    });

    document.getElementById("metaRow").textContent = `inference time: ${data.latency_ms} ms`;
    resultBox.style.display = "block";
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.style.display = "block";
    resultBox.style.display = "none";
  } finally {
    checkBtn.disabled = false;
    checkBtn.textContent = "Check URL";
  }
}

checkBtn.addEventListener("click", checkUrl);
urlInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") checkUrl();
});
document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    urlInput.value = chip.dataset.url;
    checkUrl();
  });
});
