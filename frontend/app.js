// HemaScan frontend — plain JavaScript, no build step.

if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const RISK_LABEL = {
  none: "No anaemia",
  mild: "Mild anaemia",
  moderate: "Moderate anaemia",
  severe: "Severe anaemia",
};
const GROUP_LABEL = {
  woman: "Woman", pregnant: "Pregnant", man: "Man",
  child_6_23m: "Baby 6–23 mo", child_24_59m: "Child 2–4", child_5_11y: "Child 5–11", child_12_14y: "Child 12–14",
};
const SPEECH_LANG = { en: "en-IN", hi: "hi-IN", te: "te-IN" };
const SAMPLES = ["/samples/sample_healthy.jpg", "/samples/sample_mild.jpg", "/samples/sample_pale.jpg"];

const state = {
  group: "woman",
  file: null,
  img: null,
  box: null, // {x,y,w,h} in canvas pixels
  result: null,
};

/* ---------------- Tabs ---------------- */
$$(".tab").forEach((t) =>
  t.addEventListener("click", () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === t));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${t.dataset.view}`));
    if (t.dataset.view === "dashboard") loadDashboard();
    if (t.dataset.view === "about") loadStatus();
  })
);

/* ---------------- Step navigation ---------------- */
function goStep(n) {
  [1, 2, 3].forEach((i) => $(`#step-${i}`).classList.toggle("hidden", i !== n));
  $$(".stepper li").forEach((li) => {
    const s = Number(li.dataset.step);
    li.classList.toggle("active", s === n);
    li.classList.toggle("done", s < n);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

$("#group-chips").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  state.group = chip.dataset.group;
  $$("#group-chips .chip").forEach((c) => c.classList.toggle("active", c === chip));
});

$("#to-step-2").addEventListener("click", () => goStep(2));
$("#back-1").addEventListener("click", () => goStep(1));
$("#new-screening").addEventListener("click", () => {
  resetPhoto();
  $("#patient-ref").value = "";
  $("#age").value = "";
  speechSynthesis.cancel();
  goStep(1);
});

/* ---------------- Villages ---------------- */
fetch("/api/villages")
  .then((r) => r.json())
  .then((list) => {
    const sel = $("#village");
    list.forEach((v) => sel.add(new Option(v.name, v.name)));
  });

/* ---------------- Photo input ---------------- */
const fileInput = $("#file");
$("#take-photo").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => fileInput.files[0] && loadFile(fileInput.files[0]));
$("#retake").addEventListener("click", () => {
  resetPhoto();
  fileInput.click();
});
$("#use-sample").addEventListener("click", async (e) => {
  e.preventDefault();
  const url = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  const blob = await (await fetch(url)).blob();
  loadFile(new File([blob], url.split("/").pop(), { type: blob.type }), true);
});

const drop = $("#drop");
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); })
);
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); })
);
drop.addEventListener("drop", (e) => e.dataTransfer.files[0] && loadFile(e.dataTransfer.files[0]));

function resetPhoto() {
  state.file = null;
  state.img = null;
  state.box = null;
  fileInput.value = "";
  $(".drop-empty").classList.remove("hidden");
  $(".drop-preview").classList.add("hidden");
}

function loadFile(file, isSample = false) {
  if (!file.type.startsWith("image/")) return alert("Please choose an image file.");
  state.file = file;
  const img = new Image();
  img.onload = () => {
    state.img = img;
    const canvas = $("#canvas");
    const scale = Math.min(1, 900 / Math.max(img.width, img.height));
    canvas.width = Math.round(img.width * scale);
    canvas.height = Math.round(img.height * scale);
    // Default box: the lower-middle band, where the inner eyelid usually is.
    state.box = isSample
      ? { x: canvas.width * 0.3, y: canvas.height * 0.58, w: canvas.width * 0.4, h: canvas.height * 0.14 }
      : { x: canvas.width * 0.2, y: canvas.height * 0.45, w: canvas.width * 0.6, h: canvas.height * 0.3 };
    $(".drop-empty").classList.add("hidden");
    $(".drop-preview").classList.remove("hidden");
    draw();
  };
  img.src = URL.createObjectURL(file);
}

/* ---------------- Region selection (drag / resize box) ---------------- */
const canvas = $("#canvas");
const ctx = canvas.getContext("2d");
let drag = null;

function draw() {
  if (!state.img) return;
  const { x, y, w, h } = state.box;
  ctx.drawImage(state.img, 0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "rgba(0,0,0,.45)";
  ctx.fillRect(0, 0, canvas.width, y);
  ctx.fillRect(0, y + h, canvas.width, canvas.height - y - h);
  ctx.fillRect(0, y, x, h);
  ctx.fillRect(x + w, y, canvas.width - x - w, h);
  const lw = Math.max(2, canvas.width / 250);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = lw;
  ctx.setLineDash([lw * 4, lw * 3]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  const hs = handleSize();
  ctx.fillStyle = "#b91c3c";
  ctx.fillRect(x + w - hs / 2, y + h - hs / 2, hs, hs);
  ctx.strokeRect(x + w - hs / 2, y + h - hs / 2, hs, hs);
}
const handleSize = () => Math.max(18, canvas.width / 25);

function toCanvas(e) {
  const r = canvas.getBoundingClientRect();
  return { x: ((e.clientX - r.left) / r.width) * canvas.width, y: ((e.clientY - r.top) / r.height) * canvas.height };
}

canvas.addEventListener("pointerdown", (e) => {
  if (!state.box) return;
  canvas.setPointerCapture(e.pointerId);
  const p = toCanvas(e);
  const b = state.box;
  const hs = handleSize() * 1.2;
  if (Math.abs(p.x - (b.x + b.w)) < hs && Math.abs(p.y - (b.y + b.h)) < hs) {
    drag = { mode: "resize" };
  } else if (p.x > b.x && p.x < b.x + b.w && p.y > b.y && p.y < b.y + b.h) {
    drag = { mode: "move", dx: p.x - b.x, dy: p.y - b.y };
  } else {
    drag = { mode: "new", ox: p.x, oy: p.y };
    state.box = { x: p.x, y: p.y, w: 10, h: 10 };
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const p = toCanvas(e);
  const b = state.box;
  const W = canvas.width, H = canvas.height;
  if (drag.mode === "move") {
    b.x = Math.min(Math.max(0, p.x - drag.dx), W - b.w);
    b.y = Math.min(Math.max(0, p.y - drag.dy), H - b.h);
  } else if (drag.mode === "resize") {
    b.w = Math.min(Math.max(30, p.x - b.x), W - b.x);
    b.h = Math.min(Math.max(20, p.y - b.y), H - b.y);
  } else {
    b.x = Math.max(0, Math.min(drag.ox, p.x));
    b.y = Math.max(0, Math.min(drag.oy, p.y));
    b.w = Math.max(10, Math.min(W, Math.abs(p.x - drag.ox)));
    b.h = Math.max(10, Math.min(H, Math.abs(p.y - drag.oy)));
  }
  draw();
});
["pointerup", "pointercancel"].forEach((ev) => canvas.addEventListener(ev, () => {
  if (drag && state.box.w < 30) Object.assign(state.box, { w: 30 });
  if (drag && state.box.h < 20) Object.assign(state.box, { h: 20 });
  drag = null;
  draw();
}));

/* ---------------- Analyse ---------------- */
$("#analyze").addEventListener("click", analyse);

async function analyse() {
  if (!state.file) return;
  const b = state.box;
  const roi = { x: b.x / canvas.width, y: b.y / canvas.height, w: b.w / canvas.width, h: b.h / canvas.height };
  const fd = new FormData();
  fd.append("image", state.file);
  fd.append("roi", JSON.stringify(roi));
  fd.append("group", state.group);
  fd.append("age", $("#age").value);
  fd.append("sex", ["woman", "pregnant"].includes(state.group) ? "F" : state.group === "man" ? "M" : "");
  fd.append("village", $("#village").value);
  fd.append("lang", $("#lang").value);
  fd.append("patient_ref", $("#patient-ref").value);

  showLoading(true);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");
    state.result = data;
    renderResult(data);
    goStep(3);
  } catch (err) {
    alert(err.message);
  } finally {
    showLoading(false);
  }
}

let loadingTimer;
function showLoading(on) {
  $("#loading").classList.toggle("hidden", !on);
  clearInterval(loadingTimer);
  if (!on) return;
  const msgs = ["Checking photo quality…", "Finding the inner eyelid…", "Measuring pallor…", "Estimating haemoglobin…", "Writing explanation…"];
  let i = 0;
  $("#loading-text").textContent = msgs[0];
  loadingTimer = setInterval(() => { i = Math.min(i + 1, msgs.length - 1); $("#loading-text").textContent = msgs[i]; }, 600);
}

function renderResult(d) {
  const hero = $("#result-hero");
  hero.className = `card result-hero ${d.quality.ok ? d.risk : ""}`;
  $("#risk-badge").textContent = d.quality.ok ? RISK_LABEL[d.risk] : "Retake photo";
  const ok = d.quality.ok;
  // A rejected photo gets no number: an estimate from a bad photo would mislead.
  $("#hb-value").textContent = ok ? d.hb_estimate.toFixed(1) : "—";
  $("#hb-range").textContent = ok
    ? `Likely range ${d.hb_range[0]}–${d.hb_range[1]} g/dL · anaemia below ${d.thresholds.normal} for ${GROUP_LABEL[state.group]}`
    : "No estimate: please retake the photo using the tips below";
  $(".gauge").classList.toggle("hidden", !ok);
  $(".result-grid").classList.toggle("hidden", !ok);
  const pos = Math.min(100, Math.max(0, ((d.hb_estimate - 6) / 12) * 100));
  const t = d.thresholds; // colour bands follow this group's WHO cut-offs
  $(".g-severe").style.flex = t.moderate - 6;
  $(".g-moderate").style.flex = t.mild - t.moderate;
  $(".g-mild").style.flex = t.normal - t.mild;
  $(".g-normal").style.flex = 18 - t.normal;
  $("#gauge-marker").style.left = "0%";
  requestAnimationFrame(() => setTimeout(() => ($("#gauge-marker").style.left = pos + "%"), 50));
  $("#confidence-pill").textContent = `Confidence: ${d.confidence}`;
  $("#confidence-pill").classList.toggle("hidden", !ok);
  $("#method-pill").textContent = d.method + (d.calibration.startsWith("default") ? " · uncalibrated demo" : "");

  const warn = $("#quality-warn");
  if (d.quality.issues.length) {
    warn.classList.remove("hidden");
    warn.innerHTML = `<b>${d.quality.ok ? "Photo could be better" : "Photo not good enough — result not saved"}</b><ul>${d.quality.issues.map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul>`;
  } else warn.classList.add("hidden");

  $("#img-crop").src = d.crop;
  $("#img-heat").src = d.heatmap;
  $("#explanation").textContent = d.explanation.text;
  $("#disclaimer").textContent = "⚠️ " + d.explanation.disclaimer;

  const ref = $("#referral");
  if (d.referral && d.quality.ok) {
    ref.classList.remove("hidden");
    ref.innerHTML = `<div class="icon">🏥</div><div><h3>Refer to ${escapeHtml(d.referral.name)}</h3>
      <p>${d.referral.distance_km} km away · visit <b>${d.referral.urgency}</b> for a blood test${d.risk === "severe" ? " and treatment" : " and iron supplements"}.</p></div>`;
  } else ref.classList.add("hidden");
}

/* ---------------- Voice ---------------- */
$("#speak").addEventListener("click", () => {
  if (!state.result) return;
  const lang = $("#lang").value;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(state.result.explanation.text);
  u.lang = SPEECH_LANG[lang];
  const voice = speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith(lang));
  if (voice) u.voice = voice;
  else if (lang !== "en") alert("This device has no voice installed for this language, so it may read with an English voice. On Android: Settings → Text-to-speech → install the Hindi/Telugu voice.");
  u.rate = 0.92;
  speechSynthesis.speak(u);
});
speechSynthesis.getVoices();

/* ---------------- Dashboard ---------------- */
let map, layer;
async function loadDashboard() {
  const s = await (await fetch("/api/stats")).json();
  $("#s-total").textContent = s.total;
  $("#s-rate").textContent = s.anaemia_rate + "%";
  $("#s-severe").textContent = s.by_risk.severe;
  $("#s-hotspot").textContent = s.villages[0] ? s.villages[0].village : "—";

  const max = Math.max(1, ...Object.values(s.by_risk));
  $("#risk-bars").innerHTML = ["none", "mild", "moderate", "severe"].map((k) => `
    <div class="bar-row"><span>${RISK_LABEL[k].replace(" anaemia", "")}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${(100 * s.by_risk[k]) / max}%;background:var(--${k})"></div></div>
    <b>${s.by_risk[k]}</b></div>`).join("");

  $("#village-table tbody").innerHTML = s.villages.map((v) => `
    <tr><td>${escapeHtml(v.village)}</td><td>${v.total}</td><td><b style="color:${rateColor(v.rate)}">${v.rate}%</b></td><td>${v.avg_hb}</td></tr>`).join("")
    || `<tr><td colspan="4" class="muted">No screenings yet. Click “Load demo data”.</td></tr>`;

  const recent = await (await fetch("/api/screenings?limit=25")).json();
  $("#recent-table tbody").innerHTML = recent.map((r) => `
    <tr><td>${r.created_at.slice(0, 10)}</td><td>${escapeHtml(r.patient_ref || "—")}</td><td>${GROUP_LABEL[r.grp] || r.grp}</td>
    <td>${escapeHtml(r.village || "—")}</td><td>${r.hb_estimate}</td><td><span class="tag ${r.risk}">${r.risk}</span></td>
    <td>${escapeHtml(r.referred_to || "—")}</td></tr>`).join("")
    || `<tr><td colspan="7" class="muted">No screenings yet.</td></tr>`;

  drawMap(s);
}

function rateColor(rate) {
  if (rate >= 60) return "#dc2626";
  if (rate >= 45) return "#ea580c";
  if (rate >= 30) return "#ca8a04";
  return "#16a34a";
}

function drawMap(s) {
  if (typeof L === "undefined") {
    $("#map").innerHTML = '<p class="muted" style="padding:24px">Map needs an internet connection to load.</p>';
    return;
  }
  if (!map) {
    map = L.map("map").setView([17.98, 79.6], 10);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18, attribution: "© OpenStreetMap",
    }).addTo(map);
    layer = L.layerGroup().addTo(map);
  }
  layer.clearLayers();
  s.health_centres.forEach((c) => {
    L.marker([c.lat, c.lng], {
      icon: L.divIcon({ className: "", html: '<div style="font-size:20px">🏥</div>', iconSize: [20, 20] }),
    }).bindPopup(`<b>${escapeHtml(c.name)}</b><br>${c.type}`).addTo(layer);
  });
  s.villages.filter((v) => v.lat != null).forEach((v) => {
    L.circleMarker([v.lat, v.lng], {
      radius: 8 + Math.sqrt(v.total) * 2.5, color: rateColor(v.rate), fillColor: rateColor(v.rate), fillOpacity: 0.45, weight: 2,
    }).bindPopup(`<b>${escapeHtml(v.village)}</b><br>${v.total} screened · <b>${v.rate}%</b> positive<br>Avg Hb ${v.avg_hb} g/dL · ${v.severe} severe`).addTo(layer);
  });
  setTimeout(() => {
    map.invalidateSize();
    const pts = s.villages.filter((v) => v.lat != null).map((v) => [v.lat, v.lng]);
    if (pts.length) map.fitBounds(pts, { padding: [40, 40], maxZoom: 12 });
  }, 100);
}

$("#seed-demo").addEventListener("click", async () => { await fetch("/api/demo/seed", { method: "POST" }); loadDashboard(); });
$("#clear-demo").addEventListener("click", async () => { await fetch("/api/demo", { method: "DELETE" }); loadDashboard(); });

/* ---------------- About / status ---------------- */
async function loadStatus() {
  const h = await (await fetch("/api/health")).json();
  $("#sys-status").textContent = `System: CNN model ${h.cnn_model ? "loaded" : "not loaded (colour method)"} · calibration: ${h.calibration} · AI explanations: ${h.llm ? "Claude" : "built-in templates"}`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
