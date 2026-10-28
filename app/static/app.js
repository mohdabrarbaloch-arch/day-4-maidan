/* Maidan — frontend.
   No framework, no build step. The interesting part is the SSE reader:
   tokens arrive as they are sampled, so the UI shows the model committing
   to each one instead of waiting for a finished string. */

(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const state = {
    models: [],
    activeModel: null,
    backend: "local",
    token: localStorage.getItem("maidan_token") || "",
    user: null,
    streaming: false,
    lastRun: null,
    authMode: "login",
  };

  const els = {
    modelList: $("modelList"),
    modelCount: $("modelCount"),
    activeModelName: $("activeModelName"),
    activeModelMeta: $("activeModelMeta"),
    prompt: $("prompt"),
    charCount: $("charCount"),
    tokenEstimate: $("tokenEstimate"),
    runBtn: $("runBtn"),
    clearBtn: $("clearBtn"),
    sampleBtn: $("sampleBtn"),
    output: $("output"),
    stream: $("stream"),
    emptyState: $("emptyState"),
    stats: $("stats"),
    statTokens: $("statTokens"),
    statLatency: $("statLatency"),
    statSeed: $("statSeed"),
    statStop: $("statStop"),
    replayBtn: $("replayBtn"),
    backendPill: $("backendPill"),
    backendDot: $("backendDot"),
    backendLabel: $("backendLabel"),
    backendHint: $("backendHint"),
    cloudSeg: $("cloudSeg"),
    authBtn: $("authBtn"),
    authModal: $("authModal"),
    authForm: $("authForm"),
    authError: $("authError"),
    authTitle: $("authTitle"),
    authSub: $("authSub"),
    authSubmit: $("authSubmit"),
    authSwitch: $("authSwitch"),
    authSwitchText: $("authSwitchText"),
    nameField: $("nameField"),
    historyModal: $("historyModal"),
    historyList: $("historyList"),
    historyBtn: $("historyBtn"),
    toast: $("toast"),
  };

  const params = {
    temperature: $("temperature"),
    topP: $("topP"),
    topK: $("topK"),
    maxTokens: $("maxTokens"),
    repPenalty: $("repPenalty"),
    seed: $("seed"),
  };

  const outputs = {
    temperature: $("temperatureOut"),
    topP: $("topPOut"),
    topK: $("topKOut"),
    maxTokens: $("maxTokensOut"),
    repPenalty: $("repPenaltyOut"),
    seed: $("seedOut"),
  };

  const DEFAULTS = {
    temperature: 0.8,
    topP: 0.95,
    topK: 40,
    maxTokens: 80,
    repPenalty: 1.15,
  };

  const EXAMPLES = {
    "maidan-tiny": [
      "The context window is",
      "A low temperature makes the distribution",
      "The most useful evaluation is",
    ],
    "maidan-coder": [
      "A function should do",
      "Every error you swallow is",
      "Health checks should exercise",
    ],
    "maidan-muse": [
      "The rain had stopped",
      "The bus was late and",
      "Morning came slowly, the way",
    ],
  };

  // ── helpers ──────────────────────────────────────────────────────
  function toast(message, bad = false) {
    els.toast.textContent = message;
    els.toast.className = "toast" + (bad ? " bad" : "");
    els.toast.hidden = false;
    clearTimeout(toast._t);
    toast._t = setTimeout(() => { els.toast.hidden = true; }, 3200);
  }

  function authHeaders() {
    return state.token ? { Authorization: `Bearer ${state.token}` } : {};
  }

  async function api(path, options = {}) {
    const res = await fetch(path, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(),
        ...(options.headers || {}),
      },
    });
    if (res.status === 204) return null;
    const text = await res.text();
    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { detail: text }; }
    if (!res.ok) {
      const detail = (data && (data.detail || data.error)) || `HTTP ${res.status}`;
      const err = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      err.status = res.status;
      throw err;
    }
    return data;
  }

  // ── models ───────────────────────────────────────────────────────
  async function loadModels() {
    try {
      const models = await api("/api/models");
      state.models = models;
      renderModels();
      if (!state.activeModel && models.length) selectModel(models[0].id);
    } catch (err) {
      els.modelList.innerHTML = `<p class="hint" style="padding:8px">Could not load models: ${err.message}</p>`;
    }
  }

  function renderModels() {
    const visible = state.models.filter((m) => m.backend === state.backend);
    els.modelCount.textContent = String(visible.length);

    if (!visible.length) {
      els.modelList.innerHTML =
        `<p class="hint" style="padding:10px">No ${state.backend} models available. ` +
        (state.backend === "cloud"
          ? "Set OPENAI_API_KEY or GROQ_API_KEY to enable the cloud backend."
          : "") +
        `</p>`;
      return;
    }

    els.modelList.innerHTML = visible
      .map((m) => {
        const active = m.id === state.activeModel ? " active" : "";
        const tags = (m.strengths || [])
          .map((s) => `<span class="tag${m.backend === "cloud" ? " cloud" : ""}">${s}</span>`)
          .join("");
        return `
          <button type="button" class="model-card${active}" data-model="${m.id}"
                  role="radio" aria-checked="${m.id === state.activeModel}">
            <div class="model-top">
              <span class="model-name">${m.name}</span>
              <span class="model-params">${m.params}</span>
            </div>
            <p class="model-desc">${m.description}</p>
            <div class="model-tags">${tags}</div>
          </button>`;
      })
      .join("");

    els.modelList.querySelectorAll(".model-card").forEach((card) => {
      card.addEventListener("click", () => selectModel(card.dataset.model));
    });
  }

  function selectModel(id) {
    const model = state.models.find((m) => m.id === id);
    if (!model) return;
    state.activeModel = id;
    els.activeModelName.textContent = model.name;
    els.activeModelMeta.textContent =
      `${model.family} · ${model.params} · ${model.context_window.toLocaleString()} ctx · ${model.backend}`;
    renderModels();
  }

  // ── backend status ───────────────────────────────────────────────
  async function loadBackends() {
    try {
      const status = await api("/api/backends");
      els.backendDot.className = "dot " + (status.local ? "ok" : "bad");
      els.backendLabel.textContent = status.cloud
        ? `local + ${status.providers.join(", ")}`
        : "local only";
      els.backendHint.textContent = status.message;
      els.cloudSeg.disabled = !status.cloud;
      if (!status.cloud && state.backend === "cloud") setBackend("local");
    } catch {
      els.backendDot.className = "dot bad";
      els.backendLabel.textContent = "offline";
    }
  }

  function setBackend(backend) {
    state.backend = backend;
    document.querySelectorAll(".seg").forEach((b) => {
      b.classList.toggle("active", b.dataset.backend === backend);
    });
    const first = state.models.find((m) => m.backend === backend);
    if (first) selectModel(first.id);
    else renderModels();
  }

  // ── params ───────────────────────────────────────────────────────
  function readParams() {
    const seedRaw = params.seed.value.trim();
    return {
      temperature: parseFloat(params.temperature.value),
      top_p: parseFloat(params.topP.value),
      top_k: parseInt(params.topK.value, 10),
      max_tokens: parseInt(params.maxTokens.value, 10),
      repetition_penalty: parseFloat(params.repPenalty.value),
      seed: seedRaw === "" ? null : parseInt(seedRaw, 10),
    };
  }

  function syncParamLabels() {
    outputs.temperature.textContent = parseFloat(params.temperature.value).toFixed(2);
    outputs.topP.textContent = parseFloat(params.topP.value).toFixed(2);
    outputs.topK.textContent = params.topK.value;
    outputs.maxTokens.textContent = params.maxTokens.value;
    outputs.repPenalty.textContent = parseFloat(params.repPenalty.value).toFixed(2);
    outputs.seed.textContent = params.seed.value.trim() || "random";
  }

  function resetParams() {
    params.temperature.value = DEFAULTS.temperature;
    params.topP.value = DEFAULTS.topP;
    params.topK.value = DEFAULTS.topK;
    params.maxTokens.value = DEFAULTS.maxTokens;
    params.repPenalty.value = DEFAULTS.repPenalty;
    params.seed.value = "";
    syncParamLabels();
  }

  // ── prompt meta ──────────────────────────────────────────────────
  function updatePromptMeta() {
    const text = els.prompt.value;
    els.charCount.textContent = text.length.toLocaleString();
    // Rough estimate: ~4 characters per token is the usual English heuristic.
    els.tokenEstimate.textContent = Math.ceil(text.length / 4).toLocaleString();
  }

  // ── generation ───────────────────────────────────────────────────
  function setStreaming(on) {
    state.streaming = on;
    els.runBtn.disabled = on;
    els.runBtn.querySelector(".btn-label").textContent = on ? "Generating" : "Generate";
    els.runBtn.querySelector(".spinner").hidden = !on;
  }

  function showStream() {
    els.emptyState.hidden = true;
    els.stream.hidden = false;
    els.stream.innerHTML = "";
  }

  function appendToken(text) {
    const cursor = els.stream.querySelector(".cursor");
    if (cursor) cursor.remove();
    els.stream.appendChild(document.createTextNode(text));
    const c = document.createElement("span");
    c.className = "cursor";
    els.stream.appendChild(c);
    els.output.scrollTop = els.output.scrollHeight;
  }

  function showError(message) {
    const div = document.createElement("span");
    div.className = "err";
    div.textContent = message;
    els.stream.appendChild(div);
  }

  async function generate() {
    if (state.streaming) return;

    const prompt = els.prompt.value.trim();
    if (!prompt) {
      toast("Write a prompt first.", true);
      els.prompt.focus();
      return;
    }
    if (!state.token) {
      openAuth("login");
      toast("Sign in to generate — runs are saved to your account.");
      return;
    }

    const body = {
      prompt,
      model_id: state.activeModel,
      backend: state.backend,
      ...readParams(),
    };

    setStreaming(true);
    showStream();
    els.stats.hidden = true;

    try {
      const res = await fetch("/api/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders() },
        body: JSON.stringify(body),
      });

      if (!res.ok) {
        const text = await res.text();
        let detail = `HTTP ${res.status}`;
        try {
          const parsed = JSON.parse(text);
          detail = parsed.detail || parsed.error || detail;
        } catch { /* keep the status code */ }
        if (res.status === 401) {
          state.token = "";
          localStorage.removeItem("maidan_token");
          updateAuthUI();
          openAuth("login");
        }
        showError(detail);
        return;
      }

      await readStream(res);
    } catch (err) {
      showError(err.message || "Request failed");
    } finally {
      setStreaming(false);
      const cursor = els.stream.querySelector(".cursor");
      if (cursor) cursor.remove();
    }
  }

  async function readStream(res) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        const line = frame.split("\n").find((l) => l.startsWith("data: "));
        if (!line) continue;
        let event;
        try { event = JSON.parse(line.slice(6)); } catch { continue; }
        handleEvent(event);
      }
    }
  }

  function handleEvent(event) {
    if (event.type === "token") {
      appendToken(event.token);
    } else if (event.type === "error") {
      showError(event.message);
    } else if (event.type === "done") {
      els.statTokens.textContent = event.tokens_generated ?? "—";
      els.statLatency.textContent = event.latency_ms != null ? `${event.latency_ms} ms` : "—";
      els.statSeed.textContent = event.seed ?? "—";
      els.statStop.textContent = event.stop_reason ?? "—";
      els.stats.hidden = false;
      state.lastRun = { ...event, prompt: els.prompt.value.trim(), model: state.activeModel };
    }
  }

  // ── auth ─────────────────────────────────────────────────────────
  function updateAuthUI() {
    if (state.token) {
      els.authBtn.textContent = state.user ? state.user.display_name : "Account";
      els.authBtn.title = state.user ? state.user.email : "";
    } else {
      els.authBtn.textContent = "Sign in";
      els.authBtn.title = "";
    }
  }

  function openAuth(mode) {
    state.authMode = mode;
    const isLogin = mode === "login";
    els.authTitle.textContent = isLogin ? "Sign in" : "Create an account";
    els.authSub.textContent = isLogin
      ? "Save your runs and replay them later."
      : "Runs are saved with their parameters and seed.";
    els.authSubmit.textContent = isLogin ? "Sign in" : "Create account";
    els.authSwitchText.textContent = isLogin ? "No account yet?" : "Already have one?";
    els.authSwitch.textContent = isLogin ? "Create one" : "Sign in";
    els.nameField.hidden = isLogin;
    els.authError.hidden = true;
    els.authModal.hidden = false;
    setTimeout(() => (isLogin ? $("authEmail") : $("authName")).focus(), 50);
  }

  async function submitAuth(e) {
    e.preventDefault();
    els.authError.hidden = true;
    const isLogin = state.authMode === "login";

    const payload = {
      email: $("authEmail").value.trim(),
      password: $("authPassword").value,
    };
    if (!isLogin) payload.display_name = $("authName").value.trim() || "Explorer";

    els.authSubmit.disabled = true;
    try {
      const data = await api(isLogin ? "/api/auth/login" : "/api/auth/register", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      state.token = data.access_token;
      state.user = data.user;
      localStorage.setItem("maidan_token", state.token);
      updateAuthUI();
      els.authModal.hidden = true;
      els.authForm.reset();
      toast(isLogin ? `Welcome back, ${data.user.display_name}` : "Account created");
    } catch (err) {
      els.authError.textContent = err.message;
      els.authError.hidden = false;
    } finally {
      els.authSubmit.disabled = false;
    }
  }

  async function restoreSession() {
    if (!state.token) return;
    try {
      state.user = await api("/api/auth/me");
    } catch {
      state.token = "";
      state.user = null;
      localStorage.removeItem("maidan_token");
    }
    updateAuthUI();
  }

  // ── history ──────────────────────────────────────────────────────
  async function openHistory() {
    if (!state.token) {
      openAuth("login");
      toast("Sign in to see your run history.");
      return;
    }
    els.historyModal.hidden = false;
    els.historyList.innerHTML = `<p class="history-empty">Loading…</p>`;
    try {
      const runs = await api("/api/runs?limit=25");
      renderHistory(runs);
    } catch (err) {
      els.historyList.innerHTML = `<p class="history-empty">${err.message}</p>`;
    }
  }

  function renderHistory(runs) {
    if (!runs.length) {
      els.historyList.innerHTML =
        `<p class="history-empty">No runs yet. Generate something and it will show up here.</p>`;
      return;
    }
    els.historyList.innerHTML = runs
      .map((r) => {
        const when = new Date(r.created_at).toLocaleString();
        return `
          <div class="history-item">
            <div class="history-top">
              <span class="history-model">${r.model_id} · ${r.backend}</span>
              <span class="history-time">${when}</span>
            </div>
            <p class="history-prompt">${escapeHtml(r.prompt)}</p>
            <p class="history-completion">${escapeHtml(r.completion)}</p>
            <div class="history-actions">
              <button class="btn ghost xs" data-replay="${r.id}">Load</button>
              <button class="btn ghost xs" data-delete="${r.id}">Delete</button>
            </div>
          </div>`;
      })
      .join("");

    els.historyList.querySelectorAll("[data-replay]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const run = runs.find((r) => String(r.id) === btn.dataset.replay);
        if (!run) return;
        els.prompt.value = run.prompt;
        params.temperature.value = run.temperature;
        params.topP.value = run.top_p;
        params.topK.value = run.top_k;
        params.maxTokens.value = run.max_tokens;
        params.repPenalty.value = run.repetition_penalty;
        params.seed.value = run.seed ?? "";
        syncParamLabels();
        updatePromptMeta();
        if (state.models.some((m) => m.id === run.model_id)) selectModel(run.model_id);
        els.historyModal.hidden = true;
        toast("Run loaded — press Generate to replay it.");
      });
    });

    els.historyList.querySelectorAll("[data-delete]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        try {
          await api(`/api/runs/${btn.dataset.delete}`, { method: "DELETE" });
          btn.closest(".history-item").remove();
          toast("Run deleted");
        } catch (err) {
          toast(err.message, true);
        }
      });
    });
  }

  function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
  }

  // ── wiring ───────────────────────────────────────────────────────
  function bind() {
    els.runBtn.addEventListener("click", generate);

    els.prompt.addEventListener("input", updatePromptMeta);
    els.prompt.addEventListener("keydown", (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
        e.preventDefault();
        generate();
      }
    });

    els.clearBtn.addEventListener("click", () => {
      els.prompt.value = "";
      updatePromptMeta();
      els.stream.hidden = true;
      els.stream.innerHTML = "";
      els.emptyState.hidden = false;
      els.stats.hidden = true;
      els.prompt.focus();
    });

    els.sampleBtn.addEventListener("click", () => {
      const pool = EXAMPLES[state.activeModel] || EXAMPLES["maidan-tiny"];
      els.prompt.value = pool[Math.floor(Math.random() * pool.length)];
      updatePromptMeta();
      els.prompt.focus();
    });

    els.replayBtn.addEventListener("click", () => {
      if (!state.lastRun) return;
      params.seed.value = state.lastRun.seed ?? "";
      syncParamLabels();
      generate();
    });

    Object.values(params).forEach((input) => {
      input.addEventListener("input", syncParamLabels);
    });

    $("resetParams").addEventListener("click", resetParams);
    $("randomSeed").addEventListener("click", () => {
      params.seed.value = Math.floor(Math.random() * 2147483647);
      syncParamLabels();
    });

    document.querySelectorAll(".seg").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        setBackend(btn.dataset.backend);
      });
    });

    els.authBtn.addEventListener("click", () => {
      if (state.token) {
        state.token = "";
        state.user = null;
        localStorage.removeItem("maidan_token");
        updateAuthUI();
        toast("Signed out");
      } else {
        openAuth("login");
      }
    });

    els.authForm.addEventListener("submit", submitAuth);
    els.authSwitch.addEventListener("click", () =>
      openAuth(state.authMode === "login" ? "register" : "login")
    );
    $("authClose").addEventListener("click", () => { els.authModal.hidden = true; });

    els.historyBtn.addEventListener("click", openHistory);
    $("historyClose").addEventListener("click", () => { els.historyModal.hidden = true; });

    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.hidden = true;
      });
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        els.authModal.hidden = true;
        els.historyModal.hidden = true;
      }
    });
  }

  async function init() {
    bind();
    syncParamLabels();
    updatePromptMeta();
    updateAuthUI();
    await Promise.all([loadModels(), loadBackends(), restoreSession()]);
  }

  document.addEventListener("DOMContentLoaded", init);
})();
