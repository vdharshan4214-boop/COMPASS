const API_BASE = (window.location.hostname === "localhost" && window.location.port === "8080")
  ? "http://localhost:8000"
  : "";

function saveSession(data) {
  localStorage.setItem("compass_token", data.token);
  localStorage.setItem("compass_role", data.role);
  localStorage.setItem("compass_name", data.name);
}

function getToken() { return localStorage.getItem("compass_token"); }
function getRole() { return localStorage.getItem("compass_role"); }
function getName() { return localStorage.getItem("compass_name"); }

function logout() {
  localStorage.removeItem("compass_token");
  localStorage.removeItem("compass_role");
  localStorage.removeItem("compass_name");
  window.location.href = "login.html";
}

function requireRole(role) {
  if (getToken() === null || getRole() !== role) {
    window.location.href = "login.html";
  }
}

async function api(path, { method = "GET", body = null, isForm = false } = {}) {
  const headers = {};
  const token = getToken();
  if (token) headers["Authorization"] = "Bearer " + token;
  let payload = body;
  if (body && !isForm) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(API_BASE + path, { method, headers, body: payload });
  if (res.status === 401) {
    logout();
    return null;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || "Something went wrong");
  }
  return data;
}

function parseIsoUtc(iso) {
  if (!iso) return new Date();
  let s = String(iso);
  if (!s.endsWith("Z") && !s.includes("+") && !s.includes("-", 10)) {
    s += "Z";
  }
  return new Date(s);
}

function timeAgo(iso) {
  if (!iso) return "just now";
  const dateObj = parseIsoUtc(iso);
  const diffMs = Date.now() - dateObj.getTime();
  if (isNaN(diffMs) || diffMs < 0) return "just now";
  const mins = Math.floor(diffMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function formatDateTime(iso) {
  if (!iso) return "N/A";
  const d = parseIsoUtc(iso);
  if (isNaN(d.getTime())) return String(iso);
  return d.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  });
}

function statusLabel(s) {
  return { submitted: "Submitted", in_progress: "In Progress", resolved: "Resolved" }[s] || s;
}


function ensureAdminChatWidget() {
  if (document.getElementById("compassChatWidget")) return;
  if (getRole() !== "admin") return;

  const widget = document.createElement("div");
  widget.id = "compassChatWidget";
  widget.className = "chat-widget";
  widget.innerHTML = `
    <div class="chat-widget-header">
      <div>
        <strong>Compass AI</strong>
        <small>Admin assistant</small>
      </div>
      <button type="button" class="chat-widget-minimize" aria-label="Toggle chat">−</button>
    </div>
    <div class="chat-widget-suggestions">
      <button class="chat-suggestion" type="button">Is Block A rectified?</button>
      <button class="chat-suggestion" type="button">What is the status of electrical issues?</button>
    </div>
    <div id="chatWindow" class="chat-window"></div>
    <div class="chat-widget-input-row">
      <input id="chatInput" type="text" placeholder="Ask about complaint status..." maxlength="300">
      <button type="button" class="btn btn-primary btn-sm" onclick="sendChatMessage()">Send</button>
    </div>
  `;
  document.body.appendChild(widget);

  const chatWindow = document.getElementById("chatWindow");
  const chatInput = document.getElementById("chatInput");
  const chatSuggestions = widget.querySelectorAll(".chat-suggestion");
  const chatHistory = [];

  function addChatBubble(role, content, sources = []) {
    const bubble = document.createElement("div");
    bubble.className = `chat-bubble chat-${role}`;
    const text = document.createElement("div");
    text.className = "chat-message";
    text.textContent = content;
    bubble.appendChild(text);

    if (sources && sources.length) {
      const row = document.createElement("div");
      row.className = "chat-source-row";
      sources.forEach((source) => {
        const tag = document.createElement("span");
        tag.className = "chat-source-tag";
        const parts = [source.id ? `#${source.id}` : null, source.category, source.location, source.status].filter(Boolean).slice(0,4);
        tag.textContent = parts.join(" • ");
        row.appendChild(tag);
      });
      bubble.appendChild(row);
    }

    chatWindow.appendChild(bubble);
    chatWindow.scrollTop = chatWindow.scrollHeight;
  }

  window.sendChatMessage = async function () {
    const message = (chatInput.value || "").trim();
    if (!message) return;
    chatHistory.push({ role: "user", content: message });
    addChatBubble("user", message);
    chatInput.value = "";

    try {
      const data = await api("/api/admin/chatbot", { method: "POST", body: { message, history: chatHistory } });
      const answer = data.answer || "I couldn't find a matching answer in the data.";
      addChatBubble("assistant", answer, Array.isArray(data.sources) ? data.sources : []);
      chatHistory.push({ role: "assistant", content: answer });
    } catch (error) {
      addChatBubble("assistant", "I couldn't answer that right now: " + error.message);
    }
  };

  chatInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      window.sendChatMessage();
    }
  });

  chatSuggestions.forEach((button) => {
    button.addEventListener("click", () => {
      chatInput.value = button.textContent.trim();
      window.sendChatMessage();
    });
  });

  const minimizeBtn = widget.querySelector(".chat-widget-minimize");
  minimizeBtn.addEventListener("click", () => {
    widget.classList.toggle("collapsed");
    minimizeBtn.textContent = widget.classList.contains("collapsed") ? "+" : "−";
  });

  addChatBubble("assistant", "Ask about a location or status, and I’ll answer from the complaint records.");
}

if (document.readyState === "loading") {
  window.addEventListener("DOMContentLoaded", ensureAdminChatWidget);
} else {
  ensureAdminChatWidget();
}

const COMPASS_ICON = `<svg viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
<circle cx="20" cy="20" r="18" stroke="currentColor" stroke-width="1.6"/>
<path d="M20 8 L24 20 L20 32 L16 20 Z" fill="currentColor"/>
<circle cx="20" cy="20" r="2.2" fill="var(--paper,#F6F3EC)"/>
</svg>`;
