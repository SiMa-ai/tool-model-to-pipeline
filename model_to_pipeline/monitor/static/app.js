/*
 * Copyright (c) 2025 SiMa.ai
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

let currentStream = null;
let logBuffer = [];
let flushTimer = null;

/* --------------------------------------------------------------------------
   Frontend Script for Model-to-Pipeline Monitor
   -------------------------------------------------------------------------- */

document.addEventListener("DOMContentLoaded", () => {
  // Show Specification by default when page loads
  selectStep(1);
  updateStepStates(); // initialize once
  setInterval(updateStepStates, 2000); // poll backend every 2s
});

/* --------------------------------------------------------------------------
   YAML Rendering
   -------------------------------------------------------------------------- */

/**
 * Render YAML sections into tabbed tables.
 */
function renderYamlTabs() {
  if (!yamlSections || yamlSections.length === 0) {
    return "<p style='color: gray'>No YAML loaded.</p>";
  }

  let tabs = `<ul class="nav nav-tabs" id="yamlTab" role="tablist">`;
  let content = `<div class="tab-content mt-3">`;

  yamlSections.forEach(([section, rows], idx) => {
    const active = idx === 0 ? "active" : "";
    const show = idx === 0 ? "show active" : "";

    tabs += `
      <li class="nav-item" role="presentation">
        <button class="nav-link ${active}" id="tab-${idx}" data-bs-toggle="tab"
          data-bs-target="#content-${idx}" type="button" role="tab">
          ${section}
        </button>
      </li>`;

    const tableRows = rows.map(r => `
      <tr>
        <td><code>${r.field}</code></td>
        <td>${formatValue(r.value)}</td>
        <td>${sanitizeComment(r.desc)}</td>
      </tr>`
    ).join("");

    content += `
      <div class="tab-pane fade ${show}" id="content-${idx}" role="tabpanel">
        <div class="table-responsive">
          <table class="table table-striped align-middle">
            <thead class="table-light">
              <tr>
                <th style="width: 30%">Field</th>
                <th style="width: 30%">Value</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>${tableRows}</tbody>
          </table>
        </div>
      </div>`;
  });

  tabs += `</ul>`;
  content += `</div>`;
  return tabs + content;
}

/**
 * Render formatted values with styling for booleans.
 */
function formatValue(val) {
  if (!val) return "";
  const lower = val.toLowerCase();
  if (lower === "true") return `<span class="value-text" style="color:green">✅ true</span>`;
  if (lower === "false") return `<span class="value-text" style="color:red">❌ false</span>`;
  return `<span class="value-text">${val}</span>`;
}

/**
 * Clean and highlight inline comments in YAML.
 */
function sanitizeComment(text) {
  if (!text) return "";
  let cleaned = text.replace(/^#+\s*/, "");
  return cleaned.replace(/`([^`]+)`/g,
    (m, p1) => `<span style="font-family:'Courier New', monospace; font-size:12px; color:#c7254e;">${p1}</span>`
  );
}

/* --------------------------------------------------------------------------
   Log Rendering Helpers
   -------------------------------------------------------------------------- */

/**
 * Convert ANSI escape sequences to HTML with inline styles.
 */
function ansiToHtml(text) {
  if (!text) return "";
  return text
    .replace(/\x1b\[1m/g, "<span style='font-weight:bold'>")
    .replace(/\x1b\[0m/g, "</span>")
    .replace(/\x1b\[32m/g, "<span style='color:#4caf50'>") // green
    .replace(/\x1b\[31m/g, "<span style='color:#f44336'>") // red
    .replace(/\x1b\[33m/g, "<span style='color:#ff9800'>") // yellow
    .replace(/\x1b\[34m/g, "<span style='color:#2196f3'>"); // blue
}

/**
 * Escape HTML special characters.
 */
function escapeHtml(unsafe) {
  return unsafe.replace(/[&<"'>]/g, m => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}

/* --------------------------------------------------------------------------
   Step State Handling
   -------------------------------------------------------------------------- */

/**
 * Poll backend for step states and update UI accordingly.
 */
function updateStepStates() {
  fetch("/state")
    .then(res => res.json())
    .then(states => {
      window.currentStates = states;

      // Reset all steps
      document.querySelectorAll(".step").forEach(step => {
        step.classList.remove("in-progress", "success", "fail", "disabled");
      });

      document.querySelectorAll(".step").forEach(step => {
        const stepName = step.dataset.step;

        if (stepName === "Specification") {
          // Always enabled
          return;
        }

        const stepKey = Object.keys(stepKeyMap).find(
          key => stepKeyMap[key] === stepName
        );
        const status = stepKey ? states[stepKey] : null;

        if (!status) {
          // Not yet executed → disable
          step.classList.add("disabled");
          return;
        }

        if (status === "started") {
          step.classList.add("in-progress");
        } else if (status === "success") {
          step.classList.add("success");
        } else if (status === "fail") {
          step.classList.add("fail");
        }
      });
    })
    .catch(err => console.error("Error fetching step state:", err));
}

/**
 * Start log streaming for a step using SSE.
 */
function startLogStream(stepKey, filename) {
  const content = document.getElementById("step-content");
  content.classList.add("logs");
  content.innerHTML = `<h3>${filename}</h3><pre class="log-box" id="log-output"></pre>`;
  const logOutput = document.getElementById("log-output");

  // close old stream if exists
  if (currentStream) {
    currentStream.close();
  }

  currentStream = new EventSource(`/logs/${stepKey}/stream`);

  currentStream.onmessage = function(e) {
    logBuffer.push(ansiToHtml(escapeHtml(e.data)) + "\n");

    if (!flushTimer) {
      flushTimer = setTimeout(() => {
        // Limit log length to avoid memory blowup
        if (logBuffer.length > 0) {
          logOutput.insertAdjacentHTML("beforeend", logBuffer.join(""));
          logOutput.scrollTop = logOutput.scrollHeight;
          logBuffer = [];
        }
        flushTimer = null;
      }, 300); // flush every 300ms
    }
  };

  currentStream.onerror = function(err) {
    console.error("Log stream error:", err);
    currentStream.close();
    currentStream = null;
  };
}

/**
 * Handle user clicking a step circle.
 */
function selectStep(stepId) {
  const stepElem = document.getElementById("step-" + stepId);
  const stepName = stepElem.dataset.step;
  const content = document.getElementById("step-content");

  // If disabled and not Specification → show popup
  if (stepElem.classList.contains("disabled") && stepName !== "Specification") {
    alert("This step has not been executed yet.");
    return;
  }

  // Handle Specification
  if (stepName === "Specification") {
    // Close any previous stream
    if (currentStream) {
      currentStream.close();
      currentStream = null;
    }

    content.classList.remove("logs");
    content.innerHTML = renderYamlTabs();
    setActive(stepElem);
    return;
  }

  // Map step ID → log prefix
  const logPrefixes = {
    2: "downloadmodel",
    3: "surgery",
    4: "downloadcalib",
    5: "compile",
    6: "pipelinecreate",
    7: "mpkcreate"
  };

  const stepKey = logPrefixes[stepId];

  if (stepKey) {
    // Close any previous stream
    if (currentStream) {
      currentStream.close();
      currentStream = null;
    }

    // Fetch filename once, then start streaming
    fetch(`/logs/${stepKey}`)
      .then(res => res.json())
      .then(data => {
        startLogStream(stepKey, data.filename);
      })
      .catch(() => {
        content.classList.add("logs");
        content.innerHTML = `<p style="color:red">Failed to load log.</p>`;
      });
  } else {
    // No logs for this step
    if (currentStream) {
      currentStream.close();
      currentStream = null;
    }

    content.classList.remove("logs");
    content.innerHTML = `<h2>No log available for this step</h2>`;
  }

  setActive(stepElem);
}

/**
 * Highlight the currently active step.
 */
function setActive(stepElem) {
  document.querySelectorAll(".step").forEach(step => step.classList.remove("active"));
  stepElem.classList.add("active");
}

/* --------------------------------------------------------------------------
   Step Key Mapping (Backend → Frontend)
   -------------------------------------------------------------------------- */
const stepKeyMap = {
  "spec": "Specification",
  "downloadmodel": "Download Model",
  "surgery": "Model Surgery",
  "downloadcalib": "Download Calibration Data",
  "compile": "Model Compilation",
  "pipelinecreate": "Pipeline Generation",
  "mpkcreate": "MPK Compilation"
};
