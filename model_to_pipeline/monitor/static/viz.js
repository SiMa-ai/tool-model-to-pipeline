let ops = [];
let table;
let chart;
let chartInitialized = false;
let currentView = "timeline";
let currentSearch = "";
let durationThreshold = 0;

window.addEventListener("DOMContentLoaded", loadData);

async function loadData() {
  const res = await fetch("/data");
  ops = await res.json();
  renderTable();

  // Do NOT init or draw the chart here, because stats tab might be hidden.
  // We'll init only when the stats tab is made visible.
}

/* Tab switching */
window.addEventListener("DOMContentLoaded", () => {
  const tabConfigBtn = document.getElementById("tab-config");
  const tabStatsBtn  = document.getElementById("tab-stats");
  const configTab    = document.getElementById("config-tab");
  const statsTab     = document.getElementById("stats-tab");

  function activateTab(activeBtn, activeContent) {
    document.querySelectorAll(".icon-tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    activeBtn.classList.add("active");
    activeContent.classList.add("active");

    // If we just revealed the stats tab, initialize + draw the chart now.
    if (activeContent === statsTab) {
      ensureStatsChart();
    }
  }

  tabConfigBtn.addEventListener("click", () => activateTab(tabConfigBtn, configTab));
  tabStatsBtn.addEventListener("click",  () => activateTab(tabStatsBtn,  statsTab));

  // If your page loads with STATS visible by default, call this once:
  if (statsTab.classList.contains("active")) {
    ensureStatsChart();
  }
});

/* Create/draw chart only when visible */
function ensureStatsChart() {
  if (!chartInitialized) {
    initChart();            // create echarts instance + observers
    drawByCurrentView();    // first draw
    chartInitialized = true;

    // Extra safety: re-measure after paint & fonts
    requestAnimationFrame(() => chart.resize());
    setTimeout(() => chart.resize(), 150);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(() => chart && chart.resize());
    }
  } else {
    // Stats tab became visible again; just ensure correct view + resize
    drawByCurrentView();
    requestAnimationFrame(() => chart.resize());
  }
}

function drawByCurrentView() {
  const view = document.getElementById("viewSelector")?.value || "timeline";
  if (view === "timeline") showTimeline();
  else if (view === "duration") showDurations();
  else showGantt();
}

/* Keep your renderTable() as-is */
function renderTable() {
  table = new Tabulator("#table", {
    data: ops,
    layout: "fitColumns",
    height: "350px",
    columns: [
      { title: "Idx", field: "idx", width: 60, sorter: "number" },
      { title: "Name", field: "name", widthGrow: 2 },
      { title: "Start", field: "start_cycle", sorter: "number" },
      { title: "End", field: "end_cycle", sorter: "number" },
      { title: "Duration", field: "duration", sorter: "number" },
    ],
  });
}

/* Init chart AFTER the stats panel is visible */
function initChart() {
  const chartEl = document.getElementById("chart");
  chart = echarts.init(chartEl);

  // Window resize
  window.addEventListener("resize", () => chart.resize());

  // Observe size changes of the content panel and chart container
  const contentEl = document.querySelector(".content");
  const ro = new ResizeObserver(() => chart && chart.resize());
  if (contentEl) ro.observe(contentEl);
  if (chartEl)   ro.observe(chartEl);

  // (Optional) keep a reference if you want to disconnect later
  chart._resizeObserver = ro;
}

/* View selector handler (unchanged) */
function handleViewChange() {
  currentView = document.getElementById("viewSelector").value;
  drawByCurrentView();
}


/* ────────────────────────────────
   Search, sort, and threshold
────────────────────────────────── */

function filterTable() {
  currentSearch = document.getElementById("searchBox").value.toLowerCase();
  if (!table) return;
  table.setFilter("name", "like", currentSearch);
  redrawChart();
}

function resetSort() {
  ops.sort((a, b) => a.idx - b.idx);
  table.setData(ops);
  showTimeline();
}

function sortByDuration() {
  ops.sort((a, b) => b.duration - a.duration);
  table.setData(ops);
  showDurations();
}

function updateThreshold() {
  const slider = document.getElementById("threshold-slider");
  durationThreshold = parseInt(slider.value);
  document.getElementById("threshold-value").textContent = durationThreshold;
  redrawChart();
}

/* ────────────────────────────────
   Safe Chart Redraw Handler
────────────────────────────────── */
/* ────────────────────────────────
   Safe Chart Redraw Handler
────────────────────────────────── */
function redrawChart() {
  chart.off("datazoom");     // remove old listener
  chart.clear();             // ✅ fully clear previous chart state

  if (currentView === "timeline") {
    drawTimeline(ops);
    attachZoomHandlerForTimeline();
  } else if (currentView === "duration") {
    drawDurations(ops);
    attachZoomHandlerForTimeline();
  } else if (currentView === "gantt") {
    drawGantt(ops);
    attachZoomHandlerForGantt();
  }
}

/* ────────────────────────────────
   Zoom label toggles
────────────────────────────────── */
function attachZoomHandlerForTimeline() {
  chart.on("datazoom", () => toggleLabelsBasedOnZoom());
}

function attachZoomHandlerForGantt() {
  chart.on("datazoom", () => toggleLabelsBasedOnZoom());
}

/* ────────────────────────────────
   GANTT VIEW (updated)
────────────────────────────────── */
function drawGantt(data) {
  const filtered = data.filter(o => o.duration >= durationThreshold);

  const option = {
    title: {
      text: "Gantt Chart (Start–End Cycles)",
      left: "center",
      textStyle: baseTextStyle,
    },
    textStyle: baseTextStyle,
    tooltip: {
      trigger: "item",
      textStyle: baseTextStyle,
      formatter: p => {
        const o = p.data.raw;
        return `${o.name}<br/>Start: ${o.start_cycle}<br/>End: ${o.end_cycle}<br/>Duration: ${o.duration} cycles`;
      },
    },
    xAxis: {
      type: "value",
      name: "Cycles",
      nameTextStyle: baseTextStyle,
      axisLabel: { ...baseTextStyle },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: filtered.map(o => o.name),
      axisLabel: { ...baseTextStyle },
    },
    dataZoom: [
      { type: "slider", yAxisIndex: 0 },
      { type: "inside", yAxisIndex: 0 },
    ],
    series: [
      {
        type: "custom",
        name: "Gantt",
        renderItem: (params, api) => {
          const start = api.value(0);
          const end = api.value(1);
          const name = api.value(2);
          const categoryIndex = api.value(3);
          const showLabel = api.value(4);
          const duration = end - start;

          const startCoord = api.coord([start, categoryIndex]);
          const endCoord = api.coord([end, categoryIndex]);
          const barHeight = api.size([0, 1])[1] * 0.6;

          return {
            type: "group",
            children: [
              {
                type: "rect",
                shape: {
                  x: startCoord[0],
                  y: startCoord[1] - barHeight / 2,
                  width: endCoord[0] - startCoord[0],
                  height: barHeight,
                },
                style: api.style({
                  fill: getBarColor(name),
                }),
              },
              ...(showLabel
                ? [
                    {
                      type: "text",
                      style: {
                        text: `${duration}`,
                        x: startCoord[0] + (endCoord[0] - startCoord[0]) / 2,
                        y: startCoord[1],
                        fill: "#222",
                        textAlign: "center",
                        textVerticalAlign: "middle",
                        font: "12px Roboto, sans-serif",
                      },
                    },
                  ]
                : []),
            ],
          };
        },
        encode: { x: [0, 1], y: 3 },
        data: filtered.map(o => ({
          value: [o.start_cycle, o.end_cycle, o.name, o.name, false],
          raw: o,
        })),
      },
    ],
  };

  chart.setOption(option, true);
}

/* ────────────────────────────────
   Gantt zoom-controlled label behavior
────────────────────────────────── */
function attachZoomLabelBehaviorForGantt() {
  chart.on("datazoom", () => {
    const option = chart.getOption();
    const zoom = option.dataZoom?.[0];
    if (!zoom) return;

    const start = zoom.start ?? 0;
    const end = zoom.end ?? 100;
    const visibleRatio = (end - start) / 100;
    const showLabels = visibleRatio < 0.1;

    // Update Gantt label visibility flag
    const series = option.series[0];
    series.data.forEach(d => (d.value[4] = showLabels));

    chart.setOption(option, false, true);
  });
}

/* ────────────────────────────────
   Chart drawing helpers
────────────────────────────────── */

function showTimeline() {
  currentView = "timeline";
  drawTimeline(ops);
}

function showDurations() {
  currentView = "duration";
  drawDurations(ops);
}

function showGantt() {
  currentView = "gantt";
  drawGantt(ops);
}

/* Common font style */
const baseTextStyle = {
  fontFamily: "Roboto, sans-serif",
  fontSize: 12,
  color: "#333",
};

/* ────────────────────────────────
   TIMELINE VIEW
────────────────────────────────── */
function drawTimeline(data) {
  const filtered = data.filter(o => o.duration >= durationThreshold);
  const option = {
    title: { text: "Operation Timeline", left: "center", textStyle: baseTextStyle },
    textStyle: baseTextStyle,
    tooltip: {
      trigger: "item",
      textStyle: baseTextStyle,
      formatter: p => {
        const o = filtered[p.dataIndex];
        return `${o.name}<br/>Start: ${o.start_cycle}<br/>End: ${o.end_cycle}<br/>Duration: ${o.duration} cycles`;
      },
    },
    xAxis: { type: "value", name: "Cycles", nameTextStyle: baseTextStyle, axisLabel: { ...baseTextStyle } },
    yAxis: { type: "category", inverse: true, data: filtered.map(o => o.name), axisLabel: { ...baseTextStyle } },
    dataZoom: [
      { type: "slider", yAxisIndex: 0 },
      { type: "inside", yAxisIndex: 0 },
    ],
    series: [
      {
        type: "bar",
        data: filtered.map(o => ({
          name: o.name,
          value: o.duration,
          itemStyle: { color: getBarColor(o.name) },
        })),
        label: { show: false, position: "right", formatter: "{c}", ...baseTextStyle },
      },
    ],
  };
  chart.setOption(option, true);
  toggleLabelsBasedOnZoom();
}

/* ────────────────────────────────
   DURATION VIEW
────────────────────────────────── */
function drawDurations(data) {
  const filtered = data.filter(o => o.duration >= durationThreshold);
  const sorted = [...filtered].sort((a, b) => b.duration - a.duration);
  const option = {
    title: { text: "Operation Durations (Descending)", left: "center", textStyle: baseTextStyle },
    textStyle: baseTextStyle,
    tooltip: {
      trigger: "item",
      textStyle: baseTextStyle,
      formatter: p => `${p.name}<br/>Duration: ${p.value} cycles`,
    },
    xAxis: { type: "value", name: "Cycles", nameTextStyle: baseTextStyle, axisLabel: { ...baseTextStyle } },
    yAxis: { type: "category", inverse: true, data: sorted.map(o => o.name), axisLabel: { ...baseTextStyle } },
    dataZoom: [
      { type: "slider", yAxisIndex: 0 },
      { type: "inside", yAxisIndex: 0 },
    ],
    series: [
      {
        type: "bar",
        data: sorted.map(o => ({
          name: o.name,
          value: o.duration,
          itemStyle: { color: getBarColor(o.name) },
        })),
        label: { show: false, position: "right", formatter: "{c}", ...baseTextStyle },
      },
    ],
  };
  chart.setOption(option, true);
  toggleLabelsBasedOnZoom();
}

/* ────────────────────────────────
   GANTT VIEW
────────────────────────────────── */
function drawGantt(data) {
  const filtered = data.filter(o => o.duration >= durationThreshold);

  const option = {
    title: {
      text: "Gantt Chart (Start–End Cycles)",
      left: "center",
      textStyle: baseTextStyle,
    },
    textStyle: baseTextStyle,
    tooltip: {
      trigger: "item",
      textStyle: baseTextStyle,
      formatter: p => {
        const o = p.data.raw;
        return `${o.name}<br/>Start: ${o.start_cycle}<br/>End: ${o.end_cycle}<br/>Duration: ${o.duration} cycles`;
      },
    },
    xAxis: {
      type: "value",
      name: "Cycles",
      nameTextStyle: baseTextStyle,
      axisLabel: { ...baseTextStyle },
    },
    yAxis: {
      type: "category",
      inverse: true,
      data: filtered.map(o => o.name),
      axisLabel: { ...baseTextStyle },
    },
    dataZoom: [
      { type: "slider", yAxisIndex: 0 },
      { type: "inside", yAxisIndex: 0 },
    ],
    series: [
      {
        type: "custom",
        name: "Gantt",
        // we keep a global showLabels flag here
        showLabels: false,
        renderItem: (params, api) => {
          const start = api.value(0);
          const end = api.value(1);
          const name = api.value(2);
          const categoryIndex = api.value(3);
          const duration = end - start;

          const startCoord = api.coord([start, categoryIndex]);
          const endCoord = api.coord([end, categoryIndex]);
          const barHeight = api.size([0, 1])[1] * 0.6;
          const showLabel = api.visual("showLabels");

          const groupChildren = [
            {
              type: "rect",
              shape: {
                x: startCoord[0],
                y: startCoord[1] - barHeight / 2,
                width: endCoord[0] - startCoord[0],
                height: barHeight,
              },
              style: api.style({
                fill: getBarColor(name),
              }),
            },
          ];

          if (showLabel) {
            groupChildren.push({
              type: "text",
              style: {
                text: `${duration}`,
                x: startCoord[0] + (endCoord[0] - startCoord[0]) / 2,
                y: startCoord[1],
                fill: "#222",
                textAlign: "center",
                textVerticalAlign: "middle",
                font: "12px Roboto, sans-serif",
              },
            });
          }

          return { type: "group", children: groupChildren };
        },
        encode: { x: [0, 1], y: 3 },
        data: filtered.map(o => ({
          value: [o.start_cycle, o.end_cycle, o.name, o.name],
          raw: o,
        })),
      },
    ],
  };

  chart.setOption(option, true);
}

/* ────────────────────────────────
   Dynamic label toggle for GANTT
────────────────────────────────── */
function attachZoomLabelBehaviorForGantt(filtered) {
  chart.off("datazoom");
  chart.on("datazoom", () => {
    const option = chart.getOption();
    const zoom = option.dataZoom?.[0];
    if (!zoom) return;

    const start = zoom.start ?? 0;
    const end = zoom.end ?? 100;
    const visibleRatio = (end - start) / 100;
    const showLabels = visibleRatio < 0.1;

    // Update data items with new label visibility flag
    const updatedSeries = option.series[0];
    updatedSeries.data = filtered.map(o => ({
      value: [
        o.start_cycle,
        o.end_cycle,
        o.name,
        o.name,
        showLabels, // controls label show/hide
      ],
      raw: o,
    }));

    chart.setOption(option, true);
  });
}


/* ────────────────────────────────
   Label zoom behavior
────────────────────────────────── */
function toggleLabelsBasedOnZoom() {
  if (!chart) return;
  const option = chart.getOption();
  const zoom = option.dataZoom?.[0];
  if (!zoom) return;

  const start = zoom.start ?? 0;
  const end = zoom.end ?? 100;
  const visibleRatio = (end - start) / 100;
  const showLabels = visibleRatio < 0.1;

  if (currentView === "gantt") {
    // custom series: pass flag as visual or global param
    chart.setOption({ series: [{ visual: { showLabels } }] }, false);
  } else {
    // normal bar chart
    option.series[0].label.show = showLabels;
    chart.setOption(option, false, true);
  }
}

/* ────────────────────────────────
   Highlight logic
────────────────────────────────── */
function getBarColor(name) {
  if (!currentSearch) return "#3498db"; // default
  return name.toLowerCase().includes(currentSearch)
    ? "#f39c12" // match highlight
    : "#d3d3d3"; // faded non-match
}
