/* Federated-learning run report — all figures are built here from the JSON
   payload the Python script embeds, so light/dark and the round selector
   re-render the same code path instead of duplicating figure definitions.

   Adapted from the CIFAR-10 example's report.js. Differences: clients are
   named sites (ucd, unipd) rather than integer ids, the metric panels plot the
   scalars fl_server.py logs (there is no shared test set), and there is no
   confusion-matrix panel. */

(function () {
  "use strict";

  const P = window.__FL_PAYLOAD__;

  /* ---------------------------------------------------------------- theme --
     Colours come from the design system's validated slots: eight categorical
     hues in fixed order (identity), and a blue<->red diverging pair with a
     neutral grey midpoint for polarity. */

  const THEMES = {
    light: {
      surface: "#fcfcfb",
      ink: "#0b0b0b",
      secondary: "#52514e",
      muted: "#898781",
      grid: "#e1e0d9",
      axis: "#c3c2b7",
      series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
      // Polarity: warm <-> cool poles, neutral grey at zero.
      diverging: [[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
      hoverBg: "#ffffff",
    },
    dark: {
      surface: "#1a1a19",
      ink: "#ffffff",
      secondary: "#c3c2b7",
      muted: "#898781",
      grid: "#2c2c2a",
      axis: "#383835",
      series: ["#3987e5", "#d95926", "#199e70", "#c98500",
               "#d55181", "#008300", "#9085e9", "#e66767"],
      diverging: [[0, "#e66767"], [0.5, "#383835"], [1, "#3987e5"]],
      hoverBg: "#242422",
    },
  };

  const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
  const SYMBOLS = ["circle", "diamond", "square", "cross", "x"];
  const DASHES = ["solid", "dash", "dot"];

  let mode = "light";
  const state = {
    round: P.run.rounds.length ? P.run.rounds[P.run.rounds.length - 1] : null,
  };

  const t = () => THEMES[mode];

  /* Colour follows the site id, not its position in a round: slot order is
     fixed, and past eight sites the symbol/dash carries identity so hue is
     never the only channel. */
  const clientIndex = {};
  P.run.client_ids.forEach((cid, i) => { clientIndex[cid] = i; });
  const clientName = (cid) => P.run.client_names[cid] || cid;
  const clientColor = (cid, theme) =>
    (theme || t()).series[clientIndex[cid] % 8];
  const clientSymbol = (cid) =>
    SYMBOLS[Math.floor(clientIndex[cid] / 8) % SYMBOLS.length];
  const clientDash = (cid) =>
    DASHES[Math.floor(clientIndex[cid] / 8) % DASHES.length];

  /* --------------------------------------------------------------- layout -- */

  function baseLayout(extra) {
    const c = t();
    return Object.assign({
      paper_bgcolor: c.surface,
      plot_bgcolor: c.surface,
      font: { family: FONT, size: 12, color: c.secondary },
      margin: { l: 54, r: 18, t: 10, b: 44 },
      hoverlabel: {
        bgcolor: c.hoverBg,
        bordercolor: c.axis,
        font: { family: FONT, size: 12.5, color: c.ink },
      },
      legend: {
        orientation: "h",
        y: -0.22,
        x: 0,
        font: { size: 11.5, color: c.secondary },
        bgcolor: "rgba(0,0,0,0)",
      },
      showlegend: true,
    }, extra || {});
  }

  function axis(title, extra) {
    const c = t();
    return Object.assign({
      title: title ? { text: title, font: { size: 11.5, color: c.muted } } : undefined,
      gridcolor: c.grid,
      griddash: "solid",
      zeroline: false,
      linecolor: c.axis,
      tickfont: { size: 11, color: c.muted },
      automargin: true,
    }, extra || {});
  }

  /* A hairline marking the round the selector is pointing at. */
  function roundMarker() {
    if (state.round === null) return [];
    return [{
      type: "line", xref: "x", yref: "paper",
      x0: state.round, x1: state.round, y0: 0, y1: 1,
      line: { color: t().axis, width: 1 },
      layer: "below",
    }];
  }

  const CONFIG = { displayModeBar: false, responsive: true };
  const CONFIG_3D = {
    responsive: true, displaylogo: false,
    modeBarButtonsToRemove: ["resetCameraLastSave3d", "hoverClosest3d"],
  };

  /* ------------------------------------------------- metrics over rounds -- */

  /* Emphasis, not equal series: the n-weighted global figure is the heavy ink
     line on top; the per-site series are thin context beneath it. `block` is
     P.evals.val or P.evals.train. Missing values arrive as null and plot as
     gaps. */
  function metricFigure(block, key, axisTitle, asPercent, globalName) {
    const c = t();
    const traces = [];
    const scale = (v) => (v === null ? null : (asPercent ? v * 100 : v));
    const hover = asPercent ? "%{y:.1f}%" : "%{y:.3f}";

    P.run.client_ids.forEach((cid) => {
      const rows = block.clients[cid] || [];
      if (!rows.length) return;
      traces.push({
        type: "scatter", mode: "lines+markers",
        name: clientName(cid),
        legendgroup: "c" + cid,
        x: rows.map((r) => r.round),
        y: rows.map((r) => scale(r[key])),
        customdata: rows.map((r) => r.n),
        line: { color: clientColor(cid), width: 1.5, dash: clientDash(cid) },
        marker: { size: 6, symbol: clientSymbol(cid), color: clientColor(cid) },
        opacity: 0.72,
        hovertemplate: hover + " (n=%{customdata})<extra>" + clientName(cid) + "</extra>",
      });
    });

    const g = block.global || [];
    if (g.length) {
      const y = g.map((r) => scale(r[key]));
      const last = y.length - 1;
      traces.push({
        type: "scatter", mode: "lines+markers+text",
        name: globalName,
        x: g.map((r) => r.round),
        y: y,
        line: { color: c.ink, width: 2.5, shape: "linear" },
        marker: { size: 7, color: c.ink, line: { color: c.surface, width: 2 } },
        // Label the endpoint only; the axis and the tooltip carry the rest.
        text: y.map((v, i) => (i === last && v !== null
          ? (asPercent ? v.toFixed(1) + "%" : v.toFixed(3)) : "")),
        textposition: "top left",
        textfont: { size: 12, color: c.ink },
        cliponaxis: false,
        legendrank: 0,
        hovertemplate: hover + "<extra>" + globalName + "</extra>",
      });
    }

    const nRounds = P.run.rounds.length;
    const layout = baseLayout({
      hovermode: "x unified",
      xaxis: axis("Round", { dtick: nRounds > 30 ? 5 : 1, showgrid: false }),
      yaxis: axis(axisTitle, asPercent ? { ticksuffix: "%", rangemode: "tozero" } : {}),
      shapes: roundMarker(),
      margin: { l: 58, r: 34, t: 10, b: 40 },
    });
    return { data: traces, layout: layout };
  }

  /* ------------------------------------------------------- similarity ----- */

  function heatmap(ids, matrix, opts) {
    const c = t();
    const labels = ids.map(clientName);
    const showText = ids.length <= 8;
    return {
      type: "heatmap",
      z: matrix,
      x: labels,
      y: labels,
      zmin: opts.zmin, zmax: opts.zmax,
      colorscale: opts.colorscale,
      xgap: 2, ygap: 2,
      text: matrix.map((row) => row.map(opts.fmt)),
      texttemplate: showText ? "%{text}" : undefined,
      textfont: { size: 11 },
      hovertemplate: "%{y} vs %{x}<br>%{text}<extra></extra>",
      showscale: true,
      colorbar: {
        thickness: 12, len: 0.82, outlinewidth: 0, ticklen: 3,
        tickfont: { size: 10.5, color: c.muted },
        title: { text: opts.scaleTitle, font: { size: 10.5, color: c.muted },
                 side: "right" },
      },
    };
  }

  function simLayout() {
    return baseLayout({
      showlegend: false,
      xaxis: axis(null, { side: "top", showgrid: false, ticks: "" }),
      yaxis: axis(null, { autorange: "reversed", showgrid: false, ticks: "" }),
      margin: { l: 46, r: 10, t: 26, b: 10 },
    });
  }

  function simFigure(block) {
    if (!block) return null;
    const fmt = (v) => (v === null ? "n/a" : (v >= 0 ? "+" : "") + v.toFixed(2));
    return {
      data: [heatmap(block.ids, block.matrix, {
        zmin: P.similarity.range[0], zmax: P.similarity.range[1],
        colorscale: t().diverging, fmt: fmt,
        scaleTitle: "opposed ← cosine → aligned",
      })],
      layout: simLayout(),
    };
  }

  function simTrendFigure() {
    const c = t();
    const rows = P.similarity.trend;
    const ys = rows.map((r) => r.avg);
    const lo = Math.min(0, ...ys), hi = Math.max(0, ...ys);
    const pad = Math.max(0.05, 0.18 * (hi - lo));
    const nRounds = P.run.rounds.length;
    return {
      data: [{
        type: "scatter", mode: "lines+markers",
        x: rows.map((r) => r.round), y: ys,
        line: { color: c.series[0], width: 2 },
        marker: { size: 7, color: c.series[0], line: { color: c.surface, width: 2 } },
        hovertemplate: "Round %{x}<br>mean pairwise %{y:+.3f}<extra></extra>",
      }],
      layout: baseLayout({
        showlegend: false,
        hovermode: "x unified",
        xaxis: axis("Round", { dtick: nRounds > 30 ? 5 : 1, showgrid: false }),
        yaxis: axis("Mean pairwise cosine", { range: [lo - pad, hi + pad], tickformat: "+.2f" }),
        shapes: roundMarker().concat([{
          type: "line", xref: "paper", yref: "y",
          x0: 0, x1: 1, y0: 0, y1: 0,
          line: { color: c.axis, width: 1 }, layer: "below",
        }]),
        margin: { l: 62, r: 18, t: 10, b: 40 },
      }),
    };
  }

  /* -------------------------------------------------------- trajectory ---- */

  function trajectoryFigure() {
    const c = t();
    const traces = [];
    const g = P.pca.global;
    const byRound = {};
    g.forEach((d) => { byRound[d.round] = d.xyz; });
    const rounds = g.map((d) => d.round);

    traces.push({
      type: "scatter3d", mode: "lines+markers+text",
      name: "Global model",
      x: g.map((d) => d.xyz[0]), y: g.map((d) => d.xyz[1]), z: g.map((d) => d.xyz[2]),
      line: { color: c.ink, width: 5 },
      marker: { size: 4, color: c.ink },
      text: rounds.map((r) => "R" + r),
      textposition: "top center",
      textfont: { size: 10, color: c.secondary },
      hovertemplate: "Global model, %{text}<extra></extra>",
      legendrank: 0,
    });

    /* One pair of traces per site (not per round): a gapped line trace for
       the pull arrows and a marker trace for the endpoints, sharing a legend
       entry. A client update in round r starts from the global model of r-1. */
    const clientRounds = Object.keys(P.pca.clients).map(Number).sort((a, b) => a - b);
    P.run.client_ids.forEach((cid) => {
      const lx = [], ly = [], lz = [], mx = [], my = [], mz = [], mr = [];
      clientRounds.forEach((r) => {
        const to = (P.pca.clients[String(r)] || {})[cid];
        if (!to) return;
        mx.push(to[0]); my.push(to[1]); mz.push(to[2]); mr.push(r);
        const from = byRound[r - 1];
        if (!from) return;
        lx.push(from[0], to[0], null);
        ly.push(from[1], to[1], null);
        lz.push(from[2], to[2], null);
      });
      if (!mx.length) return;
      const col = clientColor(cid);
      if (lx.length) {
        traces.push({
          type: "scatter3d", mode: "lines",
          x: lx, y: ly, z: lz,
          line: { color: col, width: 2, dash: "dot" },
          legendgroup: "c" + cid, showlegend: false,
          opacity: 0.75, hoverinfo: "skip",
        });
      }
      traces.push({
        type: "scatter3d", mode: "markers",
        name: clientName(cid),
        x: mx, y: my, z: mz,
        marker: { size: 5, color: col, symbol: clientSymbol(cid) },
        legendgroup: "c" + cid,
        customdata: mr,
        hovertemplate: clientName(cid) + " update, round %{customdata}<extra></extra>",
      });
    });

    const sceneAxis = (title) => ({
      title: { text: title, font: { size: 11, color: c.muted } },
      backgroundcolor: c.surface,
      gridcolor: c.grid,
      zerolinecolor: c.axis,
      showbackground: true,
      tickfont: { size: 10, color: c.muted },
    });

    return {
      data: traces,
      layout: baseLayout({
        margin: { l: 0, r: 0, t: 0, b: 0 },
        legend: { orientation: "h", y: 0, x: 0, font: { size: 11.5, color: c.secondary } },
        scene: {
          xaxis: sceneAxis("PC 1"), yaxis: sceneAxis("PC 2"), zaxis: sceneAxis("PC 3"),
          bgcolor: c.surface,
          camera: { eye: { x: 1.6, y: 1.5, z: 1.1 } },
        },
      }),
    };
  }

  /* ------------------------------------------------------------ rendering -- */

  function draw(id, fig, config) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!fig) {
      Plotly.purge(el);
      el.innerHTML = "";
      return;
    }
    if (el.querySelector(".empty") || !el.querySelector(".plot-container")) {
      Plotly.purge(el);
      el.innerHTML = "";
    }
    Plotly.react(el, fig.data, fig.layout, config || CONFIG);
  }

  function renderMetrics() {
    const v = P.evals.val, tr = P.evals.train;
    draw("fig-val-auc", metricFigure(v, "auc", "Validation AUC", false, "Global (n-weighted)"));
    draw("fig-val-acc", metricFigure(v, "acc", "Validation accuracy", true, "Global (n-weighted)"));
    draw("fig-val-loss", metricFigure(v, "loss", "Validation loss", false, "Global (n-weighted)"));
    draw("fig-train-auc", metricFigure(tr, "auc", "Train AUC", false, "All sites (n-weighted)"));
    draw("fig-train-acc", metricFigure(tr, "acc", "Train accuracy", true, "All sites (n-weighted)"));
    draw("fig-train-loss", metricFigure(tr, "loss", "Train loss", false, "All sites (n-weighted)"));
  }

  function renderSimilarity() {
    if (!P.similarity.mean) return;
    draw("fig-sim-mean", simFigure(P.similarity.mean));
    const block = P.similarity.per_round[String(state.round)] || null;
    const el = document.getElementById("fig-sim-round");
    const note = document.getElementById("sim-round-note");
    if (block) {
      if (note) note.textContent = "Round " + state.round;
      draw("fig-sim-round", simFigure(block));
    } else {
      if (note) note.textContent = "Round " + state.round + " — not available";
      draw("fig-sim-round", null);
      if (el) el.innerHTML = '<p class="empty">Round ' + state.round
        + " has no comparable update deltas (it needs the previous round's "
        + "global model and at least two sites).</p>";
    }
    draw("fig-sim-trend", simTrendFigure());
  }

  function renderTrajectory() {
    if (!P.pca.global.length) return;
    draw("fig-trajectory", trajectoryFigure(), CONFIG_3D);
  }

  function renderAll() {
    renderMetrics();
    renderSimilarity();
    renderTrajectory();
  }

  /* --------------------------------------------------------------- wiring -- */

  function setTheme(next) {
    mode = next;
    document.documentElement.dataset.theme = next;
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent = next === "dark" ? "Light theme" : "Dark theme";
      btn.setAttribute("aria-label", "Switch to " + (next === "dark" ? "light" : "dark") + " theme");
    }
    renderAll();
  }

  function init() {
    const prefersDark = window.matchMedia
      && window.matchMedia("(prefers-color-scheme: dark)").matches;
    mode = prefersDark ? "dark" : "light";
    document.documentElement.dataset.theme = mode;

    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.textContent = mode === "dark" ? "Light theme" : "Dark theme";
      btn.addEventListener("click", () => setTheme(mode === "dark" ? "light" : "dark"));
    }

    const roundSel = document.getElementById("sel-round");
    const roundVal = document.getElementById("round-val");
    if (roundSel) {
      const initialIdx = P.run.rounds.indexOf(state.round);
      if (initialIdx >= 0) {
        roundSel.value = String(initialIdx);
      }
      if (roundVal && state.round !== null) {
        roundVal.textContent = state.round;
      }
      const updateRound = (e) => {
        const idx = Number(e.target.value);
        const r = P.run.rounds[idx];
        if (r !== undefined && r !== state.round) {
          state.round = r;
          if (roundVal) roundVal.textContent = r;
          renderMetrics();
          renderSimilarity();
        }
      };
      roundSel.addEventListener("input", updateRound);
      roundSel.addEventListener("change", updateRound);
    }

    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", (e) => setTheme(e.matches ? "dark" : "light"));
    }

    renderAll();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
