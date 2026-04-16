async function loadSummary() {
  const response = await fetch("./site_data/summary.json");
  if (!response.ok) {
    throw new Error("Unable to load summary.json");
  }
  return response.json();
}

function fmtNumber(value, digits = 2) {
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPct(value, digits = 2) {
  return `${fmtNumber(Number(value) * 100, digits)}%`;
}

function metricCard(label, value) {
  return `
    <div class="metric-item">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `;
}

function stackedStat(label, value) {
  return `
    <div class="stacked-stat">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `;
}

function barRow(label, value, cssClass) {
  const width = Math.max(0, Math.min(100, value * 100));
  return `
    <div class="bar-row">
      <span><strong>${label}</strong><strong>${fmtPct(value, 1)}</strong></span>
      <div class="bar-track">
        <div class="bar-fill ${cssClass}" style="width:${width}%"></div>
      </div>
    </div>
  `;
}

function renderMetrics(containerId, summary) {
  const container = document.getElementById(containerId);
  container.innerHTML = [
    metricCard("Trades", fmtNumber(summary.trade_count, 0)),
    metricCard("Net PnL", `$${fmtNumber(summary.net_pnl_sum)}`),
    metricCard("Gross PnL", `$${fmtNumber(summary.gross_pnl_sum)}`),
    metricCard("Sharpe", fmtNumber(summary.sharpe_ratio, 2)),
    metricCard("Expected Return / Trade", fmtPct(summary.expected_return_per_trade)),
    metricCard("Win Rate", fmtPct(summary.win_rate)),
    metricCard("Avg Duration", `${fmtNumber(summary.average_trade_duration_seconds)}s`),
    metricCard("Max Drawdown", `$${fmtNumber(summary.max_drawdown)}`),
  ].join("");
}

function renderDataOverview(dataOverview) {
  document.getElementById("data-overview").innerHTML = [
    stackedStat("Rows After Filters", fmtNumber(dataOverview.rows_after_filtering, 0)),
    stackedStat("Contracts", fmtNumber(dataOverview.contract_count, 0)),
    stackedStat("Train / Validation / Test", `${fmtNumber(dataOverview.train_rows, 0)} / ${fmtNumber(dataOverview.validation_rows, 0)} / ${fmtNumber(dataOverview.test_rows, 0)}`),
    stackedStat("Coverage", `${dataOverview.time_range_start} -> ${dataOverview.time_range_end}`),
  ].join("");
}

function renderExitBars(summary) {
  document.getElementById("exit-bars").innerHTML = [
    barRow("Success", summary.success_rate, "success"),
    barRow("Stop-loss", summary.stop_loss_rate, "stop"),
    barRow("Timeout", summary.timeout_rate, "timeout"),
  ].join("");
}

function tableFromRows(rows, columns) {
  const header = columns.map((column) => `<th>${column.label}</th>`).join("");
  const body = rows
    .map((row) => {
      const cells = columns
        .map((column) => `<td>${row[column.key] ?? ""}</td>`)
        .join("");
      return `<tr>${cells}</tr>`;
    })
    .join("");
  return `<table><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function renderTables(summary) {
  document.getElementById("blotter-table").innerHTML = tableFromRows(
    summary.blotter_preview,
    [
      { key: "trade_id", label: "Trade" },
      { key: "entry_time", label: "Entry" },
      { key: "direction", label: "Dir" },
      { key: "alpha", label: "Alpha" },
      { key: "entry_price", label: "Entry Px" },
      { key: "exit_price", label: "Exit Px" },
      { key: "net_pnl", label: "Net PnL" },
      { key: "exit_type", label: "Exit Type" },
      { key: "hold_seconds", label: "Hold (s)" },
    ],
  );

  document.getElementById("ledger-table").innerHTML = tableFromRows(
    summary.ledger_preview,
    [
      { key: "trade_id", label: "Trade" },
      { key: "timestamp", label: "Timestamp" },
      { key: "capital_after", label: "Capital" },
      { key: "cumulative_pnl", label: "Cum PnL" },
      { key: "drawdown", label: "Drawdown" },
      { key: "rolling_sharpe_30", label: "Rolling Sharpe" },
    ],
  );
}

function makeLineChart(points, accentClass, yLabelFormatter = (v) => fmtNumber(v)) {
  if (!points.length) {
    return "<p>No chart data available.</p>";
  }

  const width = 720;
  const height = 240;
  const padding = 26;
  const values = points.map((point) => Number(point.y));
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const spread = maxY - minY || 1;

  const plotPoints = points
    .map((point, index) => {
      const x = padding + (index / Math.max(points.length - 1, 1)) * (width - padding * 2);
      const y =
        height - padding - ((Number(point.y) - minY) / spread) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(" ");

  const ticks = [0, 0.5, 1].map((ratio) => {
    const y = height - padding - ratio * (height - padding * 2);
    const value = minY + ratio * spread;
    return `
      <line class="chart-grid" x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}"></line>
      <text class="chart-label" x="${padding}" y="${y - 6}">${yLabelFormatter(value)}</text>
    `;
  });

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Line chart">
      ${ticks.join("")}
      <line class="chart-axis" x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}"></line>
      <polyline class="chart-line ${accentClass}" points="${plotPoints}"></polyline>
    </svg>
  `;
}

function renderCharts(summary) {
  document.getElementById("capital-chart").innerHTML = makeLineChart(
    summary.charts.capital_curve,
    "accent",
    (value) => `$${fmtNumber(value, 0)}`,
  );
  document.getElementById("sharpe-chart").innerHTML = makeLineChart(
    summary.charts.rolling_sharpe,
    "teal",
    (value) => fmtNumber(value, 2),
  );
  document.getElementById("return-chart").innerHTML = makeLineChart(
    summary.charts.rolling_avg_return,
    "accent",
    (value) => fmtPct(value),
  );
}

function renderMonitoring(summary) {
  const monitoring = summary.monitoring;
  const statusPill = document.getElementById("status-pill");
  statusPill.textContent = monitoring.is_on_track ? "On Track" : "Degraded";
  statusPill.classList.add(monitoring.is_on_track ? "good" : "bad");

  document.getElementById("monitoring-cards").innerHTML = [
    stackedStat("Validation Sharpe Floor", fmtNumber(monitoring.validation_sharpe_floor, 2)),
    stackedStat("Validation Return Floor", fmtPct(monitoring.validation_expected_return_floor)),
    stackedStat("Validation Win-rate Floor", fmtPct(monitoring.validation_win_rate_floor)),
    stackedStat("Latest Test Rolling Sharpe", fmtNumber(monitoring.latest_test_rolling_sharpe, 2)),
    stackedStat("Latest Test Rolling Return", fmtPct(monitoring.latest_test_rolling_expected_return)),
    stackedStat("Latest Test Rolling Win-rate", fmtPct(monitoring.latest_test_rolling_win_rate)),
  ].join("");

  document.getElementById("on-track-summary").innerHTML = `
    <strong>Status: ${monitoring.is_on_track ? "Within benchmark" : "Below benchmark"}</strong>
    <p>
      We monitor rolling Sharpe, expected return per trade, and win rate over
      ${monitoring.benchmark_window_trades} closed trades. The current test
      window is ${monitoring.is_on_track ? "inside" : "outside"} the
      validation range used to calibrate the strategy.
    </p>
  `;
}

function renderHero(summary) {
  const config = summary.config;
  document.getElementById("hero-theta").textContent = fmtNumber(config.theta, 3);
  document.getElementById("hero-exits").textContent = `${fmtPct(config.profit_take, 1)} / ${fmtPct(config.stop_loss, 1)}`;
  document.getElementById("hero-timeout").textContent = `${config.timeout_seconds}s`;
  document.getElementById("hero-trades").textContent = fmtNumber(summary.test_summary.trade_count, 0);

  document.getElementById("theta-formula").textContent = `α > ${fmtNumber(config.theta, 3)}`;
  document.getElementById("theta-formula-down").textContent = `α < -${fmtNumber(config.theta, 3)}`;
  document.getElementById("profit-target").textContent = fmtPct(config.profit_take, 1);
  document.getElementById("stop-loss").textContent = fmtPct(config.stop_loss, 1);
  document.getElementById("timeout-seconds").textContent = `${config.timeout_seconds}s`;

  document.getElementById("entry-parameter-text").textContent =
    `We chose θ = ${fmtNumber(config.theta, 3)} because it was the best fee-adjusted validation threshold inside the 0.03–0.05 range suggested by the strategy brief. It is high enough to ignore small quote noise, but still large enough to catch dislocations that can move before the 30-second timeout.`;
}

async function main() {
  try {
    const summary = await loadSummary();
    renderHero(summary);
    renderMetrics("validation-metrics", summary.validation_summary);
    renderMetrics("test-metrics", summary.test_summary);
    renderExitBars(summary.test_summary);
    renderDataOverview(summary.data_overview);
    renderCharts(summary);
    renderMonitoring(summary);
    renderTables(summary);
  } catch (error) {
    document.body.innerHTML = `<pre style="padding:24px">Failed to load site data.\n${error.message}</pre>`;
  }
}

main();
