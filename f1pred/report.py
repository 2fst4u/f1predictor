from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import webbrowser

import pandas as pd
from jinja2 import Template

from .util import get_logger

logger = get_logger(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>F1 Predictions - {{ title }}</title>
<style>
 body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial; margin: 1rem 2rem; background: #fff; color: #111; }
 h1, h2 { margin: 0.3rem 0; }
 table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
 th, td { padding: 0.5rem; border-bottom: 1px solid #ddd; text-align: left; }
 tr:hover { background: #f7f7f7; }
 .small { color: #666; font-size: 0.9rem; }
 .pos { width: 2rem; text-align: right; }
 .delta { width: 2rem; font-weight: bold; }
 .up { color: {{ color_up }}; }
 .down { color: {{ color_down }}; }
 .neutral { color: {{ color_neutral }}; }
 .badge { display: inline-block; padding: 0.2rem 0.4rem; border-radius: 0.25rem; background: #eee; font-size: 0.8rem; }
</style>
</head>
<body>
<h1>F1 Predictions</h1>
<div class="small">{{ subtitle }}</div>

{% for sess in sessions %}
  <h2>{{ sess.session_title }}</h2>
  <table>
    <thead>
      <tr>
        <th class="pos">#</th>
        <th>Driver</th>
        <th>Team</th>
        <th>Pred pos (μ)</th>
        <th>Top3 %</th>
        <th>Win %</th>
        <th>DNF %</th>
        <th>Change</th>
      </tr>
    </thead>
    <tbody>
      {% for row in sess.rows %}
      <tr>
        <td class="pos">{{ row.rank }}</td>
        <td>{{ row.name }} <span class="badge">{{ row.code }}</span></td>
        <td>{{ row.team }}</td>
        <td>{{ "%.2f"|format(row.mean_pos) }}</td>
        <td>{{ "%.1f"|format(row.p_top3 * 100) }}</td>
        <td>{{ "%.1f"|format(row.p_win * 100) }}</td>
        <td>{{ "%.1f"|format(row.p_dnf * 100) }}</td>
        <td class="delta">
          {% if row.delta is not none %}
            {% if row.delta < 0 %}<span class="up">↑ {{ -row.delta }}</span>
            {% elif row.delta > 0 %}<span class="down">↓ {{ row.delta }}</span>
            {% else %}<span class="neutral">·</span>{% endif %}
          {% else %}
            <span class="neutral">·</span>
          {% endif %}
        </td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
{% endfor %}

<footer class="small">Generated at {{ generated_at }} | Model {{ model_version }}</footer>
</body>
</html>
"""

BACKTEST_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>F1 Backtest Summary</title>
<style>
 body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, "Helvetica Neue", Arial; margin: 1rem 2rem; background: #fff; color: #111; }
 h1, h2 { margin: 0.3rem 0; }
 table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
 th, td { padding: 0.5rem; border-bottom: 1px solid #ddd; text-align: left; }
 tr:hover { background: #f7f7f7; }
 .small { color: #666; font-size: 0.9rem; }
</style>
</head>
<body>
<h1>Backtest Summary</h1>
<div class="small">Generated at {{ generated_at }} | Model {{ model_version }}</div>

<h2>Aggregates by session</h2>
<table>
<thead>
  <tr><th>Session</th><th>N events</th><th>Spearman</th><th>Kendall</th><th>Top3 Acc</th><th>Brier (pairwise)</th><th>CRPS</th></tr>
</thead>
<tbody>
{% for row in aggregates %}
  <tr>
    <td>{{ row.event }}</td>
    <td>{{ row.n_events }}</td>
    <td>{{ "%.3f"|format(row.spearman) if row.spearman==row.spearman else "—" }}</td>
    <td>{{ "%.3f"|format(row.kendall) if row.kendall==row.kendall else "—" }}</td>
    <td>{{ "%.3f"|format(row.accuracy_top3) if row.accuracy_top3==row.accuracy_top3 else "—" }}</td>
    <td>{{ "%.3f"|format(row.brier_pairwise) if row.brier_pairwise==row.brier_pairwise else "—" }}</td>
    <td>{{ "%.3f"|format(row.crps) if row.crps==row.crps else "—" }}</td>
  </tr>
{% endfor %}
</tbody>
</table>

<h2>Recent events</h2>
<table>
<thead>
  <tr><th>Season</th><th>Round</th><th>Session</th><th>Spearman</th><th>Kendall</th><th>Top3 Acc</th><th>Brier (pairwise)</th><th>CRPS</th></tr>
</thead>
<tbody>
{% for row in recent %}
  <tr>
    <td>{{ row.season }}</td>
    <td>{{ row.round }}</td>
    <td>{{ row.event }}</td>
    <td>{{ "%.3f"|format(row.spearman) if row.spearman==row.spearman else "—" }}</td>
    <td>{{ "%.3f"|format(row.kendall) if row.kendall==row.kendall else "—" }}</td>
    <td>{{ "%.3f"|format(row.accuracy_top3) if row.accuracy_top3==row.accuracy_top3 else "—" }}</td>
    <td>{{ "%.3f"|format(row.brier_pairwise) if row.brier_pairwise==row.brier_pairwise else "—" }}</td>
    <td>{{ "%.3f"|format(row.crps) if row.crps==row.crps else "—" }}</td>
  </tr>
{% endfor %}
</tbody>
</table>

</body>
</html>
"""

def generate_html_report(path: str, title: str, subtitle: str, sessions_data: List[Dict[str, Any]],
                         cfg, open_browser: bool = False) -> None:
    template = Template(HTML_TEMPLATE)
    html = template.render(
        title=title,
        subtitle=subtitle,
        sessions=sessions_data,
        generated_at=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        model_version=cfg.app.model_version,
        color_up=cfg.output.color_up,
        color_down=cfg.output.color_down,
        color_neutral=cfg.output.color_neutral,
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding="utf-8")
    logger.info(f"HTML report written to {path}")
    if open_browser:
        try:
            webbrowser.open_new_tab(f"file://{Path(path).resolve()}")
        except Exception:
            pass

def generate_backtest_summary_html(metrics_csv: str, out_path: str, cfg) -> None:
    df = pd.read_csv(metrics_csv) if Path(metrics_csv).exists() else pd.DataFrame()
    if df.empty:
        logger.warning("No backtest metrics available to build report.")
        return
    agg = df.groupby("event").agg(
        n_events=("season", "count"),
        spearman=("spearman", "mean"),
        kendall=("kendall", "mean"),
        accuracy_top3=("accuracy_top3", "mean"),
        brier_pairwise=("brier_pairwise", "mean"),
        crps=("crps", "mean"),
    ).reset_index()
    recent = df.sort_values(["season", "round"], ascending=[False, False]).head(50).to_dict(orient="records")

    template = Template(BACKTEST_TEMPLATE)
    html = template.render(
        generated_at=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        model_version=cfg.app.model_version,
        aggregates=agg.to_dict(orient="records"),
        recent=recent,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    logger.info(f"Backtest HTML summary written to {out_path}")