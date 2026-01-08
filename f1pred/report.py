from __future__ import annotations
from typing import Dict, Any, List
from pathlib import Path
import webbrowser

import pandas as pd
from jinja2 import Environment, BaseLoader

from .util import get_logger

logger = get_logger(__name__)

HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>F1 Predictions - {{ title }}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
<style>
 :root {
   --bg-primary: #0a0e27;
   --bg-secondary: #151934;
   --bg-card: #1a1f3a;
   --text-primary: #e4e7f1;
   --text-secondary: #9ca3bc;
   --text-muted: #6b7280;
   --accent-red: #e10600;
   --accent-red-light: #ff1e1e;
   --accent-blue: #0090ff;
   --border-color: #2a2f4a;
   --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
   --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
   --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
   --gradient-header: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a1a1f 100%);
   --gradient-card: linear-gradient(145deg, rgba(26, 31, 58, 0.6), rgba(21, 25, 52, 0.4));
 }
 
 * { box-sizing: border-box; margin: 0; padding: 0; }
 
 body {
   font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
   background: var(--bg-primary);
   color: var(--text-primary);
   line-height: 1.6;
   min-height: 100vh;
 }
 
 .container {
   max-width: 1400px;
   margin: 0 auto;
   padding: 2rem;
 }
 
 .hero-header {
   background: var(--gradient-header);
   border-radius: 16px;
   padding: 3rem 2.5rem;
   margin-bottom: 2.5rem;
   box-shadow: var(--shadow-lg);
   border: 1px solid var(--border-color);
   position: relative;
   overflow: hidden;
 }
 
 .hero-header::before {
   content: '';
   position: absolute;
   top: 0;
   left: 0;
   right: 0;
   height: 4px;
   background: linear-gradient(90deg, var(--accent-red) 0%, var(--accent-red-light) 50%, var(--accent-blue) 100%);
 }
 
 h1 {
   font-family: 'Inter', sans-serif;
   font-size: 3rem;
   font-weight: 800;
   margin-bottom: 0.5rem;
   background: linear-gradient(135deg, #ffffff 0%, #e4e7f1 50%, var(--accent-red-light) 100%);
   -webkit-background-clip: text;
   -webkit-text-fill-color: transparent;
   background-clip: text;
   letter-spacing: -0.02em;
 }
 
 .subtitle {
   font-size: 1.25rem;
   color: var(--text-secondary);
   font-weight: 500;
 }
 
 .session-section {
   margin-bottom: 3rem;
 }
 
 h2 {
   font-family: 'Inter', sans-serif;
   font-size: 1.75rem;
   font-weight: 700;
   margin-bottom: 1.5rem;
   color: var(--text-primary);
   padding-bottom: 0.75rem;
   border-bottom: 2px solid var(--accent-red);
   display: inline-block;
 }
 
 .table-card {
   background: var(--gradient-card);
   backdrop-filter: blur(10px);
   border-radius: 12px;
   overflow: hidden;
   box-shadow: var(--shadow-md);
   border: 1px solid var(--border-color);
 }
 
 table {
   border-collapse: collapse;
   width: 100%;
   font-size: 0.95rem;
 }
 
 thead {
   background: rgba(225, 6, 0, 0.1);
   position: sticky;
   top: 0;
   z-index: 10;
 }
 
 th {
   padding: 1rem 0.75rem;
   text-align: left;
   font-family: 'Inter', sans-serif;
   font-weight: 600;
   font-size: 0.85rem;
   text-transform: uppercase;
   letter-spacing: 0.05em;
   color: var(--text-secondary);
   border-bottom: 2px solid var(--border-color);
 }

 th[title] {
   cursor: help;
   text-decoration: underline dotted;
   text-decoration-color: var(--text-muted);
 }
 
 td {
   padding: 1rem 0.75rem;
   border-bottom: 1px solid rgba(42, 47, 74, 0.5);
 }
 
 tbody tr {
   transition: all 0.2s ease;
 }
 
 tbody tr:hover {
   background: rgba(225, 6, 0, 0.08);
   transform: translateX(4px);
 }
 
 tbody tr:last-child td {
   border-bottom: none;
 }
 
 .pos {
   width: 3rem;
   text-align: center;
   font-weight: 700;
   font-family: 'Inter', sans-serif;
   font-size: 1.1rem;
   color: var(--accent-red-light);
 }
 
 .driver-cell {
   font-weight: 500;
   min-width: 200px;
 }
 
 .badge {
   display: inline-block;
   padding: 0.25rem 0.5rem;
   margin-left: 0.5rem;
   border-radius: 6px;
   background: rgba(225, 6, 0, 0.2);
   border: 1px solid rgba(225, 6, 0, 0.3);
   font-size: 0.75rem;
   font-weight: 600;
   font-family: 'Inter', sans-serif;
   letter-spacing: 0.05em;
   color: var(--accent-red-light);
 }
 
 .team-cell {
   color: var(--text-secondary);
   display: flex;
   align-items: center;
   gap: 0.5rem;
 }
 
 .team-logo {
   height: 20px;
   width: auto;
   max-width: 50px;
   vertical-align: middle;
   filter: brightness(1.2) contrast(1.1);
   object-fit: contain;
 }
 
 .team-name {
   flex: 1;
 }
 
 .prob-cell {
   position: relative;
   min-width: 80px;
 }
 
 .prob-value {
   position: relative;
   z-index: 1;
   font-weight: 500;
   font-family: 'Inter', sans-serif;
 }
 
 .prob-bar {
   position: absolute;
   left: 0;
   top: 50%;
   transform: translateY(-50%);
   height: 70%;
   background: linear-gradient(90deg, rgba(225, 6, 0, 0.3), rgba(225, 6, 0, 0.1));
   border-radius: 4px;
   z-index: 0;
   transition: width 0.3s ease;
 }
 
 .delta {
   width: 4rem;
   text-align: center;
   font-weight: 700;
   font-family: 'Inter', sans-serif;
 }
 
 .up {
   color: #10b981;
   filter: drop-shadow(0 0 8px rgba(16, 185, 129, 0.4));
 }
 
 .down {
   color: #ef4444;
   filter: drop-shadow(0 0 8px rgba(239, 68, 68, 0.4));
 }
 
 .neutral {
   color: var(--text-muted);
 }
 
 footer {
   margin-top: 4rem;
   padding: 2rem;
   text-align: center;
   color: var(--text-muted);
   font-size: 0.9rem;
   border-top: 1px solid var(--border-color);
 }
 
 @media (max-width: 1024px) {
   h1 { font-size: 2.5rem; }
   .hero-header { padding: 2rem 1.5rem; }
   .container { padding: 1.5rem; }
   table { font-size: 0.9rem; }
   th, td { padding: 0.75rem 0.5rem; }
 }
 
 @media (max-width: 768px) {
   h1 { font-size: 2rem; }
   .subtitle { font-size: 1rem; }
   h2 { font-size: 1.5rem; }
   .container { padding: 1rem; }
   .hero-header { padding: 1.5rem 1rem; }
   table { font-size: 0.85rem; }
   th, td { padding: 0.6rem 0.4rem; }
   .badge { font-size: 0.7rem; padding: 0.2rem 0.4rem; }
 }
</style>
</head>
<body>
<div class="container">
  <div class="hero-header">
    <h1>F1 Predictions</h1>
    <div class="subtitle">{{ subtitle }}</div>
  </div>

  {% for sess in sessions %}
  <div class="session-section">
    <h2>{{ sess.session_title }}</h2>
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th class="pos" title="Predicted finishing rank">#</th>
            <th>Driver</th>
            <th>Team</th>
            <th title="Mean predicted finishing position across 5000 simulations">Pred pos (μ)</th>
            <th title="Probability of finishing on the podium (1st, 2nd, or 3rd)">Top3 %</th>
            <th title="Probability of winning the race">Win %</th>
            <th title="Estimated probability of Did Not Finish (Retirement)">DNF %</th>
            <th title="Difference between actual and predicted position (Green = improvement)">Change</th>
          </tr>
        </thead>
        <tbody>
          {% for row in sess.rows %}
          <tr>
            <td class="pos">{{ row.rank }}</td>
            <td class="driver-cell">{{ row.name }} <span class="badge">{{ row.code }}</span></td>
            <td class="team-cell">
              {% set logo_url = '' %}
              {% set t = row.team|lower %}
              {% if 'red bull' in t or 'rb' in t or 'alpha' in t or 'toro' in t %}
                  {% if 'visa' in t or 'app' in t or 'rb f1' in t or 'alpha' in t or 'toro' in t %}
                      {% set logo_url = 'https://www.seeklogo.com/images/V/visa-cash-app-rb-formula-one-team-logo-FA0F966B2F-seeklogo.com.png' %}
                  {% else %}
                      {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Logo_of_Red_bull.svg/120px-Logo_of_Red_bull.svg.png' %}
                  {% endif %}
              {% endif %}
              
              {% if not logo_url %}
                {% if 'mercedes' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Mercedes_AMG_Petronas_F1_Logo.svg/120px-Mercedes_AMG_Petronas_F1_Logo.svg.png' %}
                {% elif 'ferrari' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Scuderia_Ferrari_Logo.svg/80px-Scuderia_Ferrari_Logo.svg.png' %}
                {% elif 'mclaren' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/e/e6/McLaren_Racing_logo.png' %}
                {% elif 'aston' in t and 'martin' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/0/05/Aston_Martin_F1_Team_logo_2024.jpg' %}
                {% elif 'alpine' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Logo_BWT_Alpine_F1_Team_-_2022.svg/120px-Logo_BWT_Alpine_F1_Team_-_2022.svg.png' %}
                {% elif 'williams' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Williams_Racing_2022_logo.svg/120px-Williams_Racing_2022_logo.svg.png' %}
                {% elif 'haas' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/MoneyGram_Haas_F1_Team_Logo.svg/120px-MoneyGram_Haas_F1_Team_Logo.svg.png' %}
                {% elif 'sauber' in t or 'stake' in t or 'alfa' in t %}
                    {% set logo_url = 'https://upload.wikimedia.org/wikipedia/commons/8/87/Logo_of_Stake_F1_Team_Kick_Sauber.png' %}
                {% endif %}
              {% endif %}
              {% if logo_url %}<img class="team-logo" src="{{ logo_url }}" alt="{{ row.team }}" onerror="this.style.display='none'">{% endif %}
              <span class="team-name">{{ row.team }}</span>
            </td>
            <td><span class="prob-value">{{ "%.2f"|format(row.mean_pos) }}</span></td>
            <td class="prob-cell">
              <div class="prob-bar" style="width: {{ row.p_top3 * 100 }}%;"></div>
              <span class="prob-value">{{ "%.1f"|format(row.p_top3 * 100) }}</span>
            </td>
            <td class="prob-cell">
              <div class="prob-bar" style="width: {{ row.p_win * 100 }}%;"></div>
              <span class="prob-value">{{ "%.1f"|format(row.p_win * 100) }}</span>
            </td>
            <td class="prob-cell">
              <div class="prob-bar" style="width: {{ row.p_dnf * 100 }}%; background: linear-gradient(90deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.1));"></div>
              <span class="prob-value">{{ "%.1f"|format(row.p_dnf * 100) }}</span>
            </td>
            <td class="delta">
              {% if row.delta is not none %}
                {% if row.delta < 0 %}<span class="up" aria-label="Gained {{ -row.delta }} positions">↑{{ -row.delta }}</span>
                {% elif row.delta > 0 %}<span class="down" aria-label="Lost {{ row.delta }} positions">↓{{ row.delta }}</span>
                {% else %}<span class="neutral" aria-label="No change">·</span>{% endif %}
              {% else %}
                <span class="neutral" aria-label="No data">·</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  {% endfor %}

  <footer>Generated at {{ generated_at }} | Model {{ model_version }}</footer>
</div>

<script>
// Map team names to publicly available logo URLs from Wikipedia Commons
function getTeamLogo(teamName) {
  if (!teamName) return '';
  const t = teamName.toLowerCase();
  
  // Specific checks first
  if (t.includes('visa') || t.includes('app') || t.includes('rb f1') || t.includes('alphatauri') || t.includes('toro rosso')) {
      return 'https://www.seeklogo.com/images/V/visa-cash-app-rb-formula-one-team-logo-FA0F966B2F-seeklogo.com.png';
  }
  
  // Red Bull (main)
  if (t.includes('red bull')) {
       return 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Logo_of_Red_bull.svg/120px-Logo_of_Red_bull.svg.png';
  }
  
  if (t.includes('mercedes')) return 'https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Mercedes_AMG_Petronas_F1_Logo.svg/120px-Mercedes_AMG_Petronas_F1_Logo.svg.png';
  if (t.includes('ferrari')) return 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Scuderia_Ferrari_Logo.svg/80px-Scuderia_Ferrari_Logo.svg.png';
  if (t.includes('mclaren')) return 'https://upload.wikimedia.org/wikipedia/commons/e/e6/McLaren_Racing_logo.png';
  if (t.includes('aston') && t.includes('martin')) return 'https://upload.wikimedia.org/wikipedia/commons/0/05/Aston_Martin_F1_Team_logo_2024.jpg';
  if (t.includes('alpine')) return 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Logo_BWT_Alpine_F1_Team_-_2022.svg/120px-Logo_BWT_Alpine_F1_Team_-_2022.svg.png';
  if (t.includes('williams')) return 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Williams_Racing_2022_logo.svg/120px-Williams_Racing_2022_logo.svg.png';
  if (t.includes('haas')) return 'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/MoneyGram_Haas_F1_Team_Logo.svg/120px-MoneyGram_Haas_F1_Team_Logo.svg.png';
  if (t.includes('sauber') || t.includes('stake') || t.includes('alfa rom')) return 'https://upload.wikimedia.org/wikipedia/commons/8/87/Logo_of_Stake_F1_Team_Kick_Sauber.png';
  
  return '';
}

// Populate logo URLs on page load
document.addEventListener('DOMContentLoaded', function() {
  const teamLogos = document.querySelectorAll('.team-logo');
  teamLogos.forEach(function(img) {
    const teamName = img.getAttribute('data-team');
    const logoUrl = getTeamLogo(teamName);
    if (logoUrl) {
      img.src = logoUrl;
    } else {
      img.style.display = 'none';
    }
  });
});
</script>
</body>
</html>
"""

BACKTEST_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>F1 Backtest Summary</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Roboto:wght@400;500&display=swap" rel="stylesheet">
<style>
 :root {
   --bg-primary: #0a0e27;
   --bg-secondary: #151934;
   --bg-card: #1a1f3a;
   --text-primary: #e4e7f1;
   --text-secondary: #9ca3bc;
   --text-muted: #6b7280;
   --accent-red: #e10600;
   --accent-red-light: #ff1e1e;
   --accent-blue: #0090ff;
   --border-color: #2a2f4a;
   --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.3);
   --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.4), 0 2px 4px -1px rgba(0, 0, 0, 0.3);
   --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.5), 0 4px 6px -2px rgba(0, 0, 0, 0.3);
   --gradient-header: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #2a1a1f 100%);
   --gradient-card: linear-gradient(145deg, rgba(26, 31, 58, 0.6), rgba(21, 25, 52, 0.4));
 }
 
 * { box-sizing: border-box; margin: 0; padding: 0; }
 
 body {
   font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
   background: var(--bg-primary);
   color: var(--text-primary);
   line-height: 1.6;
   min-height: 100vh;
 }
 
 .container {
   max-width: 1400px;
   margin: 0 auto;
   padding: 2rem;
 }
 
 .hero-header {
   background: var(--gradient-header);
   border-radius: 16px;
   padding: 3rem 2.5rem;
   margin-bottom: 2.5rem;
   box-shadow: var(--shadow-lg);
   border: 1px solid var(--border-color);
   position: relative;
   overflow: hidden;
 }
 
 .hero-header::before {
   content: '';
   position: absolute;
   top: 0;
   left: 0;
   right: 0;
   height: 4px;
   background: linear-gradient(90deg, var(--accent-red) 0%, var(--accent-red-light) 50%, var(--accent-blue) 100%);
 }
 
 h1 {
   font-family: 'Inter', sans-serif;
   font-size: 3rem;
   font-weight: 800;
   background: linear-gradient(135deg, #ffffff 0%, #e4e7f1 50%, var(--accent-red-light) 100%);
   -webkit-background-clip: text;
   -webkit-text-fill-color: transparent;
   background-clip: text;
   letter-spacing: -0.02em;
 }
 
 .subtitle {
   font-size: 1rem;
   color: var(--text-secondary);
   font-weight: 500;
   margin-top: 0.5rem;
 }
 
 .section {
   margin-bottom: 3rem;
 }
 
 h2 {
   font-family: 'Inter', sans-serif;
   font-size: 1.75rem;
   font-weight: 700;
   margin-bottom: 1.5rem;
   color: var(--text-primary);
   padding-bottom: 0.75rem;
   border-bottom: 2px solid var(--accent-red);
   display: inline-block;
 }
 
 .table-card {
   background: var(--gradient-card);
   backdrop-filter: blur(10px);
   border-radius: 12px;
   overflow: hidden;
   box-shadow: var(--shadow-md);
   border: 1px solid var(--border-color);
   margin-bottom: 1.5rem;
 }
 
 table {
   border-collapse: collapse;
   width: 100%;
   font-size: 0.95rem;
 }
 
 thead {
   background: rgba(225, 6, 0, 0.1);
   position: sticky;
   top: 0;
   z-index: 10;
 }
 
 th {
   padding: 1rem 0.75rem;
   text-align: left;
   font-family: 'Inter', sans-serif;
   font-weight: 600;
   font-size: 0.85rem;
   text-transform: uppercase;
   letter-spacing: 0.05em;
   color: var(--text-secondary);
   border-bottom: 2px solid var(--border-color);
 }
 
 td {
   padding: 1rem 0.75rem;
   border-bottom: 1px solid rgba(42, 47, 74, 0.5);
 }
 
 tbody tr {
   transition: all 0.2s ease;
 }
 
 tbody tr:hover {
   background: rgba(225, 6, 0, 0.08);
   transform: translateX(4px);
 }
 
 tbody tr:last-child td {
   border-bottom: none;
 }
 
 .metric-value {
   font-family: 'Inter', sans-serif;
   font-weight: 500;
   color: var(--text-primary);
 }
 
 .metric-good {
   color: #10b981;
 }
 
 .metric-warning {
   color: #f59e0b;
 }
 
 footer {
   margin-top: 4rem;
   padding: 2rem;
   text-align: center;
   color: var(--text-muted);
   font-size: 0.9rem;
   border-top: 1px solid var(--border-color);
 }
 
 @media (max-width: 1024px) {
   h1 { font-size: 2.5rem; }
   .hero-header { padding: 2rem 1.5rem; }
   .container { padding: 1.5rem; }
   table { font-size: 0.9rem; }
   th, td { padding: 0.75rem 0.5rem; }
 }
 
 @media (max-width: 768px) {
   h1 { font-size: 2rem; }
   h2 { font-size: 1.5rem; }
   .container { padding: 1rem; }
   .hero-header { padding: 1.5rem 1rem; }
   table { font-size: 0.85rem; }
   th, td { padding: 0.6rem 0.4rem; }
 }
</style>
</head>
<body>
<div class="container">
  <div class="hero-header">
    <h1>Backtest Summary</h1>
    <div class="subtitle">Generated at {{ generated_at }} | Model {{ model_version }}</div>
  </div>

  <div class="section">
    <h2>Aggregates by session</h2>
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Session</th>
            <th>N events</th>
            <th>Spearman</th>
            <th>Kendall</th>
            <th>Top3 Acc</th>
            <th>Brier (pairwise)</th>
            <th>CRPS</th>
          </tr>
        </thead>
        <tbody>
          {% for row in aggregates %}
          <tr>
            <td class="metric-value">{{ row.event }}</td>
            <td class="metric-value">{{ row.n_events }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.spearman) if row.spearman==row.spearman else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.kendall) if row.kendall==row.kendall else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.accuracy_top3) if row.accuracy_top3==row.accuracy_top3 else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.brier_pairwise) if row.brier_pairwise==row.brier_pairwise else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.crps) if row.crps==row.crps else "—" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <div class="section">
    <h2>Recent events</h2>
    <div class="table-card">
      <table>
        <thead>
          <tr>
            <th>Season</th>
            <th>Round</th>
            <th>Session</th>
            <th>Spearman</th>
            <th>Kendall</th>
            <th>Top3 Acc</th>
            <th>Brier (pairwise)</th>
            <th>CRPS</th>
          </tr>
        </thead>
        <tbody>
          {% for row in recent %}
          <tr>
            <td class="metric-value">{{ row.season }}</td>
            <td class="metric-value">{{ row.round }}</td>
            <td class="metric-value">{{ row.event }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.spearman) if row.spearman==row.spearman else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.kendall) if row.kendall==row.kendall else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.accuracy_top3) if row.accuracy_top3==row.accuracy_top3 else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.brier_pairwise) if row.brier_pairwise==row.brier_pairwise else "—" }}</td>
            <td class="metric-value">{{ "%.3f"|format(row.crps) if row.crps==row.crps else "—" }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>

  <footer>F1 Prediction Model Performance Analysis</footer>
</div>
</body>
</html>
"""


def generate_html_report(
    path: str,
    title: str,
    subtitle: str,
    sessions_data: List[Dict[str, Any]],
    cfg,
    open_browser: bool = False,
) -> None:
    # Use Environment with autoescape=True for security
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(HTML_TEMPLATE)
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
    agg = (
        df.groupby("event")
        .agg(
            n_events=("season", "count"),
            spearman=("spearman", "mean"),
            kendall=("kendall", "mean"),
            accuracy_top3=("accuracy_top3", "mean"),
            brier_pairwise=("brier_pairwise", "mean"),
            crps=("crps", "mean"),
        )
        .reset_index()
    )
    recent = (
        df.sort_values(["season", "round"], ascending=[False, False])
        .head(50)
        .to_dict(orient="records")
    )

    # Use Environment with autoescape=True for security
    env = Environment(loader=BaseLoader(), autoescape=True)
    template = env.from_string(BACKTEST_TEMPLATE)
    html = template.render(
        generated_at=pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        model_version=cfg.app.model_version,
        aggregates=agg.to_dict(orient="records"),
        recent=recent,
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    logger.info(f"Backtest HTML summary written to {out_path}")
