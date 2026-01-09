from typing import List, Dict, Any
import jinja2

# Secure environment with autoescape enabled
ENV = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    autoescape=jinja2.select_autoescape(["html", "xml"])
)

TEAM_LOGOS = {
    "Oracle Red Bull Racing": "https://media.formula1.com/content/dam/fom-website/teams/2024/red-bull-racing-logo.png.transform/2col/image.png",
    "Red Bull": "https://media.formula1.com/content/dam/fom-website/teams/2024/red-bull-racing-logo.png.transform/2col/image.png",
    "Mercedes-AMG Petronas F1 Team": "https://media.formula1.com/content/dam/fom-website/teams/2024/mercedes-logo.png.transform/2col/image.png",
    "Mercedes": "https://media.formula1.com/content/dam/fom-website/teams/2024/mercedes-logo.png.transform/2col/image.png",
    "Scuderia Ferrari": "https://media.formula1.com/content/dam/fom-website/teams/2024/ferrari-logo.png.transform/2col/image.png",
    "Ferrari HP": "https://media.formula1.com/content/dam/fom-website/teams/2024/ferrari-logo.png.transform/2col/image.png",
    "McLaren F1 Team": "https://media.formula1.com/content/dam/fom-website/teams/2024/mclaren-logo.png.transform/2col/image.png",
    "Aston Martin Aramco": "https://media.formula1.com/content/dam/fom-website/teams/2024/aston-martin-logo.png.transform/2col/image.png",
    "BWT Alpine F1 Team": "https://media.formula1.com/content/dam/fom-website/teams/2024/alpine-logo.png.transform/2col/image.png",
    "Williams Racing": "https://media.formula1.com/content/dam/fom-website/teams/2024/williams-logo.png.transform/2col/image.png",
    "Visa Cash App RB F1 Team": "https://media.formula1.com/content/dam/fom-website/teams/2024/rb-logo.png.transform/2col/image.png",
    "MoneyGram Haas F1 Team": "https://media.formula1.com/content/dam/fom-website/teams/2024/haas-f1-team-logo.png.transform/2col/image.png",
    "Stake F1 Team Kick Sauber": "https://media.formula1.com/content/dam/fom-website/teams/2024/kick-sauber-logo.png.transform/2col/image.png",
}

def get_team_logo(team_name: str) -> str:
    """Return logo URL for team name, or a default."""
    return TEAM_LOGOS.get(team_name, "https://media.formula1.com/content/dam/fom-website/teams/2024/f1-logo.png.transform/2col/image.png")

ENV.globals["get_team_logo"] = get_team_logo

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body { font-family: sans-serif; }
        .team-logo { height: 20px; vertical-align: middle; margin-right: 5px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <h2>{{ subtitle }}</h2>
    <p>Generated at: {{ generated_at }} | Model: {{ model_version }}</p>

    {% for session in sessions %}
    <h3>{{ session.session_title }}</h3>
    <table>
        <tr>
            <th>Rank</th>
            <th>Driver</th>
            <th>Team</th>
            <th title="Probability of Top 3">P(Top3)</th>
            <th title="Probability of Win">P(Win)</th>
        </tr>
        {% for row in session.rows %}
        <tr>
            <td>{{ row.rank }}</td>
            <td>{{ row.name }}</td>
            <td>
                <img class="team-logo" src="{{ get_team_logo(row.team) }}" alt="{{ row.team }}">
                {{ row.team }}
            </td>
            <td>{{ "%.1f"|format(row.p_top3 * 100) }}%</td>
            <td>{{ "%.1f"|format(row.p_win * 100) }}%</td>
        </tr>
        {% endfor %}
    </table>
    {% endfor %}
</body>
</html>
"""

def generate_report(
    title: str,
    subtitle: str,
    sessions: List[Dict[str, Any]],
    generated_at: str,
    model_version: str
) -> str:
    """
    Generate HTML report securely using autoescaping.
    """
    template = ENV.from_string(HTML_TEMPLATE)
    return template.render(
        title=title,
        subtitle=subtitle,
        sessions=sessions,
        generated_at=generated_at,
        model_version=model_version
    )
