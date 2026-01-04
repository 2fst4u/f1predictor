from f1pred.report import HTML_TEMPLATE
from jinja2 import Template
import re

def test_team_logo_logic():
    template = Template(HTML_TEMPLATE)
    
    test_teams = [
        "Oracle Red Bull Racing",
        "Red Bull",
        "Mercedes-AMG Petronas F1 Team",
        "Mercedes",
        "Scuderia Ferrari",
        "Ferrari HP",
        "McLaren F1 Team",
        "Aston Martin Aramco",
        "BWT Alpine F1 Team",
        "Williams Racing",
        "Visa Cash App RB F1 Team",
        "MoneyGram Haas F1 Team",
        "Stake F1 Team Kick Sauber"
    ]
    
    # Mock data structure matching what the template expects
    sessions_data = [{
        "session_title": "Test Session",
        "rows": [
            {
                "rank": 1,
                "name": "Driver",
                "code": "DRV",
                "team": team_name,
                "mean_pos": 1.0,
                "p_top3": 1.0,
                "p_win": 1.0,
                "p_dnf": 0.0,
                "delta": 0,
            } for team_name in test_teams
        ]
    }]
    
    html = template.render(
        title="Test Report",
        subtitle="Logo Verification",
        sessions=sessions_data,
        generated_at="2025-01-01",
        model_version="test"
    )
    
    # Check if logos are present for each team
    print("Verifying team logos in generated HTML...")
    failed = False
    for team in test_teams:
        # We look for the img tag with the specific team name in alt attribute
        # <img class="team-logo" src="..." alt="Oracle Red Bull Racing" ...>
        # The regex looks for: src="http..." ... alt="TEAM_NAME"
        # Note: In the template, src comes before alt.
        
        # Simple check: find the substring for the team rows
        # We can search for the specific team name and check if there's an img tag preceding it with a valid src
        
        # Or parse the HTML? Regex is probably fine for this verification.
        # Pattern: <img class="team-logo" src="([^"]+)" alt="{team}"
        
        pattern = f'<img class="team-logo" src="([^"]+)" alt="{re.escape(team)}"'
        match = re.search(pattern, html)
        
        if match:
            url = match.group(1)
            print(f"[PASS] {team} -> {url[:50]}...")
        else:
            print(f"[FAIL] {team} -> No logo found")
            failed = True
            
    if failed:
        print("\nSome logo checks failed!")
        exit(1)
    else:
        print("\nAll logo checks passed!")
        exit(0)

if __name__ == "__main__":
    test_team_logo_logic()
