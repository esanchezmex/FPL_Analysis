# FPL Analysis Dashboard 🏆📊

A Python project for Fantasy Premier League analysis, featuring a web scraper for FBref data and an interactive Streamlit dashboard.

## Features

### 🕷️ FBref Scraper
- **Rate Limited**: Respects FBref's 10 requests/minute limit
- **Anti-Bot Bypass**: Uses `undetected-chromedriver` for reliable scraping
- **Complete Data**: Player match logs with xG, xA, shots, and more
- **FPL Ready**: Gameweek filtering for Fantasy Premier League analysis

### 📊 Analytics Dashboard
- **Team Analysis**: Team-level xG/xGA metrics and trends
- **Player Analysis**: Individual player performance with per-90 stats
- **Player Comparison**: Compare multiple players side-by-side
- **Interactive Filters**: Position, gameweek range, team, and venue filters

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 🖥️ Running the Dashboard

```bash
streamlit run dashboard.py
```

This will open the dashboard in your browser at `http://localhost:8501`.

> **Note:** The dashboard requires data files to exist. See [Populating Data](#-populating-data) below.

### 📥 Populating Data

The data files are not included in the repository. You must scrape the data before using the dashboard.

#### Player Data (for Dashboard)

```bash
python fbref_scraper.py --players
```

This will:
1. Scrape all Premier League player match logs
2. Save `player_match_logs.csv` (all competitions)
3. Save `player_match_logs_pl.csv` (Premier League only - used by dashboard)

⚠️ **Warning**: This takes **30-60+ minutes** due to rate limiting (hundreds of players × 6 seconds each).

#### Team Data (optional)

```bash
python fbref_scraper.py
```

This will:
1. Scrape all 20 Premier League team match logs
2. Save `premier_league_matches.csv` (all competitions)
3. Save `premier_league_only.csv` (Premier League only)

⏱️ **Runtime**: ~5 minutes

### 🔄 Refreshing Data

To update data with the latest matches, simply re-run the scraper:

```bash
# Full refresh of player data
python fbref_scraper.py --players
```

## Project Structure

```
FPL_Analysis/
├── fbref_scraper.py      # 🕷️ FBref web scraper
├── fpl_analysis.py       # 📊 Analysis engine (FPLAnalyzer class)
├── dashboard.py          # 🖥️ Streamlit dashboard
├── requirements.txt      # 📦 Python dependencies
└── README.md             # 📖 This file
```

## Data Files (Generated)

| File | Description |
|------|-------------|
| `player_match_logs.csv` | Player match data (all competitions) |
| `player_match_logs_pl.csv` | Player match data (Premier League only) |
| `premier_league_matches.csv` | Team match data (all competitions) |
| `premier_league_only.csv` | Team match data (Premier League only) |


## Rate Limiting

FBref's policy:
- Maximum 10 requests per minute
- Violations result in up to 24-hour bans

The scraper:
- Waits 6.5+ seconds between requests
- Adds random jitter (0.5-2.5s) to appear human
- Uses `undetected-chromedriver` to avoid detection

## Programmatic Usage

```python
from fpl_analysis import FPLAnalyzer

# Load data
analyzer = FPLAnalyzer('player_match_logs_pl.csv')

# Get aggregated stats (totals)
df = analyzer.fpl_data(
    min_gameweek=5,       # After GW5
    max_gameweek=15,      # Up to GW15
    position='mid',       # Midfielders only
    sort_by='xGI'
)

# Get per-90 stats
df_p90 = analyzer.fpl_data_p90(
    min_gameweek=5,
    position='att',
    min_minutes_pct=0.75  # At least 75% of possible minutes
)
```

## Legal Note

This scraper is for personal use only. Please respect FBref's terms of service and their rate limiting requirements. Data from FBref is sourced from Opta.
