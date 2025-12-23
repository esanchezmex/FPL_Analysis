"""
FBref Premier League Match Data Scraper
Uses undetected-chromedriver to bypass anti-bot detection automatically.

Usage:
    pip install undetected-chromedriver pandas beautifulsoup4
    python fbref_scraper.py
"""

import time
import random
import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import sys

# Try to import undetected_chromedriver, fall back to regular selenium
try:
    import undetected_chromedriver as uc
    USE_UNDETECTED = True
    print("Using undetected-chromedriver")
except ImportError:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    USE_UNDETECTED = False
    print("Using regular selenium (may get blocked)")

from bs4 import BeautifulSoup
import pandas as pd

# Configure logging with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Force stdout to flush immediately
import functools
print = functools.partial(print, flush=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

FBREF_BASE_URL = "https://fbref.com"
PREMIER_LEAGUE_URL = f"{FBREF_BASE_URL}/en/comps/9/Premier-League-Stats"

# Rate limiting: 6 seconds between requests to stay under 10/min limit
MIN_DELAY = 6.5
JITTER_RANGE = (0.5, 2.5)


# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Ensures we don't exceed FBref's rate limits."""
    
    def __init__(self):
        self.last_request = 0
    
    def wait(self):
        if self.last_request > 0:
            elapsed = time.time() - self.last_request
            if elapsed < MIN_DELAY:
                sleep_time = MIN_DELAY - elapsed + random.uniform(*JITTER_RANGE)
                logger.info(f"Rate limiting: waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        self.last_request = time.time()


# =============================================================================
# SCRAPER
# =============================================================================

class FBrefScraper:
    """FBref scraper using undetected-chromedriver."""
    
    def __init__(self, headless: bool = False):
        """
        Initialize scraper.
        
        Args:
            headless: If False, shows the browser window (helpful for debugging)
        """
        self.rate_limiter = RateLimiter()
        self.driver = None
        self.headless = headless
        print(f"Initializing scraper (headless={headless})...")
        self._setup_driver()
    
    def _setup_driver(self):
        """Set up the Chrome driver."""
        print("Setting up Chrome browser...")
        
        if USE_UNDETECTED:
            # undetected-chromedriver handles everything automatically
            options = uc.ChromeOptions()
            if self.headless:
                options.add_argument('--headless=new')
            options.add_argument('--window-size=1920,1080')
            
            print("Starting Chrome with undetected-chromedriver...")
            self.driver = uc.Chrome(options=options)
        else:
            # Regular selenium
            options = Options()
            if self.headless:
                options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--window-size=1920,1080')
            
            print("Starting Chrome with regular selenium...")
            self.driver = webdriver.Chrome(options=options)
        
        print("✓ Chrome browser started successfully!")
    
    def _get_page(self, url: str) -> BeautifulSoup:
        """Navigate to URL and return parsed HTML."""
        self.rate_limiter.wait()
        
        logger.info(f"Fetching: {url}")
        self.driver.get(url)
        
        # Wait for page to load
        time.sleep(random.uniform(2, 4))
        
        return BeautifulSoup(self.driver.page_source, 'html.parser')
    
    def get_teams(self) -> List[Dict[str, str]]:
        """Get all Premier League team URLs."""
        print("\nFetching Premier League teams...")
        soup = self._get_page(PREMIER_LEAGUE_URL)
        
        teams = []
        
        # Find standings table
        for table in soup.find_all('table'):
            tbody = table.find('tbody')
            if not tbody:
                continue
            
            for row in tbody.find_all('tr'):
                team_cell = row.find('td', {'data-stat': 'team'})
                if team_cell:
                    link = team_cell.find('a')
                    if link and link.get('href'):
                        name = link.get_text(strip=True)
                        url = FBREF_BASE_URL + link['href']
                        
                        if not any(t['name'] == name for t in teams):
                            teams.append({'name': name, 'url': url})
            
            # Found teams, stop looking at other tables
            if len(teams) >= 20:
                break
        
        print(f"✓ Found {len(teams)} teams")
        for t in teams:
            print(f"  - {t['name']}")
        
        return teams
    
    def get_team_matches(self, team_url: str, team_name: str) -> pd.DataFrame:
        """Get match logs for a team."""
        print(f"\nScraping matches for {team_name}...")
        
        # First get team page to find match logs link
        soup = self._get_page(team_url)
        
        # Find scores/fixtures link
        scores_link = None
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'matchlogs' in href and 'schedule' in href:
                scores_link = FBREF_BASE_URL + href
                break
        
        if not scores_link:
            # Construct URL
            match = re.search(r'/squads/([^/]+)/', team_url)
            if match:
                team_id = match.group(1)
                scores_link = f"{FBREF_BASE_URL}/en/squads/{team_id}/2025-2026/matchlogs/all_comps/schedule/"
        
        if scores_link:
            soup = self._get_page(scores_link)
        
        # Parse match table - look for table with matchlogs in id
        matches = []
        match_table = None
        
        # Find the right table
        for table in soup.find_all('table'):
            table_id = table.get('id', '')
            # Match logs table typically has 'matchlogs' in the id
            if 'matchlogs' in table_id.lower():
                match_table = table
                print(f"  Found table: {table_id}")
                break
        
        # If no matchlogs table, look for any table with match data
        if not match_table:
            for table in soup.find_all('table'):
                # Look for cells with date-related data-stat
                if table.find(['th', 'td'], {'data-stat': 'date'}):
                    match_table = table
                    break
        
        if not match_table:
            print(f"  ✗ No match table found")
            # Debug: show what tables exist
            all_tables = soup.find_all('table')
            print(f"  Found {len(all_tables)} tables on page")
            for t in all_tables[:5]:
                print(f"    - id: {t.get('id', 'none')[:50]}")
            return pd.DataFrame()
        
        tbody = match_table.find('tbody')
        if not tbody:
            print(f"  ✗ No tbody in table")
            return pd.DataFrame()
        
        for row in tbody.find_all('tr'):
            # Skip spacer/header rows
            row_class = row.get('class', [])
            if 'spacer' in row_class or 'thead' in row_class:
                continue
            
            row_data = {'Team': team_name}
            
            # Get all cells (both th and td)
            for cell in row.find_all(['th', 'td']):
                stat = cell.get('data-stat', '')
                if stat:
                    row_data[stat] = cell.get_text(strip=True)
            
            # Only add if we have actual match data (has a date)
            if row_data.get('date') and row_data.get('date') != '':
                matches.append(row_data)
        
        df = pd.DataFrame(matches)
        print(f"  ✓ Found {len(df)} matches")
        
        return df
    
    def scrape_all(self) -> pd.DataFrame:
        """Scrape all teams."""
        print("\n" + "="*60)
        print("Starting FBref Premier League scrape")
        print("="*60)
        
        start = time.time()
        teams = self.get_teams()
        
        all_matches = []
        
        for i, team in enumerate(teams, 1):
            print(f"\n[{i}/{len(teams)}] {team['name']}")
            
            try:
                df = self.get_team_matches(team['url'], team['name'])
                if not df.empty:
                    all_matches.append(df)
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        if not all_matches:
            print("No data collected!")
            return pd.DataFrame()
        
        # Combine and process
        df = pd.concat(all_matches, ignore_index=True)
        df = self._process_data(df)
        
        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"✓ Scrape complete! {len(df)} matches in {elapsed:.1f} minutes")
        print(f"{'='*60}")
        
        return df
    
    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up the data."""
        # Rename columns
        renames = {
            'date': 'Date', 'time': 'Time', 'comp': 'Comp',
            'round': 'Round', 'venue': 'Venue', 'result': 'Result',
            'goals_for': 'GF', 'goals_against': 'GA',
            'opponent': 'Opponent', 'xg_for': 'xG', 'xg_against': 'xGA',
        }
        df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
        
        # Extract gameweek
        if 'Round' in df.columns:
            df['GW'] = df['Round'].apply(
                lambda x: int(re.search(r'(\d+)$', str(x)).group(1)) 
                if pd.notna(x) and re.search(r'(\d+)$', str(x)) else None
            )
        
        # Convert types
        for col in ['GF', 'GA', 'xG', 'xGA', 'GW']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    # =========================================================================
    # PLAYER-LEVEL SCRAPING
    # =========================================================================
    
    def get_team_players(self, team_url: str, team_name: str) -> List[Dict[str, str]]:
        """Get all player URLs from a team page."""
        print(f"\nGetting players for {team_name}...")
        soup = self._get_page(team_url)
        
        players = []
        
        # Find the roster/stats table
        for table in soup.find_all('table'):
            tbody = table.find('tbody')
            if not tbody:
                continue
            
            for row in tbody.find_all('tr'):
                player_cell = row.find('th', {'data-stat': 'player'})
                if player_cell:
                    link = player_cell.find('a')
                    if link and link.get('href'):
                        player_name = link.get_text(strip=True)
                        player_url = FBREF_BASE_URL + link['href']
                        
                        # Get position if available
                        pos_cell = row.find('td', {'data-stat': 'position'})
                        position = pos_cell.get_text(strip=True) if pos_cell else ''
                        
                        if not any(p['name'] == player_name for p in players):
                            players.append({
                                'name': player_name,
                                'url': player_url,
                                'position': position,
                                'team': team_name
                            })
            
            # Found players, stop
            if len(players) > 0:
                break
        
        print(f"  ✓ Found {len(players)} players")
        return players
    
    def get_player_matches(self, player_url: str, player_name: str, 
                           team_name: str, position: str) -> pd.DataFrame:
        """Get match logs for a player with FPL-relevant stats."""
        print(f"  Scraping {player_name}...")
        
        # Navigate to player page first
        soup = self._get_page(player_url)
        
        # Find link to match logs (summary stats)
        matchlogs_link = None
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'matchlogs' in href and '2025-2026' in href and 'summary' in href:
                matchlogs_link = FBREF_BASE_URL + href
                break
        
        if not matchlogs_link:
            # Try to construct URL
            match = re.search(r'/players/([^/]+)/', player_url)
            if match:
                player_id = match.group(1)
                matchlogs_link = f"{FBREF_BASE_URL}/en/players/{player_id}/matchlogs/2025-2026/summary/"
        
        if matchlogs_link:
            soup = self._get_page(matchlogs_link)
        
        # Parse match table
        matches = []
        match_table = None
        
        for table in soup.find_all('table'):
            table_id = table.get('id', '')
            if 'matchlogs' in table_id.lower():
                match_table = table
                break
        
        if not match_table:
            for table in soup.find_all('table'):
                if table.find(['th', 'td'], {'data-stat': 'date'}):
                    match_table = table
                    break
        
        if not match_table:
            return pd.DataFrame()
        
        tbody = match_table.find('tbody')
        if not tbody:
            return pd.DataFrame()
        
        for row in tbody.find_all('tr'):
            row_class = row.get('class', [])
            if 'spacer' in row_class or 'thead' in row_class:
                continue
            
            row_data = {
                'Player': player_name,
                'Team': team_name,
                'Pos': position
            }
            
            for cell in row.find_all(['th', 'td']):
                stat = cell.get('data-stat', '')
                if stat:
                    row_data[stat] = cell.get_text(strip=True)
            
            if row_data.get('date') and row_data.get('date') != '':
                matches.append(row_data)
        
        df = pd.DataFrame(matches)
        if not df.empty:
            print(f"    ✓ {len(df)} matches")
        
        return df
    
    def scrape_all_players(self, max_players_per_team: int = None) -> pd.DataFrame:
        """
        Scrape player match logs for all Premier League players.
        
        Args:
            max_players_per_team: Limit players per team (for testing). None = all players.
        
        Returns:
            DataFrame with player match logs
        """
        print("\n" + "="*60)
        print("Starting FBref Player Match Logs Scrape")
        print("="*60)
        
        start = time.time()
        teams = self.get_teams()
        
        all_matches = []
        total_players = 0
        
        for i, team in enumerate(teams, 1):
            print(f"\n[{i}/{len(teams)}] {team['name']}")
            
            try:
                players = self.get_team_players(team['url'], team['name'])
                
                if max_players_per_team:
                    players = players[:max_players_per_team]
                
                for player in players:
                    try:
                        df = self.get_player_matches(
                            player['url'], 
                            player['name'],
                            player['team'],
                            player['position']
                        )
                        if not df.empty:
                            all_matches.append(df)
                            total_players += 1
                    except Exception as e:
                        print(f"    ✗ Error with {player['name']}: {e}")
                        continue
                        
            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue
        
        if not all_matches:
            print("No player data collected!")
            return pd.DataFrame()
        
        df = pd.concat(all_matches, ignore_index=True)
        df = self._process_player_data(df)
        
        elapsed = (time.time() - start) / 60
        print(f"\n{'='*60}")
        print(f"✓ Player scrape complete!")
        print(f"  {total_players} players, {len(df)} match records")
        print(f"  Time: {elapsed:.1f} minutes")
        print(f"{'='*60}")
        
        return df
    
    def _process_player_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process player match data for FPL analysis."""
        # Rename to match your R functions
        renames = {
            'date': 'Date',
            'comp': 'Comp', 
            'round': 'Round',
            'venue': 'Venue',
            'result': 'Result',
            'minutes': 'Min',
            'gls': 'Gls_Performance',
            'ast': 'Ast_Performance',
            'goals': 'Gls_Performance',
            'assists': 'Ast_Performance',
            'xg': 'xG_Expected',
            'npxg': 'npxG_Expected',
            'xag': 'xAG_Expected',
            'xg_assist': 'xAG_Expected',  # FBref uses this name
            'shots': 'Sh_Performance',
            'shots_on_target': 'SoT_Performance',
        }
        df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})
        
        # Extract gameweek
        if 'Round' in df.columns:
            df['GW'] = df['Round'].apply(
                lambda x: int(re.search(r'(\d+)$', str(x)).group(1)) 
                if pd.notna(x) and re.search(r'(\d+)$', str(x)) else None
            )
        
        # Add season column
        df['Season'] = '2025-2026'
        
        # Convert numeric columns
        numeric_cols = ['Min', 'Gls_Performance', 'Ast_Performance', 
                       'xG_Expected', 'npxG_Expected', 'xAG_Expected',
                       'Sh_Performance', 'SoT_Performance', 'GW']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def close(self):
        if self.driver:
            self.driver.quit()
            print("Browser closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# MAIN
# =============================================================================

def scrape_players():
    """Convenience function to scrape player data."""
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          FBref Player Match Logs Scraper                      ║
    ║                                                               ║
    ║  Scrapes player-level match data for FPL analysis             ║
    ║  WARNING: This takes 30-60+ minutes due to rate limiting      ║
    ║  (hundreds of players × 6 seconds each)                       ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        with FBrefScraper(headless=False) as scraper:
            df = scraper.scrape_all_players()
            
            if df.empty:
                print("No player data collected!")
                return
            
            # Save all matches
            df.to_csv('player_match_logs.csv', index=False)
            print(f"\n✓ Saved to player_match_logs.csv")
            
            # Filter to Premier League only
            if 'Comp' in df.columns:
                pl_df = df[df['Comp'].str.contains('Premier League', case=False, na=False)]
                pl_df.to_csv('player_match_logs_pl.csv', index=False)
                print(f"✓ Saved Premier League only to player_match_logs_pl.csv")
            
            # Summary
            print(f"\nTotal records: {len(df)}")
            print(f"Players: {df['Player'].nunique()}")
            print(f"Teams: {df['Team'].nunique()}")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


def update_gameweek_cli(gameweek: int):
    """Convenience function to update a specific gameweek."""
    try:
        with FBrefScraper(headless=False) as scraper:
            scraper.update_gameweek(gameweek)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


def main():
    """Main entry point - supports both team and player scraping."""
    import sys
    
    # Check command line args
    if len(sys.argv) > 1 and sys.argv[1] == '--players':
        scrape_players()
        return
    
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║          FBref Premier League Match Data Scraper              ║
    ║                                                               ║
    ║  Usage:                                                       ║
    ║    python fbref_scraper.py           → Team match data        ║
    ║    python fbref_scraper.py --players → Player match data      ║
    ║                                                               ║
    ║  Team scrape: ~5 minutes                                      ║
    ║  Player scrape: ~30-60 minutes (many players)                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    try:
        with FBrefScraper(headless=False) as scraper:
            df = scraper.scrape_all()
            
            if df.empty:
                print("No data collected!")
                return
            
            # Save results
            df.to_csv('premier_league_matches.csv', index=False)
            print(f"\n✓ Saved to premier_league_matches.csv")
            
            # Filter to PL only
            if 'Comp' in df.columns:
                pl_df = df[df['Comp'].str.contains('Premier League', case=False, na=False)]
                pl_df.to_csv('premier_league_only.csv', index=False)
                print(f"✓ Saved Premier League only to premier_league_only.csv")
            
            # Summary
            print(f"\nTotal matches: {len(df)}")
            print(f"Teams: {df['Team'].nunique()}")
            print(f"\nSample:")
            print(df[['Date', 'Team', 'Opponent', 'Result']].head())
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()

