"""
FPL Analysis Framework
======================

Python replication of the R analysis functions from ANALYSIS_FRAMEWORK_1.
Designed for FPL (Fantasy Premier League) data analysis with flexible 
grouping, filtering, and aggregation.

Compatible with future Streamlit dashboard integration.

Usage:
    from fpl_analysis import FPLAnalyzer

    analyzer = FPLAnalyzer('player_match_logs_pl.csv')
    
    # Total stats
    df = analyzer.fpl_data(min_gameweek=10, position='att', group_by=['Player'])
    
    # Per-90 stats
    df = analyzer.fpl_data_p90(min_gameweek=5, position='mid', group_by=['Player', 'Team'])
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union, Literal
from dataclasses import dataclass


# =============================================================================
# POSITION PATTERNS (matching R logic + FBref notation)
# =============================================================================

POSITION_PATTERNS = {
    'def': r'CB|RB|LB|RWB|LWB|DF|FB',  # Added DF, FB for FBref
    'mid': r'CM|AM|DM|LW|RW|RM|LM|CAM|CDM|MF',  # Added MF for FBref
    'att': r'FW|CF|ST|RW|LW|AM',
    'gk': r'GK',
    'all': r'.'  # Match any position
}

# FPL-specific position mapping (more strict)
FPL_POSITION_PATTERNS = {
    'def': r'^(CB|RB|LB|RWB|LWB)',       # Defenders
    'mid': r'^(CM|DM|LM|RM|CAM|CDM)',     # Midfielders (central)
    'att': r'^(FW|CF|ST|RW|LW|AM)',       # Attackers/Wingers
    'gk': r'^GK',                          # Goalkeepers
    'all': r'.'                            # Any position
}


# =============================================================================
# ANALYZER CLASS
# =============================================================================

class FPLAnalyzer:
    """
    FPL data analyzer with flexible grouping and aggregation.
    
    Attributes:
        df: The underlying player match data DataFrame
        season: Current season filter (default: '2025-2026')
    """
    
    def __init__(self, data: Union[str, pd.DataFrame], season: str = '2025-2026'):
        """
        Initialize the analyzer.
        
        Args:
            data: Either a file path to CSV or a DataFrame
            season: Season to filter for (default: '2025-2026')
        """
        if isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            self.df = data.copy()
        
        self.season = season
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """Prepare and validate the data."""
        # Rename FBref columns to expected names if they exist
        # Only rename lowercase if uppercase doesn't already exist (to avoid duplicates)
        column_renames = {
            'xg_assist': 'xAG_Expected',  # FBref uses xg_assist for expected assists
        }
        
        # Handle opponent column (only if Opponent doesn't exist)
        if 'opponent' in self.df.columns and 'Opponent' not in self.df.columns:
            column_renames['opponent'] = 'Opponent'
        elif 'opponent' in self.df.columns and 'Opponent' in self.df.columns:
            # Drop the lowercase duplicate
            self.df = self.df.drop(columns=['opponent'])
        
        self.df = self.df.rename(columns={k: v for k, v in column_renames.items() if k in self.df.columns})
        
        # Handle duplicate columns (can happen when appending data)
        if self.df.columns.duplicated().any():
            print(f"Warning: Found duplicate columns, keeping first occurrence")
            self.df = self.df.loc[:, ~self.df.columns.duplicated()]
        
        # Ensure numeric columns
        numeric_cols = [
            'Min', 'GW', 
            'Gls_Performance', 'Ast_Performance',
            'npxG_Expected', 'xG_Expected', 'xAG_Expected',
            'Sh_Performance', 'SoT_Performance'
        ]
        
        for col in numeric_cols:
            if col in self.df.columns:
                # Ensure it's a Series (not DataFrame from duplicate cols)
                col_data = self.df[col]
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.iloc[:, 0]
                self.df[col] = pd.to_numeric(col_data, errors='coerce')
        
        # Ensure required columns exist
        required = ['Player', 'Team', 'GW']
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            print(f"Warning: Missing columns: {missing}")
    
    def _filter_data(
        self, 
        min_gameweek: int = 0,
        max_gameweek: Optional[int] = None,
        position: str = 'all',
        teams: Optional[List[str]] = None,
        players: Optional[List[str]] = None,
        venue: Optional[Literal['Home', 'Away']] = None,
        comp: str = 'Premier League'
    ) -> pd.DataFrame:
        """
        Apply filters to the data.
        
        Args:
            min_gameweek: Only include matches AFTER this gameweek (exclusive)
            max_gameweek: Only include matches up to this gameweek (inclusive)
            position: Position filter ('def', 'mid', 'att', 'gk', 'all')
            teams: List of teams to include (None = all)
            players: List of players to include (None = all)
            venue: Filter by Home/Away (None = both)
            comp: Competition filter (default: 'Premier League')
        
        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()
        
        # Season filter
        if 'Season' in df.columns and self.season:
            df = df[df['Season'] == self.season]
        
        # Competition filter
        if 'Comp' in df.columns and comp:
            df = df[df['Comp'].str.contains(comp, case=False, na=False)]
        
        # Gameweek filters
        if 'GW' in df.columns:
            df = df[df['GW'] > min_gameweek]
            if max_gameweek is not None:
                df = df[df['GW'] <= max_gameweek]
        
        # Position filter
        if position != 'all' and 'Pos' in df.columns:
            pattern = POSITION_PATTERNS.get(position, POSITION_PATTERNS['all'])
            df = df[df['Pos'].str.contains(pattern, case=False, na=False, regex=True)]
        
        # Team filter
        if teams is not None and 'Team' in df.columns:
            df = df[df['Team'].isin(teams)]
        
        # Player filter
        if players is not None and 'Player' in df.columns:
            df = df[df['Player'].isin(players)]
        
        # Venue filter
        if venue is not None and 'Venue' in df.columns:
            df = df[df['Venue'] == venue]
        
        return df
    
    def fpl_data(
        self,
        min_gameweek: int = 0,
        max_gameweek: Optional[int] = None,
        position: str = 'all',
        group_by: List[str] = ['Player'],
        sort_by: str = 'xGI',
        ascending: bool = False,
        teams: Optional[List[str]] = None,
        venue: Optional[Literal['Home', 'Away']] = None,
        min_xGI: float = 0
    ) -> pd.DataFrame:
        """
        Aggregate FPL data with flexible grouping (total stats).
        
        Replicates the R function fpl_data().
        
        Args:
            min_gameweek: Only include matches AFTER this gameweek
            max_gameweek: Only include matches up to this gameweek
            position: Position filter ('def', 'mid', 'att', 'gk', 'all')
            group_by: Columns to group by (e.g., ['Player'], ['Player', 'Team'], ['Team', 'Venue'])
            sort_by: Column to sort by (default: 'xGI')
            ascending: Sort ascending (default: False = descending)
            teams: List of teams to include
            venue: Filter by Home/Away
            min_xGI: Minimum xGI to include (default: 0)
        
        Returns:
            Aggregated DataFrame with rank, group columns, and stats
        """
        # Apply filters
        df = self._filter_data(
            min_gameweek=min_gameweek,
            max_gameweek=max_gameweek,
            position=position,
            teams=teams,
            venue=venue
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Validate group_by columns
        valid_groups = [col for col in group_by if col in df.columns]
        if not valid_groups:
            valid_groups = ['Player'] if 'Player' in df.columns else df.columns[:1].tolist()
        
        # Aggregation
        agg_dict = {
            'npxG_Expected': ('npxG_Expected', 'sum'),
            'Gls_Performance': ('Gls_Performance', 'sum'),
            'xAG_Expected': ('xAG_Expected', 'sum'),
            'Ast_Performance': ('Ast_Performance', 'sum'),
            'Sh_Performance': ('Sh_Performance', 'sum'),
            'SoT_Performance': ('SoT_Performance', 'sum'),
            'Min': ('Min', 'sum'),
            'games': ('GW', 'count')
        }
        
        # Only aggregate columns that exist
        agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df.columns}
        
        result = df.groupby(valid_groups, as_index=False).agg(**agg_dict)
        
        # Rename columns for consistency
        result = result.rename(columns={
            'npxG_Expected': 'npxg',
            'Gls_Performance': 'goals',
            'xAG_Expected': 'xA',
            'Ast_Performance': 'assists',
            'Sh_Performance': 'shots',
            'SoT_Performance': 'ShT',
            'Min': 'mins'
        })
        
        # Calculate derived metrics
        result['xGI'] = result.get('npxg', 0) + result.get('xA', 0)
        result['GI'] = result.get('goals', 0) + result.get('assists', 0)
        
        # Filter by min_xGI
        if min_xGI > 0:
            result = result[result['xGI'] > min_xGI]
        
        # Sort
        if sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)
        
        # Add rank
        result = result.reset_index(drop=True)
        result.insert(0, 'rank', range(1, len(result) + 1))
        
        return result
    
    def fpl_data_p90(
        self,
        min_gameweek: int = 0,
        max_gameweek: Optional[int] = None,
        position: str = 'all',
        group_by: List[str] = ['Player'],
        sort_by: str = 'xGI_p90',
        ascending: bool = False,
        teams: Optional[List[str]] = None,
        venue: Optional[Literal['Home', 'Away']] = None,
        min_minutes: Optional[float] = None,
        min_minutes_pct: float = 0.75,
        min_xGI_p90: float = 0,
        show_mins: bool = False
    ) -> pd.DataFrame:
        """
        Aggregate FPL data with per-90 minute normalization.
        
        Replicates the R function fpl_data_p90().
        
        Args:
            min_gameweek: Only include matches AFTER this gameweek
            max_gameweek: Only include matches up to this gameweek
            position: Position filter ('def', 'mid', 'att', 'gk', 'all')
            group_by: Columns to group by
            sort_by: Column to sort by (default: 'xGI_p90')
            ascending: Sort ascending (default: False)
            teams: List of teams to include
            venue: Filter by Home/Away
            min_minutes: Explicit minimum minutes threshold
            min_minutes_pct: If min_minutes is None, use this % of average (default: 0.75)
            min_xGI_p90: Minimum xGI per 90 to include
        
        Returns:
            Aggregated DataFrame with per-90 stats
        """
        # Apply filters
        df = self._filter_data(
            min_gameweek=min_gameweek,
            max_gameweek=max_gameweek,
            position=position,
            teams=teams,
            venue=venue
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Validate group_by columns
        valid_groups = [col for col in group_by if col in df.columns]
        if not valid_groups:
            valid_groups = ['Player'] if 'Player' in df.columns else df.columns[:1].tolist()
        
        # Calculate minimum minutes threshold
        if min_minutes is None and 'Player' in valid_groups:
            # Calculate based on average player minutes
            player_mins = df.groupby('Player')['Min'].sum()
            min_minutes = player_mins.mean() * min_minutes_pct
        
        # Check if this is player-level or team-level aggregation
        is_player_level = 'Player' in valid_groups
        
        if is_player_level:
            # Player-level: simple sum aggregation
            agg_dict = {
                'Min': ('Min', 'sum'),
                'npxG_Expected': ('npxG_Expected', 'sum'),
                'Gls_Performance': ('Gls_Performance', 'sum'),
                'xAG_Expected': ('xAG_Expected', 'sum'),
                'Ast_Performance': ('Ast_Performance', 'sum'),
                'Sh_Performance': ('Sh_Performance', 'sum'),
                'SoT_Performance': ('SoT_Performance', 'sum'),
            }
            agg_dict = {k: v for k, v in agg_dict.items() if v[0] in df.columns}
            
            result = df.groupby(valid_groups, as_index=False).agg(**agg_dict)
            
            # Apply min_minutes filter
            if min_minutes is not None and 'Min' in result.columns:
                result = result[result['Min'] >= min_minutes]
        
        else:
            # Team-level: average minutes per game, sum stats
            # First: aggregate per gameweek
            agg_cols = {
                'Min': ('Min', 'sum'),
                'npxG_Expected': ('npxG_Expected', 'sum'),
                'Gls_Performance': ('Gls_Performance', 'sum'),
                'xAG_Expected': ('xAG_Expected', 'sum'),
                'Ast_Performance': ('Ast_Performance', 'sum'),
                'Sh_Performance': ('Sh_Performance', 'sum'),
                'SoT_Performance': ('SoT_Performance', 'sum'),
            }
            agg_cols = {k: v for k, v in agg_cols.items() if v[0] in df.columns}
            
            per_gw = df.groupby(valid_groups + ['GW'], as_index=False).agg(**agg_cols)
            
            # Then: collapse across gameweeks (mean for mins, sum for stats)
            result_agg = {
                'Min': 'mean',  # Average minutes per game
                'npxG_Expected': 'sum',
                'Gls_Performance': 'sum',
                'xAG_Expected': 'sum',
                'Ast_Performance': 'sum',
                'Sh_Performance': 'sum',
                'SoT_Performance': 'sum',
            }
            result_agg = {k: v for k, v in result_agg.items() if k in per_gw.columns}
            
            result = per_gw.groupby(valid_groups, as_index=False).agg(result_agg)
        
        # Rename columns
        result = result.rename(columns={
            'Min': 'mins',
            'npxG_Expected': 'npxg',
            'Gls_Performance': 'goals',
            'xAG_Expected': 'xA',
            'Ast_Performance': 'assists',
            'Sh_Performance': 'shots',
            'SoT_Performance': 'ShT',
        })
        
        # Calculate per-90 stats (avoid division by zero)
        if 'mins' in result.columns and result['mins'].sum() > 0:
            result['npxg_p90'] = ((result.get('npxg', 0) / result['mins']) * 90).round(2)
            result['gls_p90'] = ((result.get('goals', 0) / result['mins']) * 90).round(2)
            result['xA_p90'] = ((result.get('xA', 0) / result['mins']) * 90).round(2)
            result['ast_p90'] = ((result.get('assists', 0) / result['mins']) * 90).round(2)
            result['xGI_p90'] = result['npxg_p90'] + result['xA_p90']
            result['GI_p90'] = result['gls_p90'] + result['ast_p90']
            
            if 'shots' in result.columns:
                result['shots_p90'] = ((result['shots'] / result['mins']) * 90).round(2)
            if 'ShT' in result.columns:
                result['ShT_p90'] = ((result['ShT'] / result['mins']) * 90).round(2)
        
        # Filter by min_xGI_p90
        if min_xGI_p90 > 0 and 'xGI_p90' in result.columns:
            result = result[result['xGI_p90'] > min_xGI_p90]
        
        # Sort
        if sort_by in result.columns:
            result = result.sort_values(sort_by, ascending=ascending)
        
        # Add rank
        result = result.reset_index(drop=True)
        result.insert(0, 'rank', range(1, len(result) + 1))
        
        # Drop raw totals (keep only p90 versions) - match R behavior
        # Optionally keep mins if show_mins=True
        drop_cols = ['npxg', 'goals', 'xA', 'assists', 'shots', 'ShT']
        if not show_mins:
            drop_cols.append('mins')
        result = result.drop(columns=[c for c in drop_cols if c in result.columns], errors='ignore')
        
        return result
    
    # =========================================================================
    # CONVENIENCE METHODS (Streamlit-friendly)
    # =========================================================================
    
    def get_unique_teams(self) -> List[str]:
        """Get list of unique teams for dropdown."""
        if 'Team' in self.df.columns:
            return sorted(self.df['Team'].dropna().unique().tolist())
        return []
    
    def get_unique_players(self) -> List[str]:
        """Get list of unique players for dropdown."""
        if 'Player' in self.df.columns:
            return sorted(self.df['Player'].dropna().unique().tolist())
        return []
    
    def get_max_gameweek(self) -> int:
        """Get the maximum gameweek in the data."""
        if 'GW' in self.df.columns:
            return int(self.df['GW'].max())
        return 0
    
    def get_available_positions(self) -> List[str]:
        """Get available position filter options."""
        return ['all', 'def', 'mid', 'att', 'gk']
    
    def get_available_group_by_options(self) -> List[str]:
        """Get available columns for grouping."""
        potential = ['Player', 'Team', 'Venue', 'Opponent', 'Pos']
        return [c for c in potential if c in self.df.columns]
    
    def get_available_sort_options(self, per_90: bool = False) -> List[str]:
        """Get available columns for sorting."""
        if per_90:
            return ['xGI_p90', 'npxg_p90', 'gls_p90', 'xA_p90', 'ast_p90', 'mins', 'shots_p90']
        return ['xGI', 'npxg', 'goals', 'xA', 'assists', 'shots', 'ShT', 'mins', 'games', 'GI']


# =============================================================================
# STANDALONE FUNCTIONS (for quick use without class)
# =============================================================================

def load_and_analyze(
    filepath: str = 'player_match_logs_pl.csv',
    min_gameweek: int = 0,
    position: str = 'all',
    group_by: List[str] = ['Player'],
    per_90: bool = False,
    sort_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Quick function to load data and run analysis.
    
    Args:
        filepath: Path to player match logs CSV
        min_gameweek: Filter to matches after this GW
        position: Position filter
        group_by: Grouping columns
        per_90: Whether to use per-90 stats
        sort_by: Column to sort by (auto-detected if None)
    
    Returns:
        Analyzed DataFrame
    """
    analyzer = FPLAnalyzer(filepath)
    
    if per_90:
        sort_col = sort_by or 'xGI_p90'
        return analyzer.fpl_data_p90(
            min_gameweek=min_gameweek,
            position=position,
            group_by=group_by,
            sort_by=sort_col
        )
    else:
        sort_col = sort_by or 'xGI'
        return analyzer.fpl_data(
            min_gameweek=min_gameweek,
            position=position,
            group_by=group_by,
            sort_by=sort_col
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Try to load data
    try:
        analyzer = FPLAnalyzer('player_match_logs_pl.csv')
        max_gw = analyzer.get_max_gameweek()
        
        print(f"Loaded data through GW {max_gw}")
        print(f"Teams: {len(analyzer.get_unique_teams())}")
        print(f"Players: {len(analyzer.get_unique_players())}")
        
        # Example: Top attackers by xGI in last 5 gameweeks
        print("\n" + "="*60)
        print("TOP ATTACKERS BY xGI (Last 5 GWs)")
        print("="*60)
        
        last_5_gw = max_gw - 5
        result = analyzer.fpl_data(
            min_gameweek=last_5_gw,
            position='att',
            group_by=['Player', 'Team'],
            sort_by='xGI'
        )
        print(result.head(15).to_string(index=False))
        
        # Example: Team-level per-90 analysis
        print("\n" + "="*60)
        print("TEAMS BY xGI PER 90 (Season)")
        print("="*60)
        
        team_result = analyzer.fpl_data_p90(
            min_gameweek=0,
            position='all',
            group_by=['Team'],
            sort_by='xGI_p90'
        )
        print(team_result.head(10).to_string(index=False))
        
    except FileNotFoundError:
        print("No player data found. Run 'python fbref_scraper.py --players' first.")
        sys.exit(1)
