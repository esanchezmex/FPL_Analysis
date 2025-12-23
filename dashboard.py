"""
FPL Analytics Dashboard
=======================

Streamlit dashboard for Premier League analytics, replicating the R Shiny dashboard.
Uses plotly for interactive charts and the fpl_analysis module for data processing.

Usage:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fpl_analysis import FPLAnalyzer

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Premier League Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 24px;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data
def load_data(filepath: str) -> pd.DataFrame:
    """Load and cache the player data."""
    return pd.read_csv(filepath)


def init_analyzer(data: pd.DataFrame) -> FPLAnalyzer:
    """Initialize the analyzer (not cached because it modifies state)."""
    return FPLAnalyzer(data, season='2025-2026')


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar(analyzer: FPLAnalyzer):
    """Render sidebar controls."""
    st.sidebar.header("⚽ Analysis Settings")
    
    max_gw = analyzer.get_max_gameweek()
    
    # Gameweek slider
    gameweeks = st.sidebar.slider(
        "Last N Gameweeks",
        min_value=1,
        max_value=max(max_gw, 10),
        value=min(6, max_gw),
        help="Analyze data from the last N gameweeks"
    )
    
    # Position filter
    position = st.sidebar.selectbox(
        "Position Filter",
        options=['all', 'att', 'mid', 'def', 'gk'],
        format_func=lambda x: {
            'all': '📊 All Positions',
            'att': '⚽ Attackers',
            'mid': '🎯 Midfielders',
            'def': '🛡️ Defenders',
            'gk': '🧤 Goalkeepers'
        }.get(x, x)
    )
    
    # Team filter (multi-select)
    teams = analyzer.get_unique_teams()
    team_filter = st.sidebar.multiselect(
        "Team Filter",
        options=teams,
        default=[],
        help="Select teams to filter (leave empty for all teams)"
    )
    team_filter = team_filter if team_filter else None  # None means all teams
    
    # Per 90 toggle
    per_90 = st.sidebar.checkbox("Show Per 90 Stats", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"📅 Data through GW {max_gw}")
    st.sidebar.caption(f"👥 {len(analyzer.get_unique_players())} players")
    st.sidebar.caption(f"🏟️ {len(analyzer.get_unique_teams())} teams")
    
    return {
        'gameweeks': gameweeks,
        'position': position,
        'team_filter': team_filter,
        'per_90': per_90,
        'max_gw': max_gw,
        'min_gw': max_gw - gameweeks
    }


# =============================================================================
# TEAM ANALYSIS TAB
# =============================================================================

def render_team_analysis(analyzer: FPLAnalyzer, settings: dict):
    """Render Team Analysis tab content."""
    
    # Table display options and venue filter
    table_options = {'Top 4': 4, 'Top 10': 10, 'All Teams': 20}
    venue_options = {'All Venues': None, 'Home Only': 'Home', 'Away Only': 'Away'}
    
    col1, col2, col3 = st.columns(3)
    with col1:
        attack_display = st.selectbox("Attack Table Display", options=list(table_options.keys()), index=1, key='attack_display')
    with col2:
        defense_display = st.selectbox("Defense Table Display", options=list(table_options.keys()), index=1, key='defense_display')
    with col3:
        venue_filter = st.selectbox("Venue Filter", options=list(venue_options.keys()), index=0, key='venue_filter')
    
    selected_venue = venue_options[venue_filter]
    
    # Top row: Attack and Defense tables side by side
    col1, col2 = st.columns(2)
    
    with col1:
        venue_label = f" ({venue_filter})" if selected_venue else ""
        st.subheader(f"🔥 Top Team Attacks{venue_label}")
        
        # Attack: group by Team (what they created)
        if settings['per_90']:
            attack_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',  # Always use all positions for team-level
                group_by=['Team'],
                sort_by='npxg_p90',  # Sort by npxG for attacks
                venue=selected_venue
            )
        else:
            attack_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='npxg',
                venue=selected_venue
            )
        
        st.dataframe(
            attack_df.head(table_options[attack_display]),
            use_container_width=True,
            hide_index=True
        )
    
    with col2:
        st.subheader(f"🛡️ Best Defenses (Lowest npxG Conceded){venue_label}")
        
        # Defense: group by Opponent (what they conceded AGAINST)
        if settings['per_90']:
            defense_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',  # Always use all positions for team-level
                group_by=['Opponent'],
                sort_by='npxg_p90',
                ascending=True,  # Lower is better for defense
                venue=selected_venue
            )
        else:
            defense_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Opponent'],
                sort_by='npxg',
                ascending=True,
                venue=selected_venue
            )
        
        # Rename Opponent to Team for display consistency
        if 'Opponent' in defense_df.columns:
            defense_df = defense_df.rename(columns={'Opponent': 'Team'})
        
        st.dataframe(
            defense_df.head(table_options[defense_display]),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("---")
    
    # Second row: Bar charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Top 10 Team Attacks (npxG)")
        
        if settings['per_90']:
            chart_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='npxg_p90',
                venue=selected_venue
            ).head(10)
            y_col = 'npxg_p90'
            y_label = 'npxG per 90'
        else:
            chart_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='npxg',
                venue=selected_venue
            ).head(10)
            y_col = 'npxg'
            y_label = 'npxG'
        
        if not chart_df.empty and y_col in chart_df.columns and 'Team' in chart_df.columns:
            fig = px.bar(
                chart_df.sort_values(y_col, ascending=True),
                x=y_col, y='Team',
                orientation='h',
                color=y_col,
                color_continuous_scale='Blues',
                title=f"Top 10 Teams by {y_label}"
            )
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Top 10 Defenses (Lowest npxG Conceded)")
        
        if settings['per_90']:
            chart_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Opponent'],
                sort_by='npxg_p90',
                ascending=True,
                venue=selected_venue
            ).head(10)
            y_col = 'npxg_p90'
            y_label = 'npxG Conceded per 90'
        else:
            chart_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Opponent'],
                sort_by='npxg',
                ascending=True,
                venue=selected_venue
            ).head(10)
            y_col = 'npxg'
            y_label = 'npxG Conceded'
        
        # Rename Opponent to Team for the chart
        if 'Opponent' in chart_df.columns:
            chart_df = chart_df.rename(columns={'Opponent': 'Team'})
        
        if not chart_df.empty and y_col in chart_df.columns and 'Team' in chart_df.columns:
            fig = px.bar(
                chart_df.sort_values(y_col, ascending=False),
                x=y_col, y='Team',
                orientation='h',
                color=y_col,
                color_continuous_scale='Greens',  # Green for good defense
                title=f"Top 10 Defenses (Lowest {y_label})"
            )
            fig.update_layout(showlegend=False, yaxis={'categoryorder': 'total descending'})
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Third row: Scatter plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚽ Attack: xG vs Actual Goals")
        
        if settings['per_90']:
            scatter_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='npxg_p90',
                venue=selected_venue
            )
            x_col, y_col = 'npxg_p90', 'gls_p90'
            x_label, y_label = 'npxG per 90', 'Goals per 90'
        else:
            scatter_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='npxg',
                venue=selected_venue
            )
            x_col, y_col = 'npxg', 'goals'
            x_label, y_label = 'npxG', 'Goals'
        
        if not scatter_df.empty and x_col in scatter_df.columns and y_col in scatter_df.columns:
            fig = px.scatter(
                scatter_df, x=x_col, y=y_col,
                text='Team',
                title=f"Attack Performance - Last {settings['gameweeks']} GWs"
            )
            # Add diagonal line (overperformance line)
            max_val = max(scatter_df[x_col].max(), scatter_df[y_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', line=dict(dash='dash', color='red'),
                name='Expected = Actual'
            ))
            fig.update_traces(textposition='top center', textfont_size=8, selector=dict(mode='markers+text'))
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for attack scatter plot")
    
    with col2:
        st.subheader("🛡️ Defense: xG Conceded vs Goals Conceded")
        
        if settings['per_90']:
            scatter_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Opponent'],
                sort_by='npxg_p90',
                venue=selected_venue
            )
            x_col, y_col = 'npxg_p90', 'gls_p90'
            x_label, y_label = 'npxG Conceded per 90', 'Goals Conceded per 90'
        else:
            scatter_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Opponent'],
                sort_by='npxg',
                venue=selected_venue
            )
            x_col, y_col = 'npxg', 'goals'
            x_label, y_label = 'npxG Conceded', 'Goals Conceded'
        
        # Rename Opponent to Team for the chart
        if 'Opponent' in scatter_df.columns:
            scatter_df = scatter_df.rename(columns={'Opponent': 'Team'})
        
        if not scatter_df.empty and x_col in scatter_df.columns and y_col in scatter_df.columns:
            fig = px.scatter(
                scatter_df, x=x_col, y=y_col,
                text='Team',
                title=f"Defense Performance - Last {settings['gameweeks']} GWs",
                color_discrete_sequence=['#e74c3c']
            )
            max_val = max(scatter_df[x_col].max(), scatter_df[y_col].max())
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', line=dict(dash='dash', color='red'),
                name='Expected = Actual'
            ))
            fig.update_traces(textposition='top center', textfont_size=8, selector=dict(mode='markers+text'))
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for defense scatter plot")


# =============================================================================
# PLAYER ANALYSIS TAB
# =============================================================================

def render_player_analysis(analyzer: FPLAnalyzer, settings: dict):
    """Render Player Analysis tab content."""
    
    # Venue filter for players
    venue_options = {'All Venues': None, 'Home Only': 'Home', 'Away Only': 'Away'}
    
    col1, col2 = st.columns([3, 1])
    with col2:
        venue_filter = st.selectbox("Venue Filter", options=list(venue_options.keys()), index=0, key='player_venue_filter')
    
    selected_venue = venue_options[venue_filter]
    venue_label = f" ({venue_filter})" if selected_venue else ""
    
    # Top Players Table
    st.subheader(f"🌟 Top Players by xGI{venue_label}")
    
    if settings['per_90']:
        player_df = analyzer.fpl_data_p90(
            min_gameweek=settings['min_gw'],
            position=settings['position'],
            group_by=['Player', 'Team'],
            sort_by='xGI_p90',
            teams=settings['team_filter'],
            venue=selected_venue,
            show_mins=True  # Include minutes column
        )
        sort_col = 'xGI_p90'
    else:
        player_df = analyzer.fpl_data(
            min_gameweek=settings['min_gw'],
            position=settings['position'],
            group_by=['Player', 'Team'],
            sort_by='xGI',
            teams=settings['team_filter'],
            venue=selected_venue
        )
        sort_col = 'xGI'
    
    # Add sorting selector
    col1, col2 = st.columns([3, 1])
    with col2:
        available_cols = [c for c in player_df.columns if c not in ['rank', 'Player', 'Team']]
        sort_by = st.selectbox("Sort by", options=[sort_col] + available_cols, index=0)
    
    if sort_by != sort_col:
        player_df = player_df.sort_values(sort_by, ascending=False)
        player_df['rank'] = range(1, len(player_df) + 1)
    
    st.dataframe(player_df.head(25), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Player xGI Distribution")
        
        if settings['per_90']:
            chart_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position=settings['position'],
                group_by=['Player', 'Team'],
                sort_by='xGI_p90',
                teams=settings['team_filter'],
                venue=selected_venue
            ).head(50)
            x_col, y_col = 'npxg_p90', 'xA_p90'
            x_label, y_label = 'npxG per 90', 'xA per 90'
        else:
            chart_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position=settings['position'],
                group_by=['Player', 'Team'],
                sort_by='xGI',
                teams=settings['team_filter'],
                venue=selected_venue
            ).head(50)
            x_col, y_col = 'npxg', 'xA'
            x_label, y_label = 'npxG', 'xA'
        
        if not chart_df.empty and x_col in chart_df.columns and y_col in chart_df.columns:
            fig = px.scatter(
                chart_df, x=x_col, y=y_col,
                color='Team',
                hover_name='Player',
                title="Player Performance Distribution"
            )
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚽ Goals vs Expected Goals")
        
        if settings['per_90']:
            chart_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position=settings['position'],
                group_by=['Player', 'Team'],
                sort_by='xGI_p90',
                teams=settings['team_filter'],
                venue=selected_venue
            ).head(30)
            x_col, y_col = 'npxg_p90', 'gls_p90'
            x_label, y_label = 'npxG per 90', 'Goals per 90'
        else:
            chart_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position=settings['position'],
                group_by=['Player', 'Team'],
                sort_by='xGI',
                teams=settings['team_filter'],
                venue=selected_venue
            ).head(30)
            x_col, y_col = 'npxg', 'goals'
            x_label, y_label = 'npxG', 'Goals'
        
        if not chart_df.empty and x_col in chart_df.columns and y_col in chart_df.columns:
            fig = px.scatter(
                chart_df, x=x_col, y=y_col,
                hover_name='Player',
                color='Team',
                title="Goals vs Expected Goals"
            )
            max_val = max(chart_df[x_col].max(), chart_df[y_col].max()) if not chart_df.empty else 1
            fig.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode='lines', line=dict(dash='dash', color='red'),
                name='Expected = Actual'
            ))
            fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top players bar chart
    st.subheader("🏆 Top 10 Players by xGI")
    
    if settings['per_90']:
        bar_df = analyzer.fpl_data_p90(
            min_gameweek=settings['min_gw'],
            position=settings['position'],
            group_by=['Player', 'Team'],
            sort_by='xGI_p90',
            teams=settings['team_filter'],
            venue=selected_venue
        ).head(10)
        y_col = 'xGI_p90'
        y_label = 'xGI per 90'
    else:
        bar_df = analyzer.fpl_data(
            min_gameweek=settings['min_gw'],
            position=settings['position'],
            group_by=['Player', 'Team'],
            sort_by='xGI',
            teams=settings['team_filter'],
            venue=selected_venue
        ).head(10)
        y_col = 'xGI'
        y_label = 'xGI'
    
    if not bar_df.empty and y_col in bar_df.columns:
        bar_df['label'] = bar_df['Player'] + ' (' + bar_df['Team'] + ')'
        fig = px.bar(
            bar_df.sort_values(y_col, ascending=True),
            x=y_col, y='label',
            orientation='h',
            color=y_col,
            color_continuous_scale='Viridis',
            title=f"Top 10 Players - Last {settings['gameweeks']} GWs"
        )
        fig.update_layout(yaxis_title='', xaxis_title=y_label, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# COMPARISON TAB
# =============================================================================

def render_comparison(analyzer: FPLAnalyzer, settings: dict):
    """Render Comparison tab content."""
    
    # Comparison type selector
    compare_type = st.radio(
        "Compare:", 
        options=['Teams', 'Players'],
        horizontal=True
    )
    
    st.markdown("---")
    
    if compare_type == 'Teams':
        # Team comparison
        teams = analyzer.get_unique_teams()
        selected_teams = st.multiselect(
            "Select Teams to Compare (2-5)",
            options=teams,
            default=teams[:3] if len(teams) >= 3 else teams,
            max_selections=5
        )
        
        if len(selected_teams) < 2:
            st.warning("Please select at least 2 teams to compare.")
            return
        
        metrics_type = st.radio("Metrics:", options=['Attacking', 'Defensive'], horizontal=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Direct Comparison")
            
            if metrics_type == 'Attacking':
                if settings['per_90']:
                    comp_df = analyzer.fpl_data_p90(
                        min_gameweek=settings['min_gw'],
                        position='all',
                        group_by=['Team'],
                        sort_by='xGI_p90'
                    )
                    comp_df = comp_df[comp_df['Team'].isin(selected_teams)]
                    x_col, y_col = 'npxg_p90', 'xA_p90'
                else:
                    comp_df = analyzer.fpl_data(
                        min_gameweek=settings['min_gw'],
                        position='all',
                        group_by=['Team'],
                        sort_by='xGI'
                    )
                    comp_df = comp_df[comp_df['Team'].isin(selected_teams)]
                    x_col, y_col = 'npxg', 'xA'
            else:
                if settings['per_90']:
                    comp_df = analyzer.fpl_data_p90(
                        min_gameweek=settings['min_gw'],
                        position='all',
                        group_by=['Opponent'],
                        sort_by='npxg_p90'
                    )
                    comp_df = comp_df[comp_df['Opponent'].isin(selected_teams)]
                    comp_df = comp_df.rename(columns={'Opponent': 'Team'})
                    x_col, y_col = 'npxg_p90', 'xA_p90'
                else:
                    comp_df = analyzer.fpl_data(
                        min_gameweek=settings['min_gw'],
                        position='all',
                        group_by=['Opponent'],
                        sort_by='npxg'
                    )
                    comp_df = comp_df[comp_df['Opponent'].isin(selected_teams)]
                    comp_df = comp_df.rename(columns={'Opponent': 'Team'})
                    x_col, y_col = 'npxg', 'xA'
            
            if not comp_df.empty and x_col in comp_df.columns and y_col in comp_df.columns:
                fig = px.scatter(
                    comp_df, x=x_col, y=y_col,
                    color='Team', size_max=15,
                    text='Team'
                )
                fig.update_traces(textposition='top center', marker=dict(size=15))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Metrics Comparison")
            
            if not comp_df.empty:
                # Create grouped bar chart
                metric_cols = [c for c in comp_df.columns if c not in ['rank', 'Team', 'Opponent']]
                
                fig = go.Figure()
                for team in comp_df['Team'].unique():
                    team_data = comp_df[comp_df['Team'] == team]
                    values = [team_data[col].values[0] if col in team_data.columns else 0 for col in metric_cols[:5]]
                    fig.add_trace(go.Bar(name=team, x=metric_cols[:5], y=values))
                
                fig.update_layout(barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Player comparison
        if settings['per_90']:
            all_players = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position=settings['position'],
                group_by=['Player', 'Team'],
                sort_by='xGI_p90'
            ).head(50)
        else:
            all_players = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position=settings['position'],
                group_by=['Player', 'Team'],
                sort_by='xGI'
            ).head(50)
        
        # Check if we have data
        if all_players.empty or 'Player' not in all_players.columns:
            st.warning(f"No players found for position filter '{settings['position']}'. Try changing the position filter in the sidebar.")
            return
        
        player_options = [f"{row['Player']} ({row['Team']})" for _, row in all_players.iterrows()]
        player_names = all_players['Player'].tolist()
        
        if not player_options:
            st.warning("No players available for comparison with current filters.")
            return
        
        selected_labels = st.multiselect(
            "Select Players to Compare (2-5)",
            options=player_options,
            default=player_options[:3] if len(player_options) >= 3 else player_options,
            max_selections=5
        )
        
        # Map back to player names
        selected_players = [player_names[player_options.index(label)] for label in selected_labels]
        
        if len(selected_players) < 2:
            st.warning("Please select at least 2 players to compare.")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Direct Comparison")
            
            comp_df = all_players[all_players['Player'].isin(selected_players)]
            
            if settings['per_90']:
                x_col, y_col = 'npxg_p90', 'xA_p90'
            else:
                x_col, y_col = 'npxg', 'xA'
            
            if not comp_df.empty and x_col in comp_df.columns and y_col in comp_df.columns:
                fig = px.scatter(
                    comp_df, x=x_col, y=y_col,
                    color='Player',
                    text='Player',
                    hover_data=['Team']
                )
                fig.update_traces(textposition='top center', marker=dict(size=15))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📊 Metrics Comparison")
            
            if not comp_df.empty:
                metric_cols = [c for c in comp_df.columns if c not in ['rank', 'Player', 'Team']]
                
                fig = go.Figure()
                for _, row in comp_df.iterrows():
                    values = [row[col] if col in comp_df.columns else 0 for col in metric_cols[:5]]
                    fig.add_trace(go.Bar(name=row['Player'], x=metric_cols[:5], y=values))
                
                fig.update_layout(barmode='group', height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Position comparison (always show)
    st.subheader("📊 Position Performance Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot by position
        position_data = []
        for pos in ['att', 'mid', 'def']:
            if settings['per_90']:
                pos_df = analyzer.fpl_data_p90(
                    min_gameweek=settings['min_gw'],
                    position=pos,
                    group_by=['Player', 'Team'],
                    sort_by='xGI_p90'
                ).head(15)
                pos_df['Position'] = {'att': 'Attackers', 'mid': 'Midfielders', 'def': 'Defenders'}[pos]
                if 'xGI_p90' in pos_df.columns:
                    position_data.append(pos_df)
            else:
                pos_df = analyzer.fpl_data(
                    min_gameweek=settings['min_gw'],
                    position=pos,
                    group_by=['Player', 'Team'],
                    sort_by='xGI'
                ).head(15)
                pos_df['Position'] = {'att': 'Attackers', 'mid': 'Midfielders', 'def': 'Defenders'}[pos]
                if 'xGI' in pos_df.columns:
                    position_data.append(pos_df)
        
        if position_data:
            combined_df = pd.concat(position_data, ignore_index=True)
            y_col = 'xGI_p90' if settings['per_90'] else 'xGI'
            
            if y_col in combined_df.columns:
                fig = px.box(
                    combined_df, x='Position', y=y_col,
                    color='Position',
                    title="xGI Distribution by Position"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Team vs Top Player scatter
        if settings['per_90']:
            team_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='xGI_p90'
            )
            player_df = analyzer.fpl_data_p90(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Player', 'Team'],
                sort_by='xGI_p90'
            )
            metric_col = 'xGI_p90'
        else:
            team_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Team'],
                sort_by='xGI'
            )
            player_df = analyzer.fpl_data(
                min_gameweek=settings['min_gw'],
                position='all',
                group_by=['Player', 'Team'],
                sort_by='xGI'
            )
            metric_col = 'xGI'
        
        # Get top player per team
        if not player_df.empty and metric_col in player_df.columns:
            top_players = player_df.groupby('Team').first().reset_index()
            
            if not team_df.empty and metric_col in team_df.columns:
                merged = team_df.merge(
                    top_players[['Team', metric_col, 'Player']],
                    on='Team',
                    suffixes=('_team', '_top_player')
                )
                
                if not merged.empty:
                    fig = px.scatter(
                        merged,
                        x=f'{metric_col}_team',
                        y=f'{metric_col}_top_player',
                        hover_name='Team',
                        hover_data=['Player'],
                        title="Team Performance vs Top Player"
                    )
                    fig.update_layout(
                        xaxis_title=f"Team {metric_col}",
                        yaxis_title=f"Top Player {metric_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.title("⚽ Premier League Analytics Dashboard")
    
    # Try to load data
    try:
        data = load_data('player_match_logs_pl.csv')
        analyzer = init_analyzer(data)
    except FileNotFoundError:
        st.error("❌ Data file not found!")
        st.info("Please run `python fbref_scraper.py --players` first to generate the data.")
        
        # File upload option
        uploaded_file = st.file_uploader("Or upload your player data CSV", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            analyzer = init_analyzer(data)
        else:
            return
    
    # Render sidebar and get settings
    settings = render_sidebar(analyzer)
    
    # Main content with tabs
    tab1, tab2, tab3 = st.tabs(["🏟️ Team Analysis", "👤 Player Analysis", "⚖️ Comparison"])
    
    with tab1:
        render_team_analysis(analyzer, settings)
    
    with tab2:
        render_player_analysis(analyzer, settings)
    
    with tab3:
        render_comparison(analyzer, settings)


if __name__ == "__main__":
    main()
