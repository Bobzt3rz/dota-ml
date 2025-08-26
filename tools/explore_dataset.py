#!/usr/bin/env python3
"""
Dota 2 Public Matches Dataset Explorer
Explores the 500k matches dataset to understand the data structure and patterns.
Enhanced with dotaconstants integration for human-readable hero names and analysis.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import Counter
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_hero_constants():
    """Load hero data from dotaconstants submodule"""
    try:
        # Path to dotaconstants submodule
        constants_path = os.path.join(os.path.dirname(__file__), '..', 'libs', 'dotaconstants', 'build')
        
        with open(os.path.join(constants_path, 'heroes.json'), 'r') as f:
            heroes_data = json.load(f)
        
        # Create ID to name mapping
        id_to_name = {}
        name_to_id = {}
        
        for hero_id, hero_info in heroes_data.items():
            hero_id = int(hero_id)
            hero_name = hero_info.get('localized_name', hero_info.get('name', f'Hero {hero_id}'))
            id_to_name[hero_id] = hero_name
            name_to_id[hero_name] = hero_id
            
        print(f"✅ Loaded {len(id_to_name)} heroes from dotaconstants")
        return id_to_name, name_to_id, heroes_data
        
    except Exception as e:
        print(f"⚠️  Could not load hero constants: {e}")
        print("📝 To fix this:")
        print("   1. Make sure dotaconstants submodule is in libs/dotaconstants/")
        print("   2. Run: cd libs/dotaconstants && npm install && npm run build")
        print("Using fallback hero mapping...")
        return {}, {}, {}

def get_hero_info(hero_id, heroes_data):
    """Get detailed hero information"""
    hero_data = heroes_data.get(str(hero_id), {})
    return {
        'name': hero_data.get('localized_name', f'Hero {hero_id}'),
        'primary_attr': hero_data.get('primary_attr', 'unknown'),
        'attack_type': hero_data.get('attack_type', 'unknown'),
        'roles': hero_data.get('roles', [])
    }

def load_data(filepath):
    """Load the JSON data from OpenDota API format and convert to DataFrame"""
    print("Loading data...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract the rows from the OpenDota API response format
    if isinstance(data, dict) and 'rows' in data:
        matches = data['rows']
        print(f"Loaded {len(matches)} matches from OpenDota API response")
        print(f"Query returned {data.get('rowCount', 'unknown')} rows")
    else:
        # If it's just a list of matches
        matches = data
        print(f"Loaded {len(matches)} matches from direct array")
    
    df = pd.DataFrame(matches)
    print(f"DataFrame shape: {df.shape}")
    return df

def basic_info(df):
    """Print basic information about the dataset"""
    print("\n" + "="*60)
    print("BASIC DATASET INFO")
    print("="*60)
    
    print(f"Number of matches: {len(df):,}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    print(f"\nColumns available:")
    for i, col in enumerate(df.columns, 1):
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        null_pct = (len(df) - non_null) / len(df) * 100
        print(f"  {i:2d}. {col:<15} | {dtype:<10} | {non_null:>7,} non-null ({null_pct:.1f}% missing)")
    
    print(f"\nSample of first few rows:")
    print(df.head(3).to_string())

def time_analysis(df):
    """Analyze temporal patterns in the data"""
    print("\n" + "="*60)
    print("TIME ANALYSIS")
    print("="*60)
    
    # Convert start_time to datetime
    df['datetime'] = pd.to_datetime(df['start_time'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.day_name()
    
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    time_span = (df['datetime'].max() - df['datetime'].min()).days
    print(f"Time span: {time_span} days")
    print(f"Average matches per day: {len(df) / time_span:.0f}")
    
    # Recent activity
    recent_days = df['date'].value_counts().sort_index().tail(7)
    print(f"\nMatches in last 7 days:")
    for date, count in recent_days.items():
        print(f"  {date}: {count:,} matches")
    
    # Create time analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Daily match distribution (last 30 days)
    daily_matches = df['date'].value_counts().sort_index().tail(30)
    ax1.plot(daily_matches.index, daily_matches.values, marker='o', linewidth=2, markersize=4)
    ax1.set_title('Daily Matches (Last 30 Days)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Matches')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Hourly distribution
    hourly_matches = df['hour'].value_counts().sort_index()
    ax2.bar(hourly_matches.index, hourly_matches.values, alpha=0.7, color='skyblue')
    ax2.set_title('Matches by Hour of Day (UTC)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Hour')
    ax2.set_ylabel('Number of Matches')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Day of week distribution
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    ax3.bar(range(len(day_counts)), day_counts.values, alpha=0.7, color='lightcoral')
    ax3.set_title('Matches by Day of Week', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Number of Matches')
    ax3.set_xticks(range(len(day_counts)))
    ax3.set_xticklabels([day[:3] for day in day_order], rotation=0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Match duration over time (sample)
    if len(df) > 10000:
        sample_df = df.sample(10000)  # Sample for performance
    else:
        sample_df = df
    ax4.scatter(sample_df['datetime'], sample_df['duration']/60, alpha=0.3, s=1)
    ax4.set_title('Match Duration Over Time (Sample)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Duration (minutes)')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('../results/time_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def game_mode_analysis(df):
    """Analyze game modes and lobby types"""
    print("\n" + "="*60)
    print("GAME MODE & LOBBY ANALYSIS")
    print("="*60)
    
    print("Game modes distribution:")
    game_modes = df['game_mode'].value_counts()
    total_matches = len(df)
    for mode, count in game_modes.head(10).items():
        pct = count / total_matches * 100
        print(f"  Mode {mode}: {count:,} matches ({pct:.1f}%)")
    
    print(f"\nLobby types distribution:")
    lobby_types = df['lobby_type'].value_counts()
    for lobby, count in lobby_types.head(10).items():
        pct = count / total_matches * 100
        print(f"  Lobby {lobby}: {count:,} matches ({pct:.1f}%)")
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Game modes
    top_modes = game_modes.head(8)
    ax1.bar(range(len(top_modes)), top_modes.values, alpha=0.7, color='lightblue')
    ax1.set_title('Top Game Modes', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Matches')
    ax1.set_xlabel('Game Mode ID')
    ax1.set_xticks(range(len(top_modes)))
    ax1.set_xticklabels(top_modes.index)
    for i, v in enumerate(top_modes.values):
        ax1.text(i, v + max(top_modes.values)*0.01, f'{v:,}', ha='center', fontsize=9)
    
    # Lobby types
    top_lobbies = lobby_types.head(8)
    ax2.bar(range(len(top_lobbies)), top_lobbies.values, alpha=0.7, color='lightgreen')
    ax2.set_title('Top Lobby Types', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Matches')
    ax2.set_xlabel('Lobby Type ID')
    ax2.set_xticks(range(len(top_lobbies)))
    ax2.set_xticklabels(top_lobbies.index)
    for i, v in enumerate(top_lobbies.values):
        ax2.text(i, v + max(top_lobbies.values)*0.01, f'{v:,}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('../results/game_modes.png', dpi=300, bbox_inches='tight')
    plt.show()

def winrate_analysis(df):
    """Analyze winrates and match outcomes"""
    print("\n" + "="*60)
    print("WINRATE ANALYSIS")
    print("="*60)
    
    radiant_wins = df['radiant_win'].sum()
    total_matches = len(df)
    radiant_winrate = radiant_wins / total_matches
    
    print(f"Total matches analyzed: {total_matches:,}")
    print(f"Radiant wins: {radiant_wins:,}")
    print(f"Dire wins: {total_matches - radiant_wins:,}")
    print(f"Radiant winrate: {radiant_winrate:.4f} ({radiant_winrate*100:.2f}%)")
    print(f"Dire winrate: {1-radiant_winrate:.4f} ({(1-radiant_winrate)*100:.2f}%)")
    
    balance_diff = abs(0.5 - radiant_winrate) * 100
    print(f"Balance deviation from 50%: {balance_diff:.2f}%")
    
    # Winrate by skill bracket
    if 'avg_rank_tier' in df.columns and df['avg_rank_tier'].notna().sum() > 0:
        rank_stats = df.groupby('avg_rank_tier').agg({
            'radiant_win': ['mean', 'count'],
            'duration': 'mean'
        }).round(3)
        rank_stats.columns = ['radiant_winrate', 'match_count', 'avg_duration']
        rank_stats = rank_stats[rank_stats['match_count'] >= 100]  # Filter small samples
        
        print(f"\nWinrate by rank tier (min 100 matches):")
        print(rank_stats.head(15))
        
        # Plot rank analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Winrate by rank
        rank_stats['radiant_winrate'].plot(kind='bar', ax=ax1, alpha=0.7, color='orange')
        ax1.set_title('Radiant Winrate by Rank Tier', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Radiant Winrate')
        ax1.set_xlabel('Average Rank Tier')
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Match count by rank
        rank_stats['match_count'].plot(kind='bar', ax=ax2, alpha=0.7, color='purple')
        ax2.set_title('Match Count by Rank Tier', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Number of Matches')
        ax2.set_xlabel('Average Rank Tier')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('../results/winrate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def duration_analysis(df):
    """Analyze match duration patterns"""
    print("\n" + "="*60)
    print("MATCH DURATION ANALYSIS")
    print("="*60)
    
    # Convert duration to minutes
    df['duration_minutes'] = df['duration'] / 60
    
    duration_stats = df['duration_minutes'].describe()
    print("Duration statistics (minutes):")
    print(f"  Mean: {duration_stats['mean']:.1f}")
    print(f"  Median: {duration_stats['50%']:.1f}")
    print(f"  Std Dev: {duration_stats['std']:.1f}")
    print(f"  Min: {duration_stats['min']:.1f}")
    print(f"  Max: {duration_stats['max']:.1f}")
    print(f"  25th percentile: {duration_stats['25%']:.1f}")
    print(f"  75th percentile: {duration_stats['75%']:.1f}")
    
    # Duration categories
    short_games = (df['duration_minutes'] < 25).sum()
    normal_games = ((df['duration_minutes'] >= 25) & (df['duration_minutes'] < 60)).sum()
    long_games = (df['duration_minutes'] >= 60).sum()
    
    print(f"\nDuration categories:")
    print(f"  Short games (<25 min): {short_games:,} ({short_games/len(df)*100:.1f}%)")
    print(f"  Normal games (25-60 min): {normal_games:,} ({normal_games/len(df)*100:.1f}%)")
    print(f"  Long games (>60 min): {long_games:,} ({long_games/len(df)*100:.1f}%)")
    
    # Create duration plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Duration histogram
    ax1.hist(df['duration_minutes'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Match Duration Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Duration (minutes)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(df['duration_minutes'].mean(), color='red', linestyle='--', label=f'Mean: {df["duration_minutes"].mean():.1f}min')
    ax1.legend()
    
    # Duration vs winrate
    duration_bins = pd.cut(df['duration_minutes'], bins=20)
    duration_winrate = df.groupby(duration_bins)['radiant_win'].agg(['mean', 'count'])
    duration_winrate = duration_winrate[duration_winrate['count'] >= 50]  # Filter small bins
    
    bin_centers = [interval.mid for interval in duration_winrate.index]
    ax2.plot(bin_centers, duration_winrate['mean'], marker='o', linewidth=2, markersize=6)
    ax2.set_title('Radiant Winrate by Match Duration', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Radiant Winrate')
    ax2.set_xlabel('Duration (minutes)')
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50% baseline')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Duration by rank tier
    if 'avg_rank_tier' in df.columns and df['avg_rank_tier'].notna().sum() > 0:
        rank_duration = df.groupby('avg_rank_tier')['duration_minutes'].agg(['mean', 'count'])
        rank_duration = rank_duration[rank_duration['count'] >= 100]
        
        ax3.bar(rank_duration.index, rank_duration['mean'], alpha=0.7, color='lightcoral')
        ax3.set_title('Average Duration by Rank Tier', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Duration (minutes)')
        ax3.set_xlabel('Average Rank Tier')
        ax3.tick_params(axis='x', rotation=45)
    
    # Duration over time (trend)
    daily_duration = df.groupby('date')['duration_minutes'].mean().tail(30)
    ax4.plot(daily_duration.index, daily_duration.values, marker='o', linewidth=2, markersize=4)
    ax4.set_title('Average Duration Trend (Last 30 Days)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Average Duration (minutes)')
    ax4.set_xlabel('Date')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def enhanced_hero_analysis(df, id_to_name, heroes_data):
    """Enhanced hero analysis with actual hero names and attributes"""
    print("\n" + "="*60)
    print("ENHANCED HERO ANALYSIS (with hero names)")
    print("="*60)
    
    # Flatten hero picks from both teams
    all_hero_picks = []
    radiant_picks = []
    dire_picks = []
    hero_winrates = {}
    
    for _, row in df.iterrows():
        if isinstance(row['radiant_team'], list) and isinstance(row['dire_team'], list):
            radiant_heroes = row['radiant_team']
            dire_heroes = row['dire_team']
            
            all_hero_picks.extend(radiant_heroes)
            all_hero_picks.extend(dire_heroes)
            radiant_picks.extend(radiant_heroes)
            dire_picks.extend(dire_heroes)
            
            # Track wins for each hero on radiant side
            for hero_id in radiant_heroes:
                if hero_id not in hero_winrates:
                    hero_winrates[hero_id] = {'wins': 0, 'games': 0}
                hero_winrates[hero_id]['games'] += 1
                if row['radiant_win']:
                    hero_winrates[hero_id]['wins'] += 1
            
            # Track losses for each hero on dire side
            for hero_id in dire_heroes:
                if hero_id not in hero_winrates:
                    hero_winrates[hero_id] = {'wins': 0, 'games': 0}
                hero_winrates[hero_id]['games'] += 1
                if not row['radiant_win']:
                    hero_winrates[hero_id]['wins'] += 1

    hero_counts = Counter(all_hero_picks)
    total_picks = len(all_hero_picks)
    total_matches = len(df)
    
    print(f"Total hero picks: {total_picks:,}")
    print(f"Unique heroes: {len(hero_counts)}")
    print(f"Picks per match: {total_picks / total_matches:.1f}")
    
    # Most popular heroes with names
    print(f"\n🏆 Most Popular Heroes:")
    for i, (hero_id, count) in enumerate(hero_counts.most_common(15), 1):
        hero_name = id_to_name.get(hero_id, f'Hero {hero_id}')
        pick_rate = count / total_picks * 100
        print(f"  {i:2d}. {hero_name:<20} {count:>6,} picks ({pick_rate:>5.2f}%)")
    
    # Calculate overall winrates
    hero_wr_data = []
    for hero_id, stats in hero_winrates.items():
        if stats['games'] >= 100:  # Minimum 100 games
            winrate = stats['wins'] / stats['games']
            hero_name = id_to_name.get(hero_id, f'Hero {hero_id}')
            hero_info = get_hero_info(hero_id, heroes_data)
            
            hero_wr_data.append({
                'hero_id': hero_id,
                'hero_name': hero_name,
                'games': stats['games'],
                'winrate': winrate,
                'primary_attr': hero_info['primary_attr'],
                'attack_type': hero_info['attack_type'],
                'roles': hero_info['roles']
            })
    
    if hero_wr_data:
        wr_df = pd.DataFrame(hero_wr_data)
        wr_df = wr_df.sort_values('winrate', ascending=False)
        
        print(f"\n🎯 Highest Winrate Heroes (min 100 games):")
        for i, row in wr_df.head(10).iterrows():
            print(f"  {i+1:2d}. {row['hero_name']:<20} {row['winrate']:>6.3f} ({row['games']:>4} games) [{row['primary_attr']}]")
        
        print(f"\n📉 Lowest Winrate Heroes:")
        for i, row in wr_df.tail(10).iterrows():
            print(f"  {len(wr_df)-(len(wr_df)-i-1)+1:2d}. {row['hero_name']:<20} {row['winrate']:>6.3f} ({row['games']:>4} games) [{row['primary_attr']}]")
        
        # Analyze by primary attribute
        if heroes_data:
            print(f"\n⚡ Winrate by Primary Attribute:")
            attr_stats = wr_df.groupby('primary_attr').agg({
                'winrate': ['mean', 'count'],
                'games': 'sum'
            }).round(3)
            attr_stats.columns = ['avg_winrate', 'hero_count', 'total_games']
            print(attr_stats)
            
            # Analyze by attack type
            print(f"\n🏹 Winrate by Attack Type:")
            attack_stats = wr_df.groupby('attack_type').agg({
                'winrate': ['mean', 'count'],
                'games': 'sum'
            }).round(3)
            attack_stats.columns = ['avg_winrate', 'hero_count', 'total_games']
            print(attack_stats)
        
        # Create enhanced visualizations
        create_enhanced_hero_plots(wr_df, hero_counts, id_to_name, heroes_data)
        
        return wr_df
    else:
        print("⚠️  Not enough data for winrate analysis")
        return None

def create_enhanced_hero_plots(wr_df, hero_counts, id_to_name, heroes_data):
    """Create enhanced hero analysis plots with names"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top picked heroes with names
    top_heroes = dict(list(hero_counts.most_common(15)))
    hero_names = [id_to_name.get(hid, f'Hero {hid}')[:12] for hid in top_heroes.keys()]
    
    bars1 = ax1.bar(range(len(top_heroes)), list(top_heroes.values()), 
                   alpha=0.8, color='lightblue', edgecolor='navy', linewidth=0.5)
    ax1.set_title('Top 15 Most Picked Heroes', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Heroes', fontsize=12)
    ax1.set_ylabel('Pick Count', fontsize=12)
    ax1.set_xticks(range(len(hero_names)))
    ax1.set_xticklabels(hero_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(top_heroes.values()):
        ax1.text(i, v + max(top_heroes.values())*0.01, f'{v:,}', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Winrate distribution by primary attribute
    if heroes_data and 'primary_attr' in wr_df.columns:
        attr_colors = {'str': 'red', 'agi': 'green', 'int': 'blue', 'all': 'purple'}
        for attr in wr_df['primary_attr'].unique():
            if attr in attr_colors:
                attr_data = wr_df[wr_df['primary_attr'] == attr]['winrate']
                ax2.hist(attr_data, alpha=0.6, label=f'{attr.upper()} ({len(attr_data)} heroes)',
                        color=attr_colors[attr], bins=15, edgecolor='black', linewidth=0.5)
        
        ax2.set_title('Winrate Distribution by Primary Attribute', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Winrate', fontsize=12)
        ax2.set_ylabel('Number of Heroes', fontsize=12)
        ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2, label='50% baseline')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # Fallback: simple winrate distribution
        ax2.hist(wr_df['winrate'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_title('Hero Winrate Distribution', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('Winrate', fontsize=12)
        ax2.set_ylabel('Number of Heroes', fontsize=12)
        ax2.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    
    # Winrate vs popularity scatter
    popularity_data = []
    winrate_data = []
    names_data = []
    colors_data = []
    
    attr_colors = {'str': 'red', 'agi': 'green', 'int': 'blue', 'all': 'purple'}
    
    for _, row in wr_df.iterrows():
        if row['hero_id'] in hero_counts:
            popularity_data.append(hero_counts[row['hero_id']])
            winrate_data.append(row['winrate'])
            names_data.append(row['hero_name'])
            colors_data.append(attr_colors.get(row['primary_attr'], 'gray'))
    
    scatter = ax3.scatter(popularity_data, winrate_data, alpha=0.7, c=colors_data, s=60, edgecolors='black', linewidth=0.5)
    ax3.set_title('Hero Winrate vs Popularity', fontsize=14, fontweight='bold', pad=20)
    ax3.set_xlabel('Pick Count', fontsize=12)
    ax3.set_ylabel('Winrate', fontsize=12)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax3.grid(True, alpha=0.3)
    
    # Add labels for extreme points
    for i, (x, y, name) in enumerate(zip(popularity_data, winrate_data, names_data)):
        if y > 0.55 or y < 0.45 or x > max(popularity_data) * 0.8:
            ax3.annotate(name[:8], (x, y), xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, alpha=0.8, weight='bold')
    
    # Top winrate heroes bar chart
    top_wr = wr_df.head(12)
    bars4 = ax4.barh(range(len(top_wr)), top_wr['winrate'], 
                    alpha=0.8, color='gold', edgecolor='darkorange', linewidth=0.5)
    ax4.set_title('Top 12 Highest Winrate Heroes', fontsize=14, fontweight='bold', pad=20)
    ax4.set_xlabel('Winrate', fontsize=12)
    ax4.set_yticks(range(len(top_wr)))
    ax4.set_yticklabels([name[:15] for name in top_wr['hero_name']])
    ax4.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add winrate labels
    for i, (wr, games) in enumerate(zip(top_wr['winrate'], top_wr['games'])):
        ax4.text(wr + 0.005, i, f'{wr:.3f} ({games})', 
                va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../results/enhanced_hero_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def hero_synergy_analysis(df, id_to_name, min_games=50):
    """Analyze hero synergies (basic version)"""
    print(f"\n🤝 Hero Synergy Analysis (min {min_games} games together)")
    print("="*60)
    
    synergy_stats = {}
    
    for _, row in df.iterrows():
        if isinstance(row['radiant_team'], list) and len(row['radiant_team']) == 5:
            team = sorted(row['radiant_team'])
            won = row['radiant_win']
            
            # Check all pairs in the team
            for i in range(len(team)):
                for j in range(i+1, len(team)):
                    pair = (team[i], team[j])
                    if pair not in synergy_stats:
                        synergy_stats[pair] = {'wins': 0, 'games': 0}
                    synergy_stats[pair]['games'] += 1
                    if won:
                        synergy_stats[pair]['wins'] += 1
        
        # Same for dire team
        if isinstance(row['dire_team'], list) and len(row['dire_team']) == 5:
            team = sorted(row['dire_team'])
            won = not row['radiant_win']
            
            for i in range(len(team)):
                for j in range(i+1, len(team)):
                    pair = (team[i], team[j])
                    if pair not in synergy_stats:
                        synergy_stats[pair] = {'wins': 0, 'games': 0}
                    synergy_stats[pair]['games'] += 1
                    if won:
                        synergy_stats[pair]['wins'] += 1
    
    # Filter and calculate winrates
    good_synergies = []
    for pair, stats in synergy_stats.items():
        if stats['games'] >= min_games:
            winrate = stats['wins'] / stats['games']
            hero1_name = id_to_name.get(pair[0], f'Hero {pair[0]}')
            hero2_name = id_to_name.get(pair[1], f'Hero {pair[1]}')
            good_synergies.append({
                'hero1': hero1_name,
                'hero2': hero2_name,
                'winrate': winrate,
                'games': stats['games']
            })
    
    good_synergies.sort(key=lambda x: x['winrate'], reverse=True)
    
    print(f"Found {len(good_synergies)} hero pairs with {min_games}+ games together")
    
    if good_synergies:
        print(f"\n🔥 Best Synergies:")
        for i, synergy in enumerate(good_synergies[:10], 1):
            print(f"  {i:2d}. {synergy['hero1']:<15} + {synergy['hero2']:<15} "
                  f"{synergy['winrate']:.3f} ({synergy['games']} games)")
        
        print(f"\n💔 Worst Synergies:")
        for i, synergy in enumerate(good_synergies[-10:], 1):
            print(f"  {i:2d}. {synergy['hero1']:<15} + {synergy['hero2']:<15} "
                  f"{synergy['winrate']:.3f} ({synergy['games']} games)")
    
    return good_synergies

def hero_analysis(df):
    """Legacy hero analysis (fallback when dotaconstants not available)"""
    print("\n" + "="*60)
    print("BASIC HERO ANALYSIS")
    print("="*60)
    
    # Flatten hero picks from both teams
    all_hero_picks = []
    radiant_picks = []
    dire_picks = []
    
    for _, row in df.iterrows():
        if isinstance(row['radiant_team'], list) and isinstance(row['dire_team'], list):
            radiant_heroes = row['radiant_team']
            dire_heroes = row['dire_team']
            
            all_hero_picks.extend(radiant_heroes)
            all_hero_picks.extend(dire_heroes)
            radiant_picks.extend(radiant_heroes)
            dire_picks.extend(dire_heroes)
    
    hero_counts = Counter(all_hero_picks)
    radiant_counts = Counter(radiant_picks)
    dire_counts = Counter(dire_picks)
    
    total_picks = len(all_hero_picks)
    total_matches = len(df)
    
    print(f"Total hero picks: {total_picks:,}")
    print(f"Unique heroes: {len(hero_counts)}")
    print(f"Expected picks per match: 10")
    print(f"Actual picks per match: {total_picks / total_matches:.1f}")
    
    print(f"\nMost popular heroes (overall):")
    for i, (hero_id, count) in enumerate(hero_counts.most_common(15), 1):
        pick_rate = count / total_picks * 100
        print(f"  {i:2d}. Hero {hero_id:3d}: {count:,} picks ({pick_rate:.2f}%)")
    
    # Calculate hero winrates (simplified - just looking at Radiant side)
    hero_winrates = {}
    for hero_id in hero_counts.keys():
        radiant_games = df[df['radiant_team'].apply(lambda x: isinstance(x, list) and hero_id in x)]
        if len(radiant_games) >= 100:  # Minimum 100 games for reliable stats
            winrate = radiant_games['radiant_win'].mean()
            hero_winrates[hero_id] = {
                'games': len(radiant_games),
                'winrate': winrate
            }
    
    print(f"\nHero winrates on Radiant side (min 100 games):")
    sorted_winrates = sorted(hero_winrates.items(), key=lambda x: x[1]['winrate'], reverse=True)
    print("Top 10 highest winrates:")
    for i, (hero_id, stats) in enumerate(sorted_winrates[:10], 1):
        print(f"  {i:2d}. Hero {hero_id:3d}: {stats['winrate']:.3f} ({stats['games']} games)")
    
    print("\nBottom 10 winrates:")
    for i, (hero_id, stats) in enumerate(sorted_winrates[-10:], 1):
        print(f"  {i:2d}. Hero {hero_id:3d}: {stats['winrate']:.3f} ({stats['games']} games)")
    
    # Create hero analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Most picked heroes
    top_heroes = dict(hero_counts.most_common(20))
    ax1.bar(range(len(top_heroes)), list(top_heroes.values()), alpha=0.7, color='lightblue')
    ax1.set_title('Top 20 Most Picked Heroes', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Hero ID')
    ax1.set_ylabel('Pick Count')
    ax1.set_xticks(range(len(top_heroes)))
    ax1.set_xticklabels(list(top_heroes.keys()), rotation=45)
    
    # Pick rate distribution
    pick_rates = [count / total_picks * 100 for count in hero_counts.values()]
    ax2.hist(pick_rates, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_title('Hero Pick Rate Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Pick Rate (%)')
    ax2.set_ylabel('Number of Heroes')
    ax2.axvline(np.mean(pick_rates), color='red', linestyle='--', label=f'Mean: {np.mean(pick_rates):.2f}%')
    ax2.legend()
    
    # Radiant vs Dire pick preference
    if len(radiant_counts) > 0 and len(dire_counts) > 0:
        common_heroes = set(radiant_counts.keys()) & set(dire_counts.keys())
        radiant_pref = []
        hero_ids = []
        
        for hero_id in list(common_heroes)[:20]:  # Top 20 common heroes
            rad_picks = radiant_counts[hero_id]
            dire_picks = dire_counts[hero_id]
            total_hero_picks = rad_picks + dire_picks
            if total_hero_picks >= 100:  # Minimum threshold
                radiant_pref.append(rad_picks / total_hero_picks * 100)
                hero_ids.append(hero_id)
        
        if radiant_pref:
            ax3.bar(range(len(radiant_pref)), radiant_pref, alpha=0.7, color='coral')
            ax3.set_title('Radiant Pick Preference by Hero', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Hero ID')
            ax3.set_ylabel('% Picked on Radiant')
            ax3.set_xticks(range(len(hero_ids)))
            ax3.set_xticklabels(hero_ids, rotation=45)
            ax3.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='50% baseline')
            ax3.legend()
    
    # Hero winrate distribution
    if hero_winrates:
        winrates = [stats['winrate'] for stats in hero_winrates.values()]
        ax4.hist(winrates, bins=20, alpha=0.7, color='gold', edgecolor='black')
        ax4.set_title('Hero Winrate Distribution (Radiant)', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Winrate')
        ax4.set_ylabel('Number of Heroes')
        ax4.axvline(np.mean(winrates), color='red', linestyle='--', label=f'Mean: {np.mean(winrates):.3f}')
        ax4.axvline(0.5, color='black', linestyle='--', alpha=0.7, label='50% baseline')
        ax4.legend()
    
    plt.tight_layout()
    plt.savefig('../results/hero_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def data_quality_check(df):
    """Check data quality and potential issues"""
    print("\n" + "="*60)
    print("DATA QUALITY CHECK")
    print("="*60)
    
    issues = []
    
    # Check for missing team data
    missing_radiant = df['radiant_team'].isna().sum()
    missing_dire = df['dire_team'].isna().sum()
    if missing_radiant > 0 or missing_dire > 0:
        issues.append(f"Missing team data: {missing_radiant} radiant, {missing_dire} dire")
    
    # Check team sizes
    valid_teams = 0
    invalid_teams = 0
    for _, row in df.head(1000).iterrows():  # Sample check
        if isinstance(row['radiant_team'], list) and isinstance(row['dire_team'], list):
            if len(row['radiant_team']) == 5 and len(row['dire_team']) == 5:
                valid_teams += 1
            else:
                invalid_teams += 1
    
    valid_team_rate = valid_teams / min(1000, len(df)) * 100
    print(f"Valid team composition rate (sample): {valid_team_rate:.1f}%")
    if invalid_teams > 0:
        print(f"Invalid team compositions found: {invalid_teams}")
    
    # Check for duplicate matches
    duplicates = df['match_id'].duplicated().sum()
    if duplicates > 0:
        issues.append(f"Duplicate match IDs: {duplicates}")
    
    # Check time consistency
    if df['start_time'].min() <= 0:
        issues.append("Invalid timestamps detected")
    
    # Check duration reasonableness
    very_short = (df['duration'] < 300).sum()  # <5 minutes
    very_long = (df['duration'] > 7200).sum()   # >2 hours
    print(f"Potentially problematic durations:")
    print(f"  Very short (<5 min): {very_short} ({very_short/len(df)*100:.2f}%)")
    print(f"  Very long (>2 hours): {very_long} ({very_long/len(df)*100:.2f}%)")
    
    if issues:
        print(f"\nData quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\nNo major data quality issues detected!")
    
    # Summary stats
    print(f"\nDataset summary:")
    print(f"  Total matches: {len(df):,}")
    print(f"  Date range: {pd.to_datetime(df['start_time'], unit='s').min().date()} to {pd.to_datetime(df['start_time'], unit='s').max().date()}")
    print(f"  Estimated total heroes picked: {len(df) * 10:,}")
    print(f"  Average match duration: {df['duration'].mean()/60:.1f} minutes")

def main():
    """Main exploration function"""
    print("🎮 Dota 2 Dataset Explorer")
    print("=" * 60)
    
    # Create results directory
    import os
    os.makedirs('../results', exist_ok=True)
    
    # Load hero constants
    id_to_name, name_to_id, heroes_data = load_hero_constants()
    has_constants = bool(id_to_name)
    
    # Load data
    try:
        df = load_data('../data/public_matches_combined_2000k.json')
    except FileNotFoundError:
        print("❌ Error: Could not find '../data/public_matches_combined_2000k.json'")
        print("Please make sure your data file is in the correct location.")
        return
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Run all analyses
    try:
        basic_info(df)
        data_quality_check(df)
        time_analysis(df)
        game_mode_analysis(df)
        winrate_analysis(df)
        duration_analysis(df)
        
        # Choose hero analysis based on constants availability
        if has_constants:
            print("\n🎯 Using enhanced hero analysis with dotaconstants!")
            wr_df = enhanced_hero_analysis(df, id_to_name, heroes_data)
            if wr_df is not None:
                synergies = hero_synergy_analysis(df, id_to_name, min_games=30)
        else:
            print("\n⚠️  Using basic hero analysis (dotaconstants not available)")
            hero_analysis(df)
        
        print("\n" + "="*60)
        print("✅ EXPLORATION COMPLETE!")
        print("="*60)
        print("📊 Results saved to ../results/ directory")
        
        if has_constants:
            print("🎯 Enhanced analysis complete with hero names!")
        else:
            print("💡 For enhanced analysis with hero names:")
            print("   1. Ensure dotaconstants submodule is in libs/dotaconstants/")
            print("   2. Run: cd libs/dotaconstants && npm install && npm run build")
        
        print("\n🚀 Next steps:")
        print("  1. Feature engineering (one-hot encode heroes)")
        print("  2. Train/validation/test split")
        print("  3. Model selection and training")
        print("  4. Hyperparameter tuning")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()