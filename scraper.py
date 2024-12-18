#%%
import requests, json
from pprint import pprint
import pandas as pd
#%%

pd.set_option('display.max_columns', None)

# base url for all FPL API endpoints
base_url = 'https://fantasy.premierleague.com/api/'

# get data from bootstrap-static endpoint
r = requests.get(base_url+'bootstrap-static/').json()

# show the top level fields
pprint(r, indent=2, depth=1, compact=True)

#%%
# get player data from 'elements' field
players = r['elements']


# create players dataframe
players = pd.json_normalize(r['elements'])

# show some information about first five players
players[['id', 'web_name', 'team', 'element_type']].head()

# create teams dataframe
teams = pd.json_normalize(r['teams'])

teams.head()
# %%

# get position information from 'element_types' field
positions = pd.json_normalize(r['element_types'])

positions.head()
# %%

# join players to teams
df = pd.merge(
    left=players,
    right=teams,
    left_on='team',
    right_on='id'
)

# show joined result
df[['first_name', 'second_name', 'name']].head()
# %%
# join player positions
df = df.merge(
    positions,
    left_on='element_type',
    right_on='id'
)

# rename columns
df = df.rename(
    columns={'name':'team_name', 'singular_name':'position_name'}
)

# show result
df[
    ['first_name', 'second_name', 'team_name', 'position_name']
].head()
# %%
base_url = 'https://fantasy.premierleague.com/api/'
r = requests.get(base_url + 'element-summary/4/').json()

# show top-level fields for player summary

pprint(r, depth=1)
# %%

# %%
# show data for first gameweek
'''
fixtures contains upcoming fixture information
history contains previous gameweek player scores
history_past provides summary of previous season totals
'''
# %%
def get_gameweek_history(player_id):
    '''get all gameweek info for a given player_id'''
    
    # send GET request to
    # https://fantasy.premierleague.com/api/element-summary/{PID}/
    r = requests.get(
            base_url + 'element-summary/' + str(player_id) + '/'
    ).json()
    
    # extract 'history' data from response into dataframe
    df = pd.json_normalize(r['history'])
    
    return df


# show player #4's gameweek history
get_gameweek_history(4)[
    [
        'total_points',
        'minutes',
        'goals_scored',
        'assists'
    ]
].head()
# %%

# select columns of interest from players df
players = players[
    ['id', 'first_name', 'second_name', 'web_name', 'team',
     'element_type']
]

# join team name
players = players.merge(
    teams[['id', 'name']],
    left_on='team',
    right_on='id',
    suffixes=['_player', None]
).drop(['team', 'id'], axis=1)

# join player positions
players = players.merge(
    positions[['id', 'singular_name_short']],
    left_on='element_type',
    right_on='id'
).drop(['element_type', 'id'], axis=1)

players.head()
# %%

def get_prev_season_stats(player_id):
    '''get all gameweek info for a given player_id'''
    
    # send GET request to
    # https://fantasy.premierleague.com/api/element-summary/{PID}/
    r = requests.get(
            base_url + 'element-summary/' + str(player_id) + '/'
    ).json()
    
    # extract 'history' data from response into dataframe
    df = pd.json_normalize(r['history_past'])
    
    return df


# show player #4's gameweek history
get_gameweek_history(9)[
    [   
        'total_points',
        'minutes',
        'goals_scored',
        'assists'
    ]
]

# %%
players
# %%
