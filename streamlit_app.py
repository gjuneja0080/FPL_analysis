import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Player Performance Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("players.csv")
    numeric_cols = ['now_cost', 'goals_scored', 'assists', 'selected_by_percent', 'points_per_game', 'minutes']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

df = load_data()

st.title("Player Performance Dashboard")

st.sidebar.header("Filters")
teams = df['team'].dropna().unique().tolist()
positions = df['position'].dropna().unique().tolist()

selected_team = st.sidebar.selectbox("Select Team:", options=["All"]+teams)
selected_position = st.sidebar.selectbox("Select Position:", options=["All"]+positions)

filtered = df.copy()

if selected_team != "All":
    filtered = filtered[filtered['team'] == selected_team]

if selected_position != "All":
    filtered = filtered[filtered['position'] == selected_position]

st.subheader("Player Data")
st.dataframe(filtered.head(50)) 

st.markdown("### Summary Statistics")
st.write(filtered[['goals_scored','assists','points_per_game','minutes','now_cost','selected_by_percent']].describe())

st.markdown("### Top 10 Players by Goals Scored")
top_goals = filtered[['name','goals_scored','position','team']].dropna().sort_values('goals_scored', ascending=False).head(10)
fig_goals = px.bar(top_goals, x='name', y='goals_scored', color='team', title="Top 10 Goal Scorers")
st.plotly_chart(fig_goals, use_container_width=True)

st.markdown("### Cost vs Points per Game")
fig_scatter = px.scatter(filtered, x='now_cost', y='points_per_game', hover_data=['name','team','position'],
                         color='position', title="Cost vs Points per Game")
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("### Distribution of Minutes Played")
fig_minutes = px.histogram(filtered, x='minutes', nbins=20, title="Distribution of Player Minutes")
st.plotly_chart(fig_minutes, use_container_width=True)
