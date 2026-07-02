import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# Page Config
# ============================================================
st.set_page_config(page_title="Cricket Player Performance Predictor", page_icon="🏏", layout="centered")

# ============================================================
# Load Model & Supporting Files
# ============================================================
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('le_dict.pkl', 'rb') as f:
        le_dict = pickle.load(f)
    with open('feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)
    player_history = pd.read_csv('player_history.csv')
    return model, scaler, le_dict, feature_cols, player_history

model, scaler, le_dict, feature_cols, player_history = load_artifacts()

player_col = 'batter'
target_col = 'runs_scored'
date_col = 'date'

# ============================================================
# App Title
# ============================================================
st.title("🏏 Cricket Player Performance Predictor")
st.markdown("PSL + IPL + International T20 + ODI data se train kiya gaya model — player ki previous performance ke hisaab se **next match** ka predicted score dekhein.")
st.divider()

# ============================================================
# User Inputs
# ============================================================
players_list = sorted(player_history[player_col].unique().tolist())
venues_list = sorted(le_dict['venue'].classes_.tolist())
teams_list = sorted(le_dict['batting_team'].classes_.tolist())
formats_list = sorted(le_dict['format'].classes_.tolist())

col1, col2 = st.columns(2)

with col1:
    player_name = st.selectbox("Player Chunein", players_list)
    batting_team = st.selectbox("Batting Team", teams_list)
    match_format = st.selectbox("Match Format", formats_list)

with col2:
    opponent_team = st.selectbox("Opponent (Bowling) Team", teams_list)
    venue_name = st.selectbox("Venue", venues_list)

# ============================================================
# Prediction Function
# ============================================================
def predict_next_match(player_name, opponent_team, venue_name, batting_team, match_format):
    player_hist = player_history[player_history[player_col] == player_name].sort_values(date_col)

    if player_hist.empty:
        return None, None

    latest = player_hist.iloc[-1]

    input_dict = {
        'venue_enc': le_dict['venue'].transform([venue_name])[0] if venue_name in le_dict['venue'].classes_ else 0,
        'batting_team_enc': le_dict['batting_team'].transform([batting_team])[0] if batting_team in le_dict['batting_team'].classes_ else 0,
        'bowling_team_enc': le_dict['bowling_team'].transform([opponent_team])[0] if opponent_team in le_dict['bowling_team'].classes_ else 0,
        'batter_enc': le_dict[player_col].transform([player_name])[0],
        'format_enc': le_dict['format'].transform([match_format])[0] if match_format in le_dict['format'].classes_ else 0,
        'prev_avg_3': player_hist[target_col].tail(3).mean(),
        'prev_avg_5': player_hist[target_col].tail(5).mean(),
        'prev_avg_10': player_hist[target_col].tail(10).mean(),
        'prev_max_5': player_hist[target_col].tail(5).max(),
        'career_avg': player_hist[target_col].mean(),
        'career_max': player_hist[target_col].max(),
        'last_match_runs': latest[target_col],
        'last_match_sr': latest['strike_rate'],
        'career_sr': player_hist['strike_rate'].mean(),
        'matches_played': latest['matches_played'] + 1,
        'form_std_5': player_hist[target_col].tail(5).std()
    }

    input_df = pd.DataFrame([input_dict])[feature_cols]
    input_df = input_df.fillna(0)
    input_scaled = scaler.transform(input_df)

    raw_prediction = model.predict(input_scaled)[0]
    # Integer mein round karna, negative score possible nahi
    final_prediction = int(round(max(raw_prediction, 0)))

    stats = {
        'Career Average': round(player_hist[target_col].mean(), 1),
        'Career Best': int(player_hist[target_col].max()),
        'Last 5 Matches Average': round(player_hist[target_col].tail(5).mean(), 1),
        'Last Match Runs': int(latest[target_col]),
        'Career Strike Rate': round(player_hist['strike_rate'].mean(), 1),
        'Total Matches Played': int(latest['matches_played'] + 1)
    }
    return final_prediction, stats

# ============================================================
# Predict Button
# ============================================================
if st.button("🔮 Predict Next Match Score", use_container_width=True):
    prediction, stats = predict_next_match(player_name, opponent_team, venue_name, batting_team, match_format)

    if prediction is None:
        st.error("Is player ka data nahi mila.")
    else:
        st.divider()
        st.subheader(f"Predicted Score for {player_name}")
        st.metric(label="Predicted Runs (Next Match)", value=f"{prediction} runs")

        st.markdown("### Player ki Career Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Career Avg", stats['Career Average'])
        c2.metric("Career Best", stats['Career Best'])
        c3.metric("Total Matches", stats['Total Matches Played'])

        c4, c5, c6 = st.columns(3)
        c4.metric("Last 5 Avg", stats['Last 5 Matches Average'])
        c5.metric("Last Match", stats['Last Match Runs'])
        c6.metric("Career SR", stats['Career Strike Rate'])

st.divider()
st.caption("Note: Model PSL, IPL, International T20, aur ODI data par train hai. Prediction historical pattern par based hai, guaranteed result nahi hai.")
