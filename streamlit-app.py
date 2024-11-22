import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("NBA Predictor App")

# Load the dataset
file_path = 'line_score.csv'  # Ensure this file is in the same directory as the Streamlit app
st.write("Loading dataset...")
try:
    game_data = pd.read_csv(file_path)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error(f"File not found at path: {file_path}. Please upload the file.")
    st.stop()

# Process the dataset
game_data['game_date_est'] = pd.to_datetime(game_data['game_date_est'])
game_data['season'] = game_data['game_date_est'].dt.year
game_data.loc[game_data['game_date_est'].dt.month >= 10, 'season'] += 1

# Function to compute team stats
@st.cache
def compute_team_stats(df):
    stats = []
    for season, season_data in df.groupby('season'):
        team_stats = {}
        for _, row in season_data.sort_values('game_date_est').iterrows():
            date = row['game_date_est']
            home_team = row['team_id_home']
            away_team = row['team_id_away']
            home_points = row['pts_home']
            away_points = row['pts_away']
            for team in [home_team, away_team]:
                if team not in team_stats:
                    team_stats[team] = {
                        'total_points': 0,
                        'total_points_against': 0,
                        'home_wins': 0,
                        'home_losses': 0,
                        'away_wins': 0,
                        'away_losses': 0
                    }
            home_stats = team_stats[home_team]
            home_stats['total_points'] += home_points
            home_stats['total_points_against'] += away_points
            if home_points > away_points:
                home_stats['home_wins'] += 1
            else:
                home_stats['home_losses'] += 1
            away_stats = team_stats[away_team]
            away_stats['total_points'] += away_points
            away_stats['total_points_against'] += home_points
            if away_points > home_points:
                away_stats['away_wins'] += 1
            else:
                away_stats['away_losses'] += 1
            stats.append({
                'date': date,
                'team': home_team,
                'total_points': home_stats['total_points'],
                'total_points_against': home_stats['total_points_against'],
                'home_wins': home_stats['home_wins'],
                'home_losses': home_stats['home_losses'],
                'away_wins': home_stats['away_wins'],
                'away_losses': home_stats['away_losses']
            })
            stats.append({
                'date': date,
                'team': away_team,
                'total_points': away_stats['total_points'],
                'total_points_against': away_stats['total_points_against'],
                'home_wins': away_stats['home_wins'],
                'home_losses': away_stats['home_losses'],
                'away_wins': away_stats['away_wins'],
                'away_losses': away_stats['away_losses']
            })
    return pd.DataFrame(stats).drop_duplicates()

team_stats = compute_team_stats(game_data)

# Train the model
st.write("Training the prediction model...")
features = [
    'home_total_points', 'home_total_points_against', 'home_home_wins',
    'home_home_losses', 'away_total_points', 'away_total_points_against',
    'away_home_wins', 'away_home_losses'
]
game_data = game_data.merge(
    team_stats.rename(columns={
        'team': 'team_id_home',
        'total_points': 'home_total_points',
        'total_points_against': 'home_total_points_against',
        'home_wins': 'home_home_wins',
        'home_losses': 'home_home_losses',
        'away_wins': 'home_away_wins',
        'away_losses': 'home_away_losses'
    }),
    left_on=['game_date_est', 'team_id_home'],
    right_on=['date', 'team_id_home'],
    how='left'
).merge(
    team_stats.rename(columns={
        'team': 'team_id_away',
        'total_points': 'away_total_points',
        'total_points_against': 'away_total_points_against',
        'home_wins': 'away_home_wins',
        'home_losses': 'away_home_losses',
        'away_wins': 'away_away_wins',
        'away_losses': 'away_away_losses'
    }),
    left_on=['game_date_est', 'team_id_away'],
    right_on=['date', 'team_id_away'],
    how='left'
)
game_data.drop(columns=['date_x', 'date_y'], inplace=True)
game_data['home_win'] = (game_data['pts_home'] > game_data['pts_away']).astype(int)

X = game_data[features]
y = game_data['home_win']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
st.success("Model trained successfully!")

# Prediction
st.header("Make a Prediction")
home_team_id = st.number_input("Enter Home Team ID", step=1, format="%d")
away_team_id = st.number_input("Enter Away Team ID", step=1, format="%d")
game_date = st.date_input("Enter Game Date")

if st.button("Predict Outcome"):
    def predict_game(home_team_id, away_team_id, game_date):
        # Determine season
        season = game_date.year if game_date.month < 10 else game_date.year + 1
        season_data = team_stats[
            (team_stats['date'] < game_date) &
            (team_stats['season'] == season)
        ]

        if season_data.empty:
            st.error("Not enough data for the selected season.")
            return

        home_stats = season_data[season_data['team'] == home_team_id].sort_values('date').iloc[-1]
        away_stats = season_data[season_data['team'] == away_team_id].sort_values('date').iloc[-1]

        input_data = pd.DataFrame([{
            'home_total_points': home_stats['total_points'],
            'home_total_points_against': home_stats['total_points_against'],
            'home_home_wins': home_stats['home_wins'],
            'home_home_losses': home_stats['home_losses'],
            'away_total_points': away_stats['total_points'],
            'away_total_points_against': away_stats['total_points_against'],
            'away_home_wins': away_stats['home_wins'],
            'away_home_losses': away_stats['home_losses']
        }])

        prediction = model.predict(input_data)[0]
        return "Home Team Wins" if prediction == 1 else "Away Team Wins"

    prediction = predict_game(home_team_id, away_team_id, pd.Timestamp(game_date))
    st.write(f"Prediction: {prediction}")
