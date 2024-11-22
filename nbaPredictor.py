import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = '/Users/ernestogomila/Desktop/nba-predictor-app/line_score.csv' 
game_data = pd.read_csv(file_path)
game_data['game_date_est'] = pd.to_datetime(game_data['game_date_est'])

# Add a "season" column
game_data['season'] = game_data['game_date_est'].dt.year
game_data.loc[game_data['game_date_est'].dt.month >= 10, 'season'] += 1

# Function to compute team stats
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

# Compute stats
team_stats = compute_team_stats(game_data)

# Prepare game data with merged features
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

# Drop redundant columns
game_data.drop(columns=['date_x', 'date_y'], inplace=True)

# Create target variable
game_data['home_win'] = (game_data['pts_home'] > game_data['pts_away']).astype(int)

# Train-test split for model training
features = [
    'home_total_points', 'home_total_points_against', 'home_home_wins',
    'home_home_losses', 'away_total_points', 'away_total_points_against',
    'away_home_wins', 'away_home_losses'
]
X = game_data[features]
y = game_data['home_win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Function to determine the season based on the game date
def get_season(game_date):
    if game_date.month >= 10:
        return game_date.year + 1
    else:
        return game_date.year

# Function to predict game outcome based on season-specific data
def predict_game(home_team_id, away_team_id, game_date):
    # Determine the season
    season = get_season(game_date)
    
    # Filter data for the specific season
    season_start = pd.Timestamp(f"{season - 1}-10-01")
    season_end = pd.Timestamp(f"{season}-06-30")
    
    season_data = team_stats[
        (team_stats['date'] >= season_start) & (team_stats['date'] <= season_end)
    ]
    
    # If no data for the season
    if season_data.empty:
        return "We don't have enough data for this season to make a prediction."
    
    # Get stats for the home team up to the game date
    home_stats = season_data[
        (season_data['team'] == home_team_id) & (season_data['date'] < game_date)
    ].sort_values('date')
    if home_stats.empty:
        return "We don't have enough data for this season to make a prediction."
    home_stats = home_stats.iloc[-1]
    
    # Get stats for the away team up to the game date
    away_stats = season_data[
        (season_data['team'] == away_team_id) & (season_data['date'] < game_date)
    ].sort_values('date')
    if away_stats.empty:
        return "We don't have enough data for this season to make a prediction."
    away_stats = away_stats.iloc[-1]
    
    # Prepare input for prediction
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
    
    # Display feature values
    print("\nFeature values used for prediction:")
    print(input_data)
    
    # Predict the outcome
    prediction = model.predict(input_data)[0]
    return "Home Team Wins" if prediction == 1 else "Away Team Wins"

# Ask for user input
home_team_id = int(input("Enter the home team ID: "))
away_team_id = int(input("Enter the away team ID: "))
game_date_input = input("Enter the game date (YYYY-MM-DD): ")
game_date = pd.Timestamp(game_date_input)

# Make prediction
prediction = predict_game(home_team_id, away_team_id, game_date)
print(f"Prediction for the game: {prediction}")
