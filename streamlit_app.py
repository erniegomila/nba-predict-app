import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import subprocess

@st.cache_data
def load_data():
    file_path = 'line_score.csv'
    game_data = pd.read_csv(file_path)
    game_data['game_date_est'] = pd.to_datetime(game_data['game_date_est'])
    game_data['season'] = game_data['game_date_est'].apply(
        lambda x: f"{x.year}-{x.year + 1}" if x.month >= 10 else f"{x.year - 1}-{x.year}"
    )
    return game_data

game_data = load_data()

@st.cache_data
def compute_team_stats(df):
    stats = []
    team_stats = {}
    for _, row in df.sort_values('game_date_est').iterrows():
        date = row['game_date_est']
        home_team = row['team_id_home']
        away_team = row['team_id_away']
        home_points = row['pts_home']
        away_points = row['pts_away']

        if date.month >= 10:
            current_season = f"{date.year}-{date.year + 1}"
        else:
            current_season = f"{date.year - 1}-{date.year}"

        for team, points_for, points_against, is_home in [
            (home_team, home_points, away_points, True),
            (away_team, away_points, home_points, False),
        ]:
            if team not in team_stats:
                team_stats[team] = {
                    'total_points': 0,
                    'total_points_against': 0,
                    'home_wins': 0,
                    'home_losses': 0,
                    'away_wins': 0,
                    'away_losses': 0,
                }

            stats_dict = team_stats[team]
            stats_dict['total_points'] += points_for
            stats_dict['total_points_against'] += points_against

            if is_home:
                if points_for > points_against:
                    stats_dict['home_wins'] += 1
                else:
                    stats_dict['home_losses'] += 1
            else:
                if points_for > points_against:
                    stats_dict['away_wins'] += 1
                else:
                    stats_dict['away_losses'] += 1

            stats.append({
                'date': date,
                'team': team,
                'season': current_season,
                'total_points': stats_dict['total_points'],
                'total_points_against': stats_dict['total_points_against'],
                'home_wins': stats_dict['home_wins'],
                'home_losses': stats_dict['home_losses'],
                'away_wins': stats_dict['away_wins'],
                'away_losses': stats_dict['away_losses'],
            })
    return pd.DataFrame(stats).drop_duplicates()

team_stats = compute_team_stats(game_data)

@st.cache_data
def train_model():
    features = [
        'home_total_points', 'home_total_points_against', 'home_home_wins',
        'home_home_losses', 'away_total_points', 'away_total_points_against',
        'away_home_wins', 'away_home_losses'
    ]

    processed_data = game_data.merge(
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

    processed_data = processed_data.drop(columns=['date_x', 'date_y'])
    processed_data['home_win'] = (processed_data['pts_home'] > processed_data['pts_away']).astype(int)

    X = processed_data[features]
    y = processed_data['home_win']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    return model, features

model, features = train_model()

def predict_game(team_abbreviation_home, team_abbreviation_away, game_date):
    home_team_data = game_data[game_data['team_abbreviation_home'] == team_abbreviation_home]
    away_team_data = game_data[game_data['team_abbreviation_away'] == team_abbreviation_away]

    if home_team_data.empty:
        st.error(f"No data available for the home team: {team_abbreviation_home}")
        return
    if away_team_data.empty:
        st.error(f"No data available for the away team: {team_abbreviation_away}")
        return

    home_team_id = home_team_data['team_id_home'].iloc[0]
    away_team_id = away_team_data['team_id_away'].iloc[0]

    season = f"{game_date.year}-{game_date.year + 1}" if game_date.month >= 10 else f"{game_date.year - 1}-{game_date.year}"
    season_games = game_data[
        (game_data['season'] == season) &
        (game_data['game_date_est'] < pd.Timestamp(game_date)) &
        (
            (game_data['team_id_home'] == home_team_id) |
            (game_data['team_id_away'] == home_team_id) |
            (game_data['team_id_home'] == away_team_id) |
            (game_data['team_id_away'] == away_team_id)
        )
    ]

    if season_games.empty:
        st.error(f"No data available for the selected teams in the season {season}.")
        return

    # Calculate cumulative stats for the home and away teams
    home_total_points = season_games[season_games['team_id_home'] == home_team_id]['pts_home'].sum() + \
                        season_games[season_games['team_id_away'] == home_team_id]['pts_away'].sum()

    home_total_points_against = season_games[season_games['team_id_home'] == home_team_id]['pts_away'].sum() + \
                                season_games[season_games['team_id_away'] == home_team_id]['pts_home'].sum()

    home_home_wins = len(season_games[season_games['team_id_home'] == home_team_id])
    home_home_losses = len(season_games[season_games['team_id_away'] == home_team_id])

    away_total_points = season_games[season_games['team_id_home'] == away_team_id]['pts_home'].sum() + \
                        season_games[season_games['team_id_away'] == away_team_id]['pts_away'].sum()

    away_total_points_against = season_games[season_games['team_id_home'] == away_team_id]['pts_away'].sum() + \
                                season_games[season_games['team_id_away'] == away_team_id]['pts_home'].sum()

    away_home_wins = len(season_games[season_games['team_id_home'] == away_team_id])
    away_home_losses = len(season_games[season_games['team_id_away'] == away_team_id])


    input_data = pd.DataFrame([{
        'home_total_points': home_total_points,
        'home_total_points_against': home_total_points_against,
        'home_home_wins': home_home_wins,
        'home_home_losses': home_home_losses,
        'away_total_points': away_total_points,
        'away_total_points_against': away_total_points_against,
        'away_home_wins': away_home_wins,
        'away_home_losses': away_home_losses,
    }])


    for feature in features:
        if feature not in input_data.columns:
            input_data[feature] = 0  

    # Make the prediction
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]  # Get probabilities for both classes


    result = f"{team_abbreviation_home} Wins" if prediction == 1 else f"{team_abbreviation_away} Wins"
    winning_team_probability = probabilities[1] if prediction == 1 else probabilities[0]


    st.markdown(
        f"""
        <div style="text-align: left; font-size: 30px; color: white; font-weight: bold;">
            Prediction: <span style="color: green;">{result}</span>
        </div>
        <div style="text-align: left; font-size: 20px; color: white;">
            Probability: {winning_team_probability:.2%} chance of winning
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display bar charts for points and wins/losses
    st.subheader("Points Scored and Points Against")
    points_data = pd.DataFrame({
        'Feature': [
            f"{team_abbreviation_home} Total Points",
            f"{team_abbreviation_home} Points Against",
            f"{team_abbreviation_away} Total Points",
            f"{team_abbreviation_away} Points Against",
        ],
        'Value': [
            home_total_points,
            home_total_points_against,
            away_total_points,
            away_total_points_against,
        ],
    })
    st.bar_chart(points_data.set_index('Feature'))

    st.subheader("Wins and Losses")
    wins_losses_data = pd.DataFrame({
        'Feature': [
            f"{team_abbreviation_home} Home Wins",
            f"{team_abbreviation_home} Home Losses",
            f"{team_abbreviation_away} Away Wins",
            f"{team_abbreviation_away} Away Losses",
        ],
        'Value': [
            home_home_wins,
            home_home_losses,
            away_home_wins,
            away_home_losses,
        ],
    })
    st.bar_chart(wins_losses_data.set_index('Feature'))


# Sidebar navigation
st.sidebar.title("Menu") 
home_button = st.sidebar.button("Home")
prediction_button = st.sidebar.button("Win/Loss Predictor")
data_insights_button = st.sidebar.button("Data Insights")
about_button = st.sidebar.button("About")
over_under_button = st.sidebar.button("Over/Under Predictor")
over_under_visuals_button = st.sidebar.button("Over/Under Visuals")

# Set default page if no button is pressed
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Update page state based on button click
if home_button:
    st.session_state.page = "Home"
elif prediction_button:
    st.session_state.page = "Prediction"
elif data_insights_button:
    st.session_state.page = "Data Insights"
elif about_button:
    st.session_state.page = "About"
elif over_under_button:
    st.session_state.page = "Over/Under Predictor"
elif over_under_visuals_button:
    st.session_state.page = "Over/Under Visuals"

# Render the selected page
if st.session_state.page == "Home":

    # Title with emoji for a vibrant look
    st.markdown("""
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);">
            <h1 style="color: white; font-family: 'Arial', sans-serif; margin-bottom: 0;">🏀 NBA Predictor App</h1>
            <strong style="font-size: 20px; color: white; margin-top: 5px;">Elevate your gaming bets with our predictions, powered by data science!</strong>
        </div>
    """, unsafe_allow_html=True)

    # Welcome message with improved layout
    st.markdown("""
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52); padding: 15px; border-radius: 15px; text-align: center; margin-top: 20px; border: 1px solid #ddd; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);">
            <h2 style="color: white; font-family: 'Verdana', sans-serif;">Welcome to the NBA Predictor App</h2>
            <strong style="font-size: 18px; color: white; font-family: 'Georgia', serif;">
                Your ultimate tool for predicting NBA outcomes, powered by data science!
            </strong>
        </div>
    """, unsafe_allow_html=True)

    # Subtitle with better spacing
    st.markdown("""
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52);padding: 15px; border-radius: 15px; text-align: center; border: 1px solid #ddd; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);margin-top: 30px; text-align: left;">
            <h3 style="color: #333; font-family: 'Helvetica', sans-serif;">📚 Intro to Data Science Final Project</h3>
            <Strong style="font-size: 16px; color: #444; line-height: 1.6;">
                <b>CAP 5768</b><br>
                <b>Instructor:</b> Dr. Juhàsz<br>
                <b>Schedule: Monday 5:00 PM - 7:40 PM</b> 
            </Strong>
        </div>
    """, unsafe_allow_html=True)

    # Team members with hover effect using CSS
    st.markdown("""
        <style>
            .team-member-list li {
                margin-bottom: 8px;
                font-size: 16px;
                color: #444;
                transition: color 0.3s ease;
            }
            .team-member-list li:hover {
                color: #0077b6;
            }
        </style>
        <div style="background: linear-gradient(to right, #ff7c7c, #ffae52);padding: 15px; border-radius: 15px; text-align: center; border: 1px solid #ddd; box-shadow: 0px 3px 6px rgba(0, 0, 0, 0.1);margin-top: 30px; text-align: left;">
            <h3 style="color: #333; font-family: 'Helvetica', sans-serif;">🌟 Team Members:</h3>
            <ul class="team-member-list" style="list-style: none; padding: 0;">
                <li><Strong>Ernesto Gomila</Strong></li>
                <li><Strong>Brandon Rodriguez</Strong></li>
                <li><Strong>Alfredo Cal</Strong></li>
                <li><Strong>Abel</Strong></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
        <div style="margin-top: 40px; text-align: center; color: #aaa; font-size: 14px;">
        Made with 🧠 by the NBA Predictor Team | Powered by Streamlit
         </div>
    """, unsafe_allow_html=True)

    # Apply background color for the entire app
    st.markdown("""
        <style>
            body {
                background-color: #f4f4f4; /* Light off-white background for readability */
            }
            .main-content {
                padding: 20px;
            }
        </style>
    """, unsafe_allow_html=True)


elif st.session_state.page == "Prediction":
    st.title("Make a Prediction")
    with st.form("prediction_form"):
        team_abbreviation_home = st.text_input("Enter Home Team Abbreviation:")
        team_abbreviation_away = st.text_input("Enter Away Team Abbreviation:")
        game_date = st.date_input("Select Game Date:")
        submit = st.form_submit_button("Predict")

    if submit:
        predict_game(team_abbreviation_home, team_abbreviation_away, game_date)

# Main app logic
if st.session_state.page == "Data Insights":

    # Load data
    def load_data():
        # Your data loading function (placeholder for actual data loading logic)
        line_score = pd.read_csv('line_score.csv')  # Update with actual data path
        common_player_info = pd.read_csv('common_player_info.csv')  # Update with actual data path
        draft_history = pd.read_csv('draft_history.csv')  # Update with actual data path
        return line_score, common_player_info, draft_history

    # Initialize Streamlit app
    st.title("Data Insights and Analysis")

    # Load data
    line_score, common_player_info, draft_history = load_data()

    # Preview datasets
    st.header("Dataset Previews")
    st.subheader("Line Score")
    st.dataframe(line_score.head())
    st.subheader("Common Player Info")
    st.dataframe(common_player_info.head())
    st.subheader("Draft History")
    st.dataframe(draft_history.head())

    # Standardize column names
    line_score.columns = line_score.columns.str.lower()
    common_player_info.columns = common_player_info.columns.str.lower()
    draft_history.columns = draft_history.columns.str.lower()

    # Analysis 1: Total Points Scored Per Team
    st.header("Analysis 1: Total Points Scored Per Team")
    if 'team_nickname_home' in line_score.columns and 'pts_home' in line_score.columns and \
            'team_nickname_away' in line_score.columns and 'pts_away' in line_score.columns:
        total_points_home = line_score.groupby('team_nickname_home')['pts_home'].sum().reset_index()
        total_points_home.columns = ['team', 'total_points']
        total_points_away = line_score.groupby('team_nickname_away')['pts_away'].sum().reset_index()
        total_points_away.columns = ['team', 'total_points']
        total_points = pd.concat([total_points_home, total_points_away]).groupby('team').sum().reset_index()

        # Create a bar chart to display the total points
        fig = px.bar(
            total_points,
            x='team',
            y='total_points',
            title='Total Points Scored Per Team',
            labels={'total_points': 'Total Points'},
            color='total_points',
        )
        st.plotly_chart(fig)
    else:
        st.error("Relevant columns for total points analysis are missing in the dataset.")

    # Insight 2: Top Scoring Games of All Time (Pie Chart)
    st.header("Insight 2: Top Scoring Games of All Time")
    try:
        # Check if required columns are present
        if {'game_id', 'pts_home', 'pts_away'}.issubset(line_score.columns):
            # Calculate total points for each game
            line_score['total_points'] = line_score['pts_home'] + line_score['pts_away']

            # Get the top 10 highest-scoring games
            top_games = line_score.nlargest(10, 'total_points')

            # Create a pie chart to display the distribution of total points
            fig = px.pie(
                top_games,
                names='game_id',
                values='total_points',
                title="Top Scoring Games of All Time",
                labels={'game_id': 'Game ID', 'total_points': 'Total Points'},
                hover_data={'total_points': True},
                color_discrete_sequence=px.colors.sequential.RdBu,
            )

            # Improve hover information
            fig.update_traces(hovertemplate="Game ID: %{label}<br>Total Points: %{value}")

            # Display the chart
            st.plotly_chart(fig)
        else:
            # Error if required columns are missing
            st.error("Columns required for Insight 2 are missing. Ensure `game_id`, `pts_home`, and `pts_away` are present.")
    except Exception as e:
        # Catch and display any errors
        st.error(f"An error occurred while processing Insight 2: {e}")

    # Analysis 3: Top Teams by Wins
    st.header("Analysis 3: Top Teams by Wins")
    if 'team_wins_losses_home' in line_score.columns and 'team_nickname_home' in line_score.columns and \
            'team_wins_losses_away' in line_score.columns:

        # Clean and convert win/loss data
        line_score['wins_home'] = (
            line_score['team_wins_losses_home']
            .str.split('-')
            .str[0]
            .apply(lambda x: int(x) if x.isdigit() else None)
        )
        line_score['wins_away'] = (
            line_score['team_wins_losses_away']
            .str.split('-')
            .str[0]
            .apply(lambda x: int(x) if x.isdigit() else None)
        )

        # Aggregate wins
        total_wins = (
            line_score.groupby('team_nickname_home')['wins_home']
            .sum()
            .reset_index()
            .rename(columns={'team_nickname_home': 'team', 'wins_home': 'total_wins'})
        )

        # Create a bar chart to display total wins
        fig = px.bar(
            total_wins,
            x='team',
            y='total_wins',
            title="Top Teams by Total Wins",
            labels={'total_wins': 'Wins'},
            color='total_wins',
        )
        st.plotly_chart(fig)
    else:
        st.error("Relevant columns to analyze wins are missing.")

    # Insight 4: Average Points per Game by Team
    st.header("Insight 4: Average Points per Game by Team")
    try:
        required_columns_insight2 = {'team_nickname_home', 'pts_home', 'game_id'}
        missing_columns_insight2 = required_columns_insight2 - set(line_score.columns)
        if not missing_columns_insight2:
            team_stats = (
                line_score.groupby('team_nickname_home')
                .agg(total_points=('pts_home', 'sum'), total_games=('game_id', 'nunique'))
                .reset_index()
            )
            team_stats['avg_points_per_game'] = team_stats['total_points'] / team_stats['total_games']

            # Create a bar chart for average points per game
            fig = px.bar(
                team_stats.sort_values(by='avg_points_per_game', ascending=False),
                x='team_nickname_home',
                y='avg_points_per_game',
                title="Average Points per Game by Team",
                labels={'team_nickname_home': 'Team', 'avg_points_per_game': 'Avg Points/Game'},
                color='avg_points_per_game',
            )
            st.plotly_chart(fig)
        else:
            st.error(f"Columns required for Insight 3 are missing: {missing_columns_insight2}")
    except Exception as e:
        st.error(f"An error occurred while processing Insight 3: {e}")

    # Insight 5: Win/Loss Analysis by Team
    st.header("Insight 5: Win/Loss Analysis by Team")
    try:
        if {'team_nickname_home', 'team_wins_losses_home'}.issubset(line_score.columns):
            win_loss_data = line_score[['team_nickname_home', 'team_wins_losses_home']].copy()

            # Split the win/loss column and handle conversion properly
            win_loss_data[['wins', 'losses']] = win_loss_data['team_wins_losses_home'].str.split('-', expand=True)

            # Convert to numeric, coercing errors to NaN
            win_loss_data['wins'] = pd.to_numeric(win_loss_data['wins'], errors='coerce')
            win_loss_data['losses'] = pd.to_numeric(win_loss_data['losses'], errors='coerce')

            # Drop rows with NaN values
            win_loss_data = win_loss_data.dropna(subset=['wins', 'losses'])

            # Calculate win/loss ratio
            win_loss_data['win_loss_ratio'] = win_loss_data['wins'] / (win_loss_data['wins'] + win_loss_data['losses'])
            avg_ratios = win_loss_data.groupby('team_nickname_home')['win_loss_ratio'].mean().reset_index()

            # Drop rows with NaN values in win/loss ratio
            avg_ratios = avg_ratios.dropna(subset=['win_loss_ratio'])

            # Create a scatter plot for Win/Loss Ratio by Team
            fig = px.scatter(
                avg_ratios,
                x='team_nickname_home',
                y='win_loss_ratio',
                size='win_loss_ratio',
                title="Win/Loss Ratio by Team",
                labels={'team_nickname_home': 'Team', 'win_loss_ratio': 'Win/Loss Ratio'},
                color='win_loss_ratio',
            )
            st.plotly_chart(fig)
        else:
            st.error("Columns required for Insight 5 are missing. Ensure `team_nickname_home` and `team_wins_losses_home` are present.")
    except Exception as e:
        st.error(f"An error occurred while processing Insight 5: {e}")

#About tab
elif st.session_state.page == "About": 
    st.title("About") 
    st.write("This app predicts NBA game outcomes and helps the vizualization of NBA data.")
    st.write("We decided to use StreamLit for its capabilities in presenting data visually. Our data comes from online sources including kaggle. The data itself is stored in sql tables, and hosted through MySQL.")
    st.title("Transferring the Data")
    st.write("First, we created an AWS RDS instance so that we could create the database in the cloud - for more efficient collaboration between the team members. We used Visual Studio Code to write a simple python script that will connect to our database and using pandas and pymysql we were able to locate the csv files we downloaded from Kaggle.")
    st.write("Our ETL process was our script looping through the local folder and for every csv file, it created a table and added the records to said table. We then installed mySQL Workbench and connected it to our cloud database in order to verify that the information was imported correctly.")
    st.title("Data Handling")
    st.write("Handling Missing Data: For missing values, we used imputation strategies where applicable")
    st.write("Handling Outliers: We used box plots to detect and remove extreme outliers, particularly for game scores and individual player statistics.") 
    st.write("Feature Scaling: Player statistics, such as points and assists, are being normalized to improve analysis consistency.")    
    st.write("There are two main functionalities regarding Data Science in this app:")
    st.title("Win/Loss Predictor")
    st.write("By adding in two team abbreviations, we can make a prediction on who will win a game on that given date. Unfortunately, due to the dataset not being fullt updated, some recent matches past 2022 are unable to be predicted.")
    st.write("Two major data points that are used are team (seasonal) record along with total points for/against the team.")
    st.title("Data Insights")
    st.write("The Data Insights page consists of 3 dataset previews. There is a line score, which records every game since the beginning of the NBA.") 
    st.write("Along with the box scores, there is info for every player, along with a draft history. Further analysis for teams are included, which highlight further data science and predicitve modeling.")


elif st.session_state.page == "Over/Under Predictor":
    st.title("Over/Under Predictor")

    with st.form("over_under_form"):
        player = st.text_input("Enter Player Name:")
        team1 = st.text_input("Enter Team 1 Abbreviation (Player's Team):")
        team2 = st.text_input("Enter Team 2 Abbreviation (Opponent):")
        game_date = st.date_input("Enter Game Date:")
        points = st.number_input("Enter Points Threshold:", min_value=0, step=1, value=20)
        submit = st.form_submit_button("Predict Over/Under")

    if submit:
        if not all([player, team1, team2, game_date, points]):
            st.error("Please fill in all the fields.")
        else:
            # Construct the command to call your model
            command = f"python3 over_under.py predict --player \"{player}\" --team1 \"{team1}\" --team2 \"{team2}\" --date \"{game_date}\" --points {int(points)}"
            
            # Use subprocess to run the command
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Prediction Complete!")
                    st.text(result.stdout)
                else:
                    st.error("Error in prediction.")
                    st.text(result.stderr)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    st.subheader("Team Abbreviations")
    st.markdown("""
    Here are common NBA team abbreviations:
    - **Atlanta Hawks** → ATL
    - **Boston Celtics** → BOS
    - **Brooklyn Nets** → BKN
    - **Charlotte Hornets** → CHA
    - **Chicago Bulls** → CHI
    - **Cleveland Cavaliers** → CLE
    - **Dallas Mavericks** → DAL
    - **Denver Nuggets** → DEN
    - **Detroit Pistons** → DET
    - **Golden State Warriors** → GSW
    - **Houston Rockets** → HOU
    - **Indiana Pacers** → IND
    - **LA Clippers** → LAC
    - **Los Angeles Lakers** → LAL
    - **Memphis Grizzlies** → MEM
    - **Miami Heat** → MIA
    - **Milwaukee Bucks** → MIL
    - **Minnesota Timberwolves** → MIN
    - **New Orleans Pelicans** → NOP
    - **New York Knicks** → NYK
    - **Oklahoma City Thunder** → OKC
    - **Orlando Magic** → ORL
    - **Philadelphia 76ers** → PHI
    - **Phoenix Suns** → PHX
    - **Portland Trail Blazers** → POR
    - **Sacramento Kings** → SAC
    - **San Antonio Spurs** → SAS
    - **Toronto Raptors** → TOR
    - **Utah Jazz** → UTA
    - **Washington Wizards** → WAS
    """)

elif st.session_state.page == "Over/Under Visuals":
    st.title("Over/Under Visuals")
    st.markdown(
        "These visuals show the accuracy and feature importances for predicting over/under on sample thresholds for player points. "
        "Each section corresponds to predictions for a specific point threshold (10, 20, and 30)."
    )

    # 10-Point Section
    st.subheader("10-Point Threshold")
    st.image("over_under_visuals/10_kpi.png", caption="Model KPI's for 10-Point Threshold")
    st.image("over_under_visuals/10_accuracy.png", caption="Accuracy for 10-Point Threshold")
    st.markdown("This chart shows the prediction outcomes (correct and incorrect) for a 10-point threshold on the test set.")
    st.image("over_under_visuals/10_feature_imp.png", caption="Feature Importances for 10-Point Threshold")
    st.markdown("This chart shows the most important features contributing to predictions for a 10-point threshold."
                "In this case we see that for low points the team the player is on is more important for wether they will be over/under the low 10-point threshold.")

    # 20-Point Section
    st.subheader("20-Point Threshold")
    st.image("over_under_visuals/20_kpi.png", caption="Model KPI's for 20-Point Threshold")
    st.image("over_under_visuals/20_accuracy.png", caption="Accuracy for 20-Point Threshold")
    st.markdown("This chart shows the prediction outcomes (correct and incorrect) for a 20-point threshold on the test set.")
    st.image("over_under_visuals/20_feature_imp.png", caption="Feature Importances for 20-Point Threshold")
    st.markdown("This chart shows the most important features contributing to predictions for a 20-point threshold."
                "In this case we see that for medium points the the player; not the team, is on is more important for wether they will be over/under the low 10-point threshold."
                "This also gives us insight into the player the model is most confident/decisive in its prediction since the weights are high for these player names.")

    # 30-Point Section
    st.subheader("30-Point Threshold")
    st.image("over_under_visuals/30_kpi.png", caption="Model KPI's for 30-Point Threshold")
    st.image("over_under_visuals/30_accuracy.png", caption="Accuracy for 30-Point Threshold")
    st.markdown("This chart shows the prediction outcomes (correct and incorrect) for a 30-point threshold on the test set."
                "We see that the higher the point threshold it seems the more accurate the model is in its prediction.")
    st.image("over_under_visuals/30_feature_imp.png", caption="Feature Importances for 30-Point Threshold")
    st.markdown("This chart shows the most important features contributing to predictions for a 30-point threshold."
                "In this case we also see that for medium points the the player; not the team, is on is more important for wether they will be over/under the low 10-point threshold."
                "This also gives us insight into the player the model is most confident/decisive in its prediction since the weights are high for these player names.")