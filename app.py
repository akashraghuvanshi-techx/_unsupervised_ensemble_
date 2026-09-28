import streamlit as st
import joblib
import pandas as pd
from xgboost import XGBRegressor


scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca_22.pkl")

model = XGBRegressor()
model.load_model("xgb_model.json")



st.title("Football Player Rating Prediction")

st.header("Player Details")



potential = st.number_input(
    "Potential",
    min_value=0,
    max_value=100,
    value=70,
    step=1
)

crossing = st.number_input(
    "Crossing",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

finishing = st.number_input(
    "Finishing",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

dribbling = st.number_input(
    "Dribbling",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

ball_control = st.number_input(
    "Ball Control",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

short_passing = st.number_input(
    "Short Passing",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

interceptions = st.number_input(
    "Interceptions",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

marking = st.number_input(
    "Marking",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)

standing_tackle = st.number_input(
    "Standing Tackle",
    min_value=0,
    max_value=100,
    value=50,
    step=1
)



preferred_foot = st.selectbox(
    "Preferred Foot",
    ["right", "left"]
)

attacking_work_rate = st.selectbox(
    "Attacking Work Rate",
    ["low", "medium", "high"]
)

defensive_work_rate = st.selectbox(
    "Defensive Work Rate",
    ["low", "medium", "high"]
)



if st.button("Predict Rating"):

    
    # Feature Engineering
   

    overall_skill_score = (
        crossing
        + finishing
        + dribbling
        + ball_control
        + short_passing
    ) / 5

    defensive_skill_score = (
        marking
        + standing_tackle
        + interceptions
    ) / 3


    # Show engineered features
    st.write("Overall Skill Score:", round(overall_skill_score, 2))
    st.write("Defensive Skill Score:", round(defensive_skill_score, 2))


   

    data = {
        "potential": potential,
        "crossing": crossing,
        "finishing": finishing,
        "heading_accuracy": 50,
        "short_passing": short_passing,
        "volleys": 50,
        "dribbling": dribbling,
        "curve": 50,
        "free_kick_accuracy": 50,
        "long_passing": 50,
        "ball_control": ball_control,
        "acceleration": 50,
        "sprint_speed": 50,
        "agility": 50,
        "reactions": 50,
        "balance": 50,
        "shot_power": 50,
        "jumping": 50,
        "stamina": 50,
        "strength": 50,
        "long_shots": 50,
        "aggression": 50,
        "interceptions": interceptions,
        "positioning": 50,
        "vision": 50,
        "penalties": 50,
        "marking": marking,
        "standing_tackle": standing_tackle,
        "sliding_tackle": 50,
        "gk_diving": 50,
        "gk_handling": 50,
        "gk_kicking": 50,
        "gk_positioning": 50,
        "gk_reflexes": 50,

        # Feature Engineering
        "overall_skill_score": overall_skill_score,
        "defensive_skill_score": defensive_skill_score,

        # One-hot encoding
        "preferred_foot_right": 1 if preferred_foot == "right" else 0,

        "attacking_work_rate_low":
            1 if attacking_work_rate == "low" else 0,

        "attacking_work_rate_medium":
            1 if attacking_work_rate == "medium" else 0,

        "defensive_work_rate_low":
            1 if defensive_work_rate == "low" else 0,

        "defensive_work_rate_medium":
            1 if defensive_work_rate == "medium" else 0
    }


    input_data = pd.DataFrame([data])


    
    input_data = input_data[scaler.feature_names_in_]


    scaled_data = scaler.transform(input_data)



    pca_data = pca.transform(scaled_data)



    prediction = model.predict(pca_data)


    st.success(
        f"Predicted Overall Rating: {prediction[0]:.2f}"
    )