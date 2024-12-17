import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import nfl_data_py as nfl

@st.cache_data
def load_data():
    pbp_data = nfl.import_pbp_data([2022])
    return pbp_data

def preprocess_data(data):
    data = data.rename(columns={
        'qtr': 'quarter',
    })
    fourth_down_plays = data[data['down'] == 4]
    fourth_down_plays['go_for_it'] = fourth_down_plays['play_type'].apply(
        lambda x: 1 if x in ['run', 'pass'] else 0
    )
    features = fourth_down_plays[['yardline_100', 'ydstogo', 'quarter']]
    target = fourth_down_plays['go_for_it']
    return features, target

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

pbp_data = load_data()
X, y = preprocess_data(pbp_data)
model = train_model(X, y)

st.title("4th Down Decision Maker")
st.write("Depending on the situation, this model will guess whether you should go for it, or opt to kick a field goal")

st.sidebar.header("Your Game Parameters:")
yardline_100 = st.sidebar.slider("Yardline (distance to end zone)", 1, 30, 15)
ydstogo = st.sidebar.slider("Yards to Go for First Down", 1, yardline_100, 5)
quarter = st.sidebar.slider("Quarter", 2, 4, 4)

user_input = pd.DataFrame({
    'yardline_100': [yardline_100],
    'ydstogo': [ydstogo],
    'quarter': [quarter]
})

user_input = user_input[['yardline_100', 'ydstogo', 'quarter']]

prediction = model.predict(user_input)[0]

st.subheader("Game Situation:")
st.write(f"**Yardline:** {yardline_100} yards to end zone")
st.write(f"**Yards to Go:** {ydstogo} yards")
st.write(f"**Quarter:** {quarter}")

st.subheader("Suggestion:")
if prediction == 1:
    st.success("**Go for it!**")
else:
    st.warning("**Kick a field goal.**")
st.caption('Click the top left sidebar if on mobile :)')
