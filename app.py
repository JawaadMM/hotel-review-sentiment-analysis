import streamlit as st
import pickle
import string

# Load the model and vectorizer
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Page title
st.title("Hotel Review Sentiment Analysis")
st.markdown("Enter your hotel review below:")

# Initialize session state
if "review_input" not in st.session_state:
    st.session_state.review_input = ""
if "sentiment_result" not in st.session_state:
    st.session_state.sentiment_result = ""
if "sentiment_label" not in st.session_state:
    st.session_state.sentiment_label = ""

# Reset input and result
def clear_input():
    st.session_state.review_input = ""
    st.session_state.sentiment_result = ""
    st.session_state.sentiment_label = ""
    st.session_state.review_box = ""

# Predict sentiment
def predict_sentiment():
    review = st.session_state.review_input.strip()
    if review:
        cleaned = review.lower().translate(str.maketrans('', '', string.punctuation))
        vec_input = vectorizer.transform([cleaned])
        prediction = model.predict(vec_input)[0]
        st.session_state.sentiment_label = prediction
        st.session_state.sentiment_result = f"This review is more <span style='color:{get_colour(prediction)}; font-weight:bold'>{prediction}</span>."
    else:
        st.warning("Please enter a review.")

# Colour mapping
def get_colour(sentiment):
    if sentiment == "positive":
        return "green"
    return "red"

# Text area
st.text_area(
    "Review:",
    height=200,
    key="review_box",
    on_change=lambda: setattr(st.session_state, "review_input", st.session_state.review_box)
)

# Buttons
col1, col2 = st.columns(2)

if col1.button("Get Sentiment", on_click=predict_sentiment):
    pass

if col2.button("Clear", on_click=clear_input):
    pass

# Show result
if st.session_state.sentiment_result:
    st.markdown(st.session_state.sentiment_result, unsafe_allow_html=True)