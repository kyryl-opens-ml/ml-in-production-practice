import pandas as pd
import streamlit as st

from serving.predictor import Predictor


@st.cache_data
def get_model() -> Predictor:
    return Predictor.default_from_model_registry()


predictor = get_model()


def single_pred():
    input_sent = st.text_input("Type english sentence", value="This is example input")
    if st.button("Run inference"):
        st.write("Input:", input_sent)
        pred = predictor.predict([input_sent])
        st.write("Pred:", pred)


def batch_pred():
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        dataframe = pd.read_csv(uploaded_file)
        st.write("Input dataframe")
        st.write(dataframe)
        dataframe_with_pred = predictor.run_inference_on_dataframe(dataframe)
        st.write("Result dataframe")
        st.write(dataframe_with_pred)


def main():
    st.header("UI serving demo")

    tab1, tab2 = st.tabs(["Single prediction", "Batch prediction"])

    with tab1:
        single_pred()

    with tab2:
        batch_pred()


if __name__ == "__main__":
    main()
