import os
import json
import numpy as np
import pandas as pd
import datetime as dt 
import streamlit as st
from streamlit_chat import message


import regex as re
import tqdm
from scripts.data_clean import read_in_data, clean_visits, clean_journeys

import openai 
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI


# OpenAI API key

# Get data
def get_data(): 
    folder_path = './location_history/'
    data = read_in_data(folder_path)
    visits = clean_visits(data)
    journeys = clean_journeys(data)
    return visits, journeys 

def generate_response(prompt, df):

    llm_turbo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    turbo_agent = create_pandas_dataframe_agent(
        llm=llm_turbo, df=df, verbose=False)
    message = turbo_agent.run(prompt)

    return message 

def get_text():
    input_text = st.text_input("You: ","", key="input")
    return input_text


if __name__ == "__main__":

    visits, journeys = get_data()

    # Chat bot 
    user_api_key = st.sidebar.text_input(
        label="#### OpenAI API key 👇",
        placeholder="Paste your OpenAI API key, sk-",
        type="password")
    os.environ["OPENAI_API_KEY"] = user_api_key

    st.title("Google Timeline Chatbot")
    st.text("Ask the chatbot about your timeline data!")

    # Storing the chat
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []

    user_input = get_text()

    if user_input:
        try:
            output = generate_response(user_input, visits)
            # store the output 
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)
        except:
            st.text('ERROR. Is the API key correct?')

    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

    st.title("Google Timeline Map")
    options = st.multiselect(
        'Which years would you like to see?',
        [2017, 2018, 2019, 2020, 2021, 2022, 2023]) 
    
    map_data = visits[visits['visit start year'].isin(options)] 
    map_data = map_data.rename(columns={'location latitude': 'LAT', 'location longitude': 'LON'})
    map_data = map_data[['LAT','LON']]
    map_data = map_data.dropna()
    
    #if st.button("Generate map"):
    st.map(data=map_data, zoom=None, use_container_width=True)
