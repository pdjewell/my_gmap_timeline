import os
import json
import numpy as np
import pandas as pd
import datetime as dt 
import zipfile
import io
import glob
import regex as re

import openai 
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import streamlit as st
from streamlit_chat import message

from scripts.data_clean import read_in_data, clean_visits, clean_journeys


# Get data
def get_data(): 
    folder_path = './location_history/'
    data = read_in_data(folder_path)
    visits = clean_visits(data)
    journeys = clean_journeys(data)
    return visits, journeys 

def get_data_from_zip(uploaded_file): 
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall()
    #folder_path = os.path.join('.', 'Takeout', 'Location History', 'Semantic Location History')
    folder_path = './Takeout/Location History/Semantic Location History/'
    files = glob.glob(folder_path + '**/*.json', recursive = True)
    data_list = []
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
        data = data['timelineObjects']
        data_list.extend(data)
        print(f'Successfully read in this file: {f}') 
    print(f'Data is {type(data_list)} of length: {len(data_list)}')
    visits = clean_visits(data_list)
    journeys = clean_journeys(data_list)
    return visits, journeys 

def generate_response(prompt, df):

    llm_turbo = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    turbo_agent = create_pandas_dataframe_agent(
        llm=llm_turbo, df=df, verbose=False)
    message = turbo_agent.run(prompt)

    return message 

def get_text():
    input_text = st.text_input("Write your message to the chatbot here: ","", key="input")
    return input_text



if __name__ == "__main__":

    st.sidebar.title("Paste your OpenAI key and upload your Timeline data")

    # Chat bot 
    user_api_key = st.sidebar.text_input(
        label="#### OpenAI API key 👇",
        placeholder="Paste your OpenAI API key, sk-",
        type="password")
    os.environ["OPENAI_API_KEY"] = user_api_key

    # Upload zip file
    uploaded_file = st.sidebar.file_uploader("Upload the ZIP file 👇", type="zip")
    if uploaded_file is not None:
        #visits, journeys = get_data_from_zip(uploaded_file)
        st.sidebar.text("Zip file uploaded")
        visits, journeys = get_data_from_zip(uploaded_file)

    # Chatbot 
    st.title("Google Timeline Chatbot")
    st.text("Upload your OpenAI API key and Google Timeline data zipfile on the left") 
    st.text("Then you can ask the chatbot about your timeline data!\n""")

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

    # Interactive map
    st.title("Google Timeline Map")

    if uploaded_file is not None:

        options = st.multiselect('Which years would you like to see?', list(visits['visit start year'].unique())) 
        
        map_data = visits[visits['visit start year'].isin(options)] 
        map_data = map_data.rename(columns={'location latitude': 'LAT', 'location longitude': 'LON'})
        map_data = map_data[['LAT','LON']]
        map_data = map_data.dropna()
        
        #if st.button("Generate map"):
        st.map(data=map_data, zoom=None, use_container_width=True)
    
    else:
        st.text("Map will load once zip file uploaded!")
