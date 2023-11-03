import json

import streamlit as st
from streamlit_extras.app_logo import add_logo
from sentence_transformers import SentenceTransformer
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement
import pandas as pd

"""
This streamlit application assumes the cheese data has been inserted into AstraDB. This can be done by the workbook.

streamlit run Streamlit-Cheese-Demo.py 

"""

SECURE_CONNECT_BUNDLE_PATH = ''
ASTRA_CLIENT_ID = ''
ASTRA_CLIENT_SECRET = ''
KEYSPACE_NAME = 'ks1'
TABLE_NAME = 'images'



# Converting links to html tags
def path_to_image_html(path):
    return '<img src="' + path + '" style=max-height:124px;"/>'


def main():
    # Custom CSS
    st.markdown(
        '''
        <style>
        .streamlit-font {
             font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;
        }
        .streamlit-expanderHeader {
            background-color: black;
            color: white; # Adjust this for expander header color
        }
        .streamlit-expanderContent {
            background-color: #f9f9f7;
            color: black; # Expander content color
        }
        .streamlit-expand:hover {
            background-color: blue;
        }
        .streamlit-stSidebar{
            min-width: 400px;
            max-width: 800px;
        }
        .streamlit-slider {
            background: (to right, rgb(246, 51, 102) 0%, rgb(246, 51, 102) 75%, rgb(213, 218, 229) 75%, rgb(213, 218, 229) 100%);
        }
        </style>
        ''',
        unsafe_allow_html=True
    )

    st.title("Astra DB Vector search : Cheese Query Engine")

    new_title = '<p style="font-family:sans-serif; color:Grey; font-size: 20px;">Demo cheese search is based on the sample dataset shown in the side window.</p>'
    st.markdown(new_title, unsafe_allow_html=True)


    with st.expander('More info'):
        st.write("""
       The dataset has been stored in AstraDB with the cheese vectored  using Hugging Face (sentence-transformer)
       which creates a 384 dimension dense vector for sematic searching.

       Based on the inout , this demo will:

       1. Search the vector database for the cheese matches, by vectoring the name and prefroming a semantic search;
       2. Allow you to refine the search with a text tokeniser across the description field;
       3. Define the number of responses;

       """)

    with open('Data/cheese_data.json', 'r') as file:
        cheese_data = json.load(file)
        logo_url = "Data/ds.png"
        st.sidebar.image(logo_url)
        st.sidebar.title("Sample JSON dataset")
        st.sidebar.json(cheese_data)

    with st.form(key = "Query"):
        vector = st.text_input(label = "Enter cheese name or type (Vector search)")
        desc = st.text_input(label="Enter a search type search ")
        limit = st.slider("Response range", min_value=1, max_value=7,value=3)
        submit = st.form_submit_button(label = "Search")

    st.divider()

    model = SentenceTransformer('obrizum/all-MiniLM-L6-v2')

    embeddings = ""
    if vector:
        embeddings = model.encode(vector).tolist()

        with st.expander('Vector embedding (384 dimension vector)'):
            st.text_area(vector, embeddings)

        st.divider()

    cloud_config = {
        'secure_connect_bundle': SECURE_CONNECT_BUNDLE_PATH
    }
    auth_provider = PlainTextAuthProvider(ASTRA_CLIENT_ID, ASTRA_CLIENT_SECRET)
    cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
    session = cluster.connect()

    new_title = '<p style="font-family:sans-serif; color:Grey; font-size: 20px;">Results:</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    if vector and desc:
        # vector and desc search
        query = f"SELECT name, aroma, image, country_origin FROM {KEYSPACE_NAME}.{TABLE_NAME} WHERE description : '{desc}' ORDER BY item_vector ANN OF {embeddings} LIMIT {limit}"

        results = session.execute(query)
        top_results = results._current_rows

        df = pd.DataFrame(top_results, columns = [ 'Cheese Name', 'Aroma', 'Country', 'Image'])
        st.markdown(df.to_html(escape=False, formatters=dict(Image=path_to_image_html)), unsafe_allow_html=True)
    elif vector:
        #vector only search
        query = f"SELECT name, aroma, image, country_origin FROM {KEYSPACE_NAME}.{TABLE_NAME} ORDER BY item_vector ANN OF {embeddings} LIMIT {limit}"

        results = session.execute(query)
        top_results = results._current_rows

        df = pd.DataFrame(top_results, columns = [ 'Cheese Name', 'Aroma', 'Country', 'Image'])
        st.markdown(df.to_html(escape=False, formatters=dict(Image=path_to_image_html)), unsafe_allow_html=True)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
