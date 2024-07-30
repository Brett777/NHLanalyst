import re
import concurrent.futures
import os
import requests

import pandas as pd
import streamlit as st
import snowflake.connector
from openai import OpenAI

client = OpenAI(api_key=st.secrets.openai_credentials.key)
st.set_page_config(page_title="AI Data Analyst Demo", page_icon=":sparkles:", layout="wide")

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)

# Set to True to use OpenAI endpoints directly. False to use DataRobot endpoints.
openAImode = True

# Snowflake connection details
user = st.secrets.snowflake_credentials.user
password = st.secrets.snowflake_credentials.password
account = st.secrets.snowflake_credentials.account
warehouse = st.secrets.snowflake_credentials.warehouse
database = st.secrets.snowflake_credentials.database
schema = st.secrets.snowflake_credentials.schema



# Session state variables
if "table_selection_button" not in st.session_state:
    st.session_state["table_selection_button"] = False
    st.session_state["ask_button"] = False
    st.session_state["selectedTables"] = []

if "selectedTables" not in st.session_state:
    st.session_state['selectedTables'] = []

if "selectedTables" not in st.session_state:
    st.session_state['selectedCSVFile'] = []

if "csv_selection_button" not in st.session_state:
    st.session_state["csv_selection_button"] = False

@st.cache_data(show_spinner=False)
def getSnowflakeTableDescriptions(tables, user, password, account, warehouse, database, schema):
    # Establish a connection to Snowflake
    try:
        conn = snowflake.connector.connect(
            user=user,
            password=password,
            account=account,
            warehouse=warehouse,
            database=database,
            schema=schema,
            # Enable case sensitivity for identifiers
            case_sensitive_identifier_quoting=True
        )
        cursor = conn.cursor()
    except Exception as e:
        print(f"Error connecting to Snowflake: {e}")
        return None

    # Function to get primary keys of a table
    def get_primary_keys(table_name):
        try:
            cursor.execute(f"""
                SELECT COLUMN_NAME
                FROM {database}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
                JOIN {database}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
                ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
                AND tc.TABLE_SCHEMA = '{schema}'
                AND tc.TABLE_NAME = '{table_name}'
                """)
            return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            print(f"Error fetching primary keys for table {table_name}: {e}")
            return []

    # Function to get columns and data types along with additional metadata
    def get_columns_and_types(table_name):
        try:
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT, COMMENT
                FROM {database}.INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = '{schema}'
                AND TABLE_NAME = '{table_name}'
                """)
            columns = cursor.fetchall()
            primary_keys = get_primary_keys(table_name)
            return [(col[0], col[1], col[2] == 'YES', col[3], col[0] in primary_keys, col[4]) for col in columns]
        except Exception as e:
            print(f"Error fetching columns and types for table {table_name}: {e}")
            return []

    # Function to get table comment
    def get_table_comment(table_name):
        try:
            cursor.execute(f"""
                SELECT COMMENT
                FROM {database}.INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = '{schema}'
                AND TABLE_NAME = '{table_name}'
                """)
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error fetching table comment for {table_name}: {e}")
            return None

    # Function to get table row count
    def get_table_row_count(table_name):
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.{table_name}")
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            print(f"Error fetching row count for table {table_name}: {e}")
            return None

    # Prepare the descriptions string
    descriptions = ""

    for table in tables:
        descriptions += f"Table: {table}\n"
        table_comment = get_table_comment(table)
        if table_comment:
            descriptions += f" Comment: {table_comment}\n"
        row_count = get_table_row_count(table)
        descriptions += f" Row Count: {row_count}\n"
        for col_name, col_type, nullable, default, is_primary, col_comment in get_columns_and_types(table):
            descriptions += f' Column: "{col_name}", Type: {col_type}, Nullable: {nullable}, Default: {default}, Primary Key: {is_primary}, Comment: {col_comment}\n'
        descriptions += "---------------------------------------------------------------\n"

    # Close the connection
    cursor.close()
    conn.close()

    return descriptions

@st.cache_data(show_spinner=False)
def suggestQuestion(description):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": """
                            <YOUR ROLE>
                            Your job is to examine some meta data and suggest 3 business analytics questions that might yeild interesting insight from the data.
                            Inspect the user's metadata and suggest 3 different questions. They might be related, or completely unrelated to one another. 
                            Your suggested questions might require analysis across multiple tables, or might be confined to 1 table.
                            Another analyst will turn your question into a SQL query. As such, your suggested question should not require advanced statistics or machine learning to answer and should be straightforward to implement in SQL.
                            </YOUR ROLE>
                            
                            <CONTEXT> 
                            You will be provided with meta data about some tables in Snowflake.
                            For each question, consider all of the tables.
                            </CONTEXT>
                            
                            <YOUR RESPONSE>                        
                            Each question should be 1 or 2 sentences, no more.
                            Your response should only contain the suggested business questions and nothing else.
                            Format as a bullet list in markdown.
                            </YOUR RESPONSE>
                            
                            <NECESSARY CONSIDERATIONS>
                            Do not refer to specific column names or tables in the data. Just use common language when suggesting a question. Let the next analyst figure out which columns and tables they'll need to use. 
                            </NECESSARY CONSIDERATIONS>
                                                        
            """},
            {"role": "user", "content": description}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@st.cache_data(show_spinner=False)
def suggestQuestion2(description):
    # description = "this is a test."
    data = pd.DataFrame({"promptText": [description]})
    deployment_id = st.secrets.datarobot_deployment_id.suggest_a_question
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY

    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    suggestion = predictions_response.json()["data"][0]["prediction"]
    return suggestion

@st.cache_data(show_spinner=False)
def summarizeTable(dictionary, table):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": f"""
                            YOUR ROLE:
                            Your job is to examine some meta data and come up with a brief description of the dataset, 2 - 5 sentences long. 
                            Inspect the user's metadata and write the description for the table called {table}.
                            This description will help an analyst better understand a new dataset that they are seeing for the first time.
                            Suggest what kinds of business analytics questions this data set could be used to help answer.
                            Describe the business value or analytics value of this data.
                            Call out anything particularly insightful, interesting or unique about the dataset.

                            CONTEXT: 
                            You will be provided with meta data about multiple tables in Snowflake, but we only care about 1 of them: {table}                             

                            YOUR RESPONSE:                        
                            A description of the {table} table 2 or 5 sentences, no more.
                            Format as markdown.    
                            Do not include any headers.                        
            """},
            {"role": "user", "content": dictionary}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@st.cache_data(show_spinner=False)
def summarizeTable2(dictionary, table):
    # table = "This is a test"
    # dictionary = "this is a test dictionary."
    data = pd.DataFrame({"promptText": [str(dictionary) + "\nTABLE TO DESCRIBE: " + str(table)]})
    deployment_id = st.secrets.datarobot_deployment_id.summarize_table
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    summary = predictions_response.json()["data"][0]["prediction"]
    return summary

@st.cache_data(show_spinner=False)
def getDataDictionary(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": """
        <ROLE>
        You are a data dictionary maker. 
        Inspect this metadata to decipher what each column in the dataset is about is about. 
        Write a short description for each column that will help an analyst effectively leverage this data in their analysis.
        </ROLE>
        
        <CONTEXT>
        You will receive the following:
        1) The first 10 rows of a dataframe
        2) A summary of the data computed using pandas .describe()
        3) For categorical data, a list of the unique values limited to the top 10 most frequent values.
        </CONTEXT>
        
        <CONSIDERATIONS>
        The description should communicate what any acronyms might mean, what the business value of the data is, and what the analytic value might be. 
        You must describe ALL of the columns in the dataset to the best of your ability. 
        Your response should be formatted in markdown as a table or list of all of the columns names, along with your best attempt to describe what the column is about.
        To format text as a table in Markdown, you can use pipes (|) and dashes (-) to create the structure.
            
        Basic example:
        | Header 1 | Header 2 | Header 3 |
        |----------|----------|----------|
        | Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
        | Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |
        | Row 3, Col 1 | Row 3, Col 2 | Row 3, Col 3 | 
        </CONSIDERATIONS>
        """},
            {"role": "user", "content": prompt}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@st.cache_data(show_spinner=False)
def getDataDictionary2(prompt):
    '''
    Submits the data, gets dictionary
    '''
    #prompt = data

    data = pd.DataFrame({"promptText": [prompt]})
    deployment_id = st.secrets.datarobot_deployment_id.data_dictionary_maker
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    code = predictions_response.json()["data"][0]["prediction"]
    return code

@st.cache_data(show_spinner=False)
def assembleDictionaryParts(parts):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": """
            <ROLE>
            You are a data dictionary assembler.          
            A data dictionary explains to users what the columns of a dataset are about, and how that data could be used for analysis.
            The user will provide you with a series of mini data dictionaries.
            Your job is to assemble a final polished data dictionary by combining the mini data dictionaries provided by the user, into 1 single data dictionary.            
            </ROLE>

            <CONTEXT>
            The user will provide a series of mini data dictionaries, all in roughly the same format. 
            The format is a table containing: the name of the column and a description of what that column means. 
            Each mini dictionary will have 5 or fewer entries. 
            It's possible that you will only be provided with 1 mini dictionary, in which case your job is pretty easy! Just format the data and respond.         
            </CONTEXT>

            <YOUR RESPONSE>
            Respond with a single data dictionary containing all of the entries provided by the user.
            Avoid duplicate entries.
            Your response should be formatted as a table in markdown where content is aligned to the left.
            To format text as a table in Markdown, you can use pipes (|) and dashes (-) to create the structure.

            Basic example:
            | Header 1 | Header 2 | Header 3 |
            |----------|----------|----------|
            | Row 1, Col 1 | Row 1, Col 2 | Row 1, Col 3 |
            | Row 2, Col 1 | Row 2, Col 2 | Row 2, Col 3 |
            | Row 3, Col 1 | Row 3, Col 2 | Row 3, Col 3 |

            You can also align text within the columns using colons :.
            :--- aligns to the left.
            :---: aligns to the center.
            ---: aligns to the right.

            Example:
            | Left Align | Center Align | Right Align |
            |:-----------|:------------:|------------:|
            | Left       | Center       | Right       |
            | Left       | Center       | Right       |
             </YOUR RESPONSE>           
            """},
            {"role": "user", "content": str(parts)}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content

@st.cache_data(show_spinner=False)
def assembleDictionaryParts2(parts):
    '''
       Submits the data, assembles dictionary
    '''
    # parts = data

    data = pd.DataFrame({"promptText": [parts]})
    deployment_id = st.secrets.datarobot_deployment_id.data_dictionary_assembler
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    assembled = predictions_response.json()["data"][0]["prediction"]
    return assembled

def getPythonCode(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": """
                <ROLE>                
                You are a Python Pandas expert
                Your job is to write Pandas code that retrieves all the data needed to fully explain the answer to the user's business question.                   
                Carefully inspect the information and metadata provided to ensure your code will execute and return data as a Pandas dataframe.
                The result dataframe should not only answer the question, but provide the necessary context so the user can fully understand. 
                For example, if the user asks, "Which State has the highest revenue?" Your query might return the top 10 states by revenue sorted in descending order. 
                This way the user can analyze the context of the answer. 
                </ ROLE>
                
                <CONTEXT>   
                The user will provide a data dictionary that tells you the data type of each column.
                They will provide a small sample of data from each column. Useful for understanding the content of the columns as you build your query.
                They will also provide a list of frequently occurring values from VARCHAR / categorical columns. This would be helpful to know when adding filters / where clauses in your query.
                Based on this metadata, build your query so that it will run without error and return some data. 
                Your query should return not just the facts directly related to the question, but also return related information that could be part of the root cause or provide additional analytics value.
                Your query will be executed from Python using the Snowflake Python Connector.
                </CONTEXT>
                
                <RESPONSE>
                Your response shall only contain a Python function called analyze_data() that returns the relevant data as a dataframe                
                Your code should get any relevant, supporting or contextual information to help the user better understand the results.
                Try to ensure that your code does not return an empty dataframe.                
                Your code should be redundant to errors, with a high likelihood of successfully executing. 
                Your function must not return a dataset that is excessively lengthy, therefore consider appropriate groupbys and aggregations.
                The resulting dataframe from your function will be analyzed by humans and plotted in charts, so consider appropriate ways to organize and sort the data so that it's easy to interpret
                The dataframe should have appropriate column names so that it's easy to interpret and easy to plot.                                      
                Include comments to explain your code.
                Your response should be formatted as markdown where code is contained within a pattern like:
                ```python
                ```                        
                <FUNCTION REQUIREMENTS>
                Name: analyze_data()
                Input: A single pandas dataframe.
                Output: A single pandas dataframe.
                Import required libraries within the function.
                </FUNCTION REQUIREMENTS>         
                  
                </RESPONSE>
                
                <NECESSARY CONSIDERATIONS>   
                Carefully consider the metadata and the sample data when constructing your function to avoid errors or an empty result.       
                For example, seemingly numeric columns might contain non-numeric formatting such as $1,234.91 which could require special handling.
                When performing date operations on a date column, consider casting that column as a DATE for error redundancy.                
                Ensure error redundancy by type casting and taking other measure to ensure code executes successfully.
                </NECESSARY CONSIDERATIONS>
                
                <REATTEMPT>
                If your query fails due to an error or returns an empty result, you will also see the following text in the user's prompt:
                'QUERY FAILED! Attempt X failed with error: <error>  
                Take this error message into consideration when building your function so that the problem doesn't happen again.                
                Try again, but don't fail this time.
                </REATTEMPT>            """},
            {"role": "user", "content": prompt}])
    print(response.choices[0].message.content)
    # Pattern to match code blocks that optionally start with ```python or just ```
    pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(pattern, response.choices[0].message.content, re.DOTALL)

    # Join all matches into a single string, separated by two newlines
    python_code = '\n\n'.join(matches)
    return python_code

def getPythonCode2(prompt):
    '''
    Submits the user's prompt to DataRobot, gets Python
    '''
    # prompt = "test"
    data = pd.DataFrame({"promptText": [prompt]})
    deployment_id = st.secrets.datarobot_deployment_id.python_code_generator
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    code = predictions_response.json()["data"][0]["prediction"]
    return code

def executePythonCode(prompt, df):
    '''
    Executes the Python Code generated by the LLM
    '''
    print("Generating code...")
    if openAImode:
        pythonCode = getPythonCode(prompt)
    else:
        pythonCode = getPythonCode2(prompt)
    print(pythonCode.replace("```python", "").replace("```", ""))
    pythonCode = pythonCode.replace("```python", "").replace("```", "")
    print("Executing...")
    function_dict = {}
    exec(pythonCode, function_dict)  # execute the code created by our LLM
    analyze_data = function_dict['analyze_data']  # get the function that our code created
    results = analyze_data(df)
    return pythonCode, results

def getSnowflakeSQL(prompt, warehouse=warehouse, database=database, schema=schema):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": f"""
                <ROLE>                
                You are a Snowflake SQL query maker.
                Your job is to write a Snowflake SQL query that retrieves all the data needed to fully explain the answer to the user's business question.                   
                Carefully inspect the information and metadata provided to ensure your query will execute and return data.
                The result set should not only answer the question, but provide the necessary context so the user can fully understand. 
                For example, if the user asks, "Which State has the highest revenue?" Your query might return the top 10 states by revenue sorted in descending order. 
                This way the user can analyze the context of the answer. 
                </ ROLE>
                
                <CONTEXT>   
                The user will provide a data dictionary that tells you the data type of each column.
                They will provide a small sample of data from each column. Useful for understanding the content of the columns as you build your query.
                They will also provide a list of frequently occurring values from VARCHAR / categorical columns. This would be helpful to know when adding filters / where clauses in your query.
                Based on this metadata, build your query so that it will run without error and return some data. 
                Your query should return not just the facts directly related to the question, but also return related information that could be part of the root cause or provide additional analytics value.
                Your query will be executed from Python using the Snowflake Python Connector.
                </CONTEXT>
                
                <RESPONSE>
                Your response shall be a single, executable Snowflake SQL query that retrieves the data supporting the answer to the question.
                In addition, your response should return any relevant, supporting or contextual information to help the user better understand the results.
                Try to ensure that your query does not return an empty result set.
                Your code may not include any operations that could alter or corrupt the data in Snowlfake.
                You may not use DELETE, UPDATE, TRUNCATE, DROP, DML Operations, ALTER TABLE or anything that could permanently alter the data in Snowflake. 
                Your code should be redundant to errors, with a high likelihood of successfully executing. 
                The database contains very large transactional tables in excess of 10M rows. Your query result must not be excessively lengthy, therefore consider appropriate groupbys and aggregations.
                The result of this query will be analyzed by humans and plotted in charts, so consider appropriate ways to organize and sort the data so that it's easy to interpret
                Do not provide multiple queries that must be executed in different steps - the query must execute in a single step.      
                Do not include any USE statements.                          
                Include comments to explain your code.
                Your response should be formatted as markdown where SQL code is contained within a pattern like:
                ```sql
                ```                        
                SNOWFLAKE ENVIRONMENT:
                Warehouse: {warehouse}
                Database: {database}
                Schema: {schema}             
                  
                </RESPONSE>
                
                <NECESSARY CONSIDERATIONS>   
                Carefully consider the metadata and the sample data when constructing your query to avoid errors or an empty result.       
                For example, seemingly numeric columns might contain non-numeric formatting such as $1,234.91 which could require special handling.
                When performing date operations on a date column, consider casting that column as a DATE for error redundancy.                 
                To ensure case sensitivity of column names, use quotes around column names.     
                This query will be executed using the Snowflake Python Connector. Make sure the query will be compatible with the Snowflake Python Connector. 
                </NECESSARY CONSIDERATIONS>
                
                <REATTEMPT>
                If your query fails due to a SQL error or returns an empty result set, you will also see the following text in the user's prompt:
                'QUERY FAILED! Attempt X failed with error: <error> SQL Code: <your failed sql query>.
                Take this failed SQL code and error message into consideration when building your query so that the problem doesn't happen again.
                You might see an error message like this: 'NoneType' object has no attribute 'head'
                This means that the query returned an empty result set.
                Try again, but don't fail this time.
                </REATTEMPT>
               """},
            {"role": "user", "content": prompt}])
    print(response.choices[0].message.content)
    # Pattern to match code blocks that optionally start with ```python or just ```
    pattern = r'```(?:sql)?\n(.*?)```'
    matches = re.findall(pattern, response.choices[0].message.content, re.DOTALL)

    # Join all matches into a single string, separated by two newlines
    sql_code = '\n\n'.join(matches)
    return sql_code

def getSnowflakeSQL2(prompt, warehouse=warehouse, database=database, schema=schema):
    data = pd.DataFrame({"promptText": [str(prompt) + "\nSNOWFLAKE ENVIRONMENT:\nwarehouse = " + str(warehouse) + "\ndatabase = " + str(database) + "\nschema = " + str(schema)]})
    deployment_id = st.secrets.datarobot_deployment_id.sql_code_generator
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    code = predictions_response.json()["data"][0]["prediction"]
    # Pattern to match code blocks that optionally start with ```python or just ```
    pattern = r'```(?:sql)?\n(.*?)```'
    matches = re.findall(pattern, code, re.DOTALL)

    # Join all matches into a single string, separated by two newlines
    sql_code = '\n\n'.join(matches)
    return sql_code

def executeSnowflakeQuery(prompt, user, password, account, warehouse, database, schema):
    # Get the SQL code
    if openAImode:
        snowflakeSQL = getSnowflakeSQL(prompt)
    else:
        snowflakeSQL = getSnowflakeSQL2(prompt)

    # Create a connection using Snowflake Connector
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema,
        quote_identifiers=(True, '')
    )
    results = None

    try:
        # Execute the query and fetch the results into a DataFrame
        with conn.cursor() as cur:
            cur.execute(snowflakeSQL)
            results = cur.fetch_pandas_all()
            results.columns = results.columns.str.upper()
    except snowflake.connector.errors.Error as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

    return snowflakeSQL, results

@st.cache_data(show_spinner=False)
def getDataSample(sampleSize):
    sampleSQLprompt = f"""
                      Select a {sampleSize} row random sample using the SAMPLE clause                
                      """
    if openAImode:
        sampleSQL = getSnowflakeSQL(sampleSQLprompt)
    else:
        sampleSQL = getSnowflakeSQL2(sampleSQLprompt)

    sql, sample = executeSnowflakeQuery(sampleSQL, user, password, account, warehouse, database, schema)
    return sample

@st.cache_data(show_spinner=False)
def getTableSample(sampleSize, table):
    sqlCode, results = executeSnowflakeQuery(f"Retrieve a random sample using SAMPLE({sampleSize} ROWS) from this table: " + str(table), user, password, account, warehouse, database, schema)
    return results

def getChartCode(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.6,
        seed=4242,
        messages=[
            {"role": "system",
             "content": """
            <ROLE>
            You are a Plotly chart maker. 
            Your task is to create a function that returns 2 Plotly visualizations of the provided data to help answer a business question.
            </ROLE>
            
            <CONTEXT>
            You will be given a business question and a pandas dataframe containing information relevant to the question. 
            </CONTEXT>
            
            <YOUR RESPONSE>
            Your job is to create 2 complementary data visualizations using the Python library Plotly. 
            Your response must be a Python function that returns 2 plotly.graph_objects.Figure objects.
            Your function will have an input parameter df, which will be a dataframe just like the one provided in the context here. 
            Therefore, your function may only make use of data and columns like the data provided in the context here. 
            </YOUR RESPONSE>
            
            <FUNCTION REQUIREMENTS>
            Name: create_charts()
            Input: A single pandas dataframe.
            Output: Two plotly.graph_objects.Figure objects.
            Import required libraries within the function.
            </FUNCTION REQUIREMENTS>
            </YOUR RESPONSE>

            <NECESSARY CONSIDERATIONS>
            ONLY REFER TO COLUMNS THAT ACTUALLY EXIST IN THE INPUT DATA. 
            You must never refer to columns that don't exist in the input dataframe.
            When referring to columns in your code, spell them EXACTLY as they appear in the pandas dataframe - this might be different from how they are referenced in the business question! Only refer to columns that exist IN THE DATAFRAME.
            For example, if the question asks "What is the total amount paid ("AMTPAID") for each type of order?" but the dataframe does not contain "AMTPAID" but rather "TOTAL_AMTPAID", you should use "TOTAL_AMTPAID" in your code because that's the column name in the data.            
            Data Availability: If some data is missing, plot what you can in the most sensible way.
            Package Imports: If your code requires a package to run, such as statsmodels, numpy, scipy, etc, you must import the package within your function.
            Data Handling:
            If there are more than 100 rows, consider grouping or aggregating data for clarity.
            Round values to 2 decimal places if they have more than 2.
            Visualization Principles:
            Choose visualizations that effectively display the data and complement each other.
            Examples:
            Heatmap and Scatter Plot Matrix
            Bar chart and Choropleth (if state abbreviations or other required geospatial identifiers are available)
            Box Plot and Violin Plot
            Line Chart and Area Chart
            Scatter Plot and Histogram
            Bubble Chart and Treemap
            Time Series Plot and Heatmap            
            Design Guidelines:
            Simple, not overly busy or complex.
            No background colors or themes; use the default theme.
            Complementary colors you could use: #0B0A0D, #243E73, #1D3159, #8BB4D9, #A67E6F, #011826, #1A3940, #8C5946, #BF8D7A, #0D0D0D, #3805F2, #2703A6, #150259, #63A1F2, #84F266, #232625, #35403A, #4C594F, #A4A69C, #BFBFB8
            Gradient - Coral to Teal: #FF5F5D, #F76F67, #EE8071, #E6907C, #DD9F86, #D5AF90, #CDBF9A, #C4CFA4, #BCD0AF, #A3CCAB, #8BB8A7, #72A4A3, #59809F, #3F7C85
            Gradient - Teal to Aqua: #3F7C85, #367B88, #2D7A8A, #24798D, #1B7890, #117893, #087796, #007699, #00759C, #00749F, #0074A2, #0073A5, #0072A7, #00CCBF
            Gradient - Dark Teal to Light Gray: #14140F,#23231E,#32312D,#41403C,#51504B,#60605A,#707069,#808078,#909087,#A0A096,#B0B0A5,#C0C0B4,#D0D0C3,#CACACA
            Gradient - Ocean Blues: #003840,#00424A,#004C55,#00565F,#006069,#006A73,#00747C,#007E86,#008891,#00929B,#009CA5,#00A6AF,#00B0B9,#00BBC9
            Include titles, axis names, and legends.
            Robustness:
            Ensure the function is free of syntax errors and logical problems.
            Handle errors gracefully and ensure type casting for data integrity.
            Formatting:
            Provide the function in the following markdown format:
            ```python
            ``` 

            <EXAMPLE CODE STRUCTURE>
            ```python
            def create_charts(df):
                import pandas as pd
                import plotly.graph_objects as go
                from plotly.subplots import make_subplots
                # Other packages you might need
                
                # Your code to create charts here

                return fig1, fig2
            ```                     
            <EXAMPLE CODE STRUCTURE>
            
            <REATTEMPT>
            If your chart code fails to execute, you will also see the following text in the user's prompt:
            'CHART CODE FAILED!  Attempt X failed with error: ..."
            Take error message into consideration when reattempting your chart code so that the problem doesn't happen again.                     
            Try again, but don't fail this time.      
            </REATTEMPT>
            </NECESSARY CONSIDERATIONS>
            """},
            {"role": "user", "content": prompt}])
    # Pattern to match code blocks that optionally start with ```python or just ```
    pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(pattern, response.choices[0].message.content, re.DOTALL)

    # Join all matches into a single string, separated by two newlines
    python_code = '\n\n'.join(matches)
    return python_code

def getChartCode2(prompt):
    #prompt = "test"
    data = pd.DataFrame({"promptText": [prompt]})
    deployment_id = st.secrets.datarobot_deployment_id.plotly_code_generator
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    code = predictions_response.json()["data"][0]["prediction"]
    # Pattern to match code blocks that optionally start with ```python or just ```
    pattern = r'```(?:python)?\n(.*?)```'
    matches = re.findall(pattern, code, re.DOTALL)

    # Join all matches into a single string, separated by two newlines
    chart_code = '\n\n'.join(matches)
    return chart_code
def createCharts(prompt, results):
    print("getting chart code...")
    if openAImode:
        chartCode = getChartCode(prompt + str(results))
    else:
        chartCode = getChartCode2(prompt + str(results))
    print(chartCode.replace("```python", "").replace("```", ""))
    function_dict = {}
    exec(chartCode.replace("```python", "").replace("```", ""), function_dict)  # execute the code created by our LLM
    print("executing chart code...")
    create_charts = function_dict['create_charts']  # get the function that our code created
    fig1, fig2 = create_charts(results)
    return fig1, fig2

def getBusinessAnalysis(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        seed=42,
        messages=[
            {"role": "system",
             "content": """
            ROLE:
            You are a business analyst.
            Your job is to write an answer to the user's question in 3 sections (heading level 3): The Bottom Line, Additional Insights, Follow Up Questions.

            CONTEXT:
            The user has asked a business question and we have represented it as a SQL query.
            We have also executed that query and retrieved the results.
            You will be provided with the user's question, the sql query and the resulting data from that query.

            YOUR RESPONSE:            
            Your response must be formatted as Markdown and include 3 sections (heading level 3): The Bottom Line, Additional Insights, Follow Up Questions.

            The Bottom Line
            Based on the context information provided, clearly and succinctly answer the user's question in plain language, tailored for someone with a business background rather than a technical one.

            Additional Insights
            This section is all about the "why". Discuss the underlying reasons or causes for the answer in "The Bottom Line" section. This section, while still business focused, should go a level deeper to help the user understand a possible root cause. Where possible, justify your answer using data or information from the dataset.
            Provide business advice based on the outcome noted in "The Bottom Line" section. 
            Suggest specific additional analyses based on the context of the question and the data available in the Table Definition. 
            Offer actionable recommendations. For example, if the data shows a declining trend in TOTAL_PROFIT, advise on potential areas to investigate using other data in the dataset, and propose analytics strategies to gain insights that might improve profitability. 

            Follow Up Questions
            Offer 2 or 3 follow up questions the user could ask to get deeper insight into the issue in another round of question and answer. When you word these questions, do not use pronouns to refer to the data - always use specific column names. Only refer to data that actually exists in the dataset. For example, don't refer to "sales volume" if there is no "sales volume" column.
            """},
            {"role": "user", "content": prompt}])
    print(response.choices[0].message.content)
    return response.choices[0].message.content

def getBusinessAnalysis2(prompt):
    data = pd.DataFrame({"promptText": [prompt]})
    deployment_id = st.secrets.datarobot_deployment_id.business_analysis
    API_URL = f'{st.secrets.datarobot_credentials.PREDICTION_SERVER}/predApi/v1.0/deployments/{deployment_id}/predictions'
    API_KEY = st.secrets.datarobot_credentials.API_KEY
    DATAROBOT_KEY = st.secrets.datarobot_credentials.DATAROBOT_KEY
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': 'Bearer {}'.format(API_KEY),
        'DataRobot-Key': DATAROBOT_KEY,
    }
    url = API_URL.format(deployment_id=deployment_id)
    predictions_response = requests.post(
        url,
        data=data.to_json(orient='records'),
        headers=headers
    )
    business_analysis = predictions_response.json()["data"][0]["prediction"]
    return business_analysis

@st.cache_data(show_spinner=False)
def get_top_frequent_values(df):
    # Select non-numeric columns
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns

    # Prepare a list to store the results
    results = []

    # Iterate over non-numeric columns
    for col in non_numeric_cols:
        # Find top 10 most frequent values for the column
        top_values = df[col].value_counts().head(10).index.tolist()

        # Convert the values to strings
        top_values = [str(value) for value in top_values]

        # Append the column name and its frequent values to the results
        results.append({'Non-numeric column name': col, 'Frequent Values': top_values})

    # Create a new DataFrame for the results
    result_df = pd.DataFrame(results)

    return result_df

def createChartsAndBusinessAnalysis(businessQuestion, results, prompt):
    attempt_count = 0
    max_attempts = 4
    fig1 = fig2 = None
    analysis = None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        while attempt_count < max_attempts:
            chart_future = executor.submit(createCharts, businessQuestion, results)
            if openAImode:
                analysis_future = executor.submit(getBusinessAnalysis, prompt + str(results))
            else:
                analysis_future = executor.submit(getBusinessAnalysis2, prompt + str(results))
            try:
                if fig1 is None or fig2 is None:
                    fig1, fig2 = chart_future.result(timeout=30)  # Add a timeout for better handling
                    with st.expander(label="Charts", expanded=True):
                        st.plotly_chart(fig1, theme="streamlit", use_container_width=True)
                        st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
                break  # If operation succeeds, break out of the loop
            except Exception as e:
                attempt_count += 1
                print(f"Chart Attempt {attempt_count} failed with error: {repr(e)}")
                fig1_str = str(fig1) if fig1 is not None else "None"
                fig2_str = str(fig2) if fig2 is not None else "None"
                businessQuestion += f"\nCHART CODE FAILED!  Attempt {attempt_count} failed with error: {repr(e)}\nFig1: {fig1_str}\nFig2: {fig2_str}"

                if attempt_count >= max_attempts:
                    print("Max charting attempts reached, handling the failure.")
                    st.write("I was unable to plot the data.")
                    # Handle the failure after the final attempt
                else:
                    print("Retrying the charts...")

        try:
            with st.expander(label="Business Analysis", expanded=True):
                analysis = analysis_future.result(timeout=30)  # Add a timeout for better handling
                st.markdown(analysis.replace("$", "\$"))
        except:
            st.write("I am unable to provide the analysis. Please rephrase the question and try again.")
@st.cache_data(show_spinner=False)
def process_tables(dictionary, selectedTables, sampleSize):
    tableSamples = []
    tableDescriptions = []
    frequentValues = pd.DataFrame()

    for table in selectedTables:
        if openAImode:
            tableDescription = summarizeTable(dictionary, table)
        else:
            tableDescription = summarizeTable2(dictionary, table)

        results = getTableSample(sampleSize=sampleSize, table=table)
        tableSamples.append(results)
        tableDescriptions.append(tableDescription)
        freqVals = get_top_frequent_values(results)
        frequentValues = pd.concat([frequentValues, freqVals], axis=0)

    smallTableSamples = []
    for table in tableSamples:
        smallSample = table.sample(n=3)
        smallTableSamples.append(smallSample)

    return tableDescriptions, tableSamples, smallTableSamples, frequentValues
@st.cache_data(show_spinner=False)
def getSnowflakeTables(user, password, account, database, schema, warehouse):
    # Establish the connection
    conn = snowflake.connector.connect(
        user=user,
        password=password,
        account=account,
        warehouse=warehouse,
        database=database,
        schema=schema
    )

    try:
        # Create a cursor object
        cursor = conn.cursor()

        # Execute a query to fetch the table names
        cursor.execute(f"""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = '{schema}'
                """)

        # Fetch all table names
        tables = [row[0] for row in cursor.fetchall()]
        tables.sort()

        return tables

    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

def mainPage():
    st.image("DataRobot Logo.svg", width=300)
    # st.image("Customer Logo.svg", width=300)

    tab1, tab2 = st.tabs(["Analyze", "Explore"])
    with tab1:
        st.title("Ask a question about the data.")

        tables = getSnowflakeTables(user, password, account, database, schema, warehouse)

        with st.sidebar:
            st.image("Snowflake.svg", width=75)
            with st.form(key='table_selection_form'):
                # selectedTables = ['LENDING_CLUB_PROFILE', 'LENDING_CLUB_TRANSACTIONS', 'LENDING_CLUB_TARGET']
                # selectedTables = ['STOP']
                selectedTables = st.multiselect(label="Choose a few tables", options=tables, key="table_select_box")
                snowflake_submit_button = st.form_submit_button(label='Analyze', type="secondary")
            if snowflake_submit_button:
                st.session_state['selectedTables'] = selectedTables
                st.session_state["table_selection_button"] = True

            #CSV Uploader
            st.image("csv_File_Logo.svg", width=35)
            csvFile = st.file_uploader(label="Or, upload a CSV file", accept_multiple_files=False)



        if st.session_state["table_selection_button"]:
            with st.spinner("Getting table definitions..."):
                dictionary = getSnowflakeTableDescriptions(st.session_state['selectedTables'], user, password, account, warehouse, database, schema)
                print(dictionary)
                if openAImode:
                    suggestedQuestions = suggestQuestion(dictionary)
                else: suggestedQuestions = suggestQuestion2(dictionary)

                print(suggestedQuestions)
                tableDescriptions, tableSamples, smallTableSamples, frequentValues = process_tables(dictionary,st.session_state['selectedTables'], sampleSize=1000)
                with tab2:
                    for i in range(0, len(tableSamples)):
                        st.subheader(st.session_state['selectedTables'][i])
                        st.caption("Displaying a random sample " + str(len(tableSamples[i])) + " rows")
                        st.write(tableDescriptions[i])
                        st.write(tableSamples[i])
            st.write(suggestedQuestions)

            # Initialize businessQuestion session state variable
            if 'businessQuestion' not in st.session_state:
                st.session_state["businessQuestion"] = ""

            # Function to clear text input
            def clear_text():
                st.session_state["businessQuestion"] = ""

            # st.session_state["businessQuestion"] = "How many customers from California had a bad loan?"
            st.session_state["businessQuestion"] = st.text_input(label="Question", value=st.session_state["businessQuestion"])

            # button columns
            buttonContainer = st.container()
            buttonCol1, buttonCol2, empty = buttonContainer.columns([1, 1, 8])
            submitQuestion = buttonCol1.button(label="Ask", use_container_width=True, type="primary")
            clearButton = buttonCol2.button(label="clear", use_container_width=True, type="secondary", on_click=clear_text)
            if submitQuestion:
                with st.spinner("Analyzing... "):
                    print("------------")
                    print(st.session_state["businessQuestion"])
                    print("------------")
                    prompt = "Business Question: " + str(st.session_state["businessQuestion"]) + str("\n Data Dictionary: \n") + str(dictionary) + str("\n Data Sample: \n") + str(smallTableSamples) + str("\n Frequent Values: \n") + str(frequentValues)
                    print(prompt)
                    print("------------")

                    attempts = 0
                    max_retries = 5
                    while attempts < max_retries:
                        print("Generating code to get the answer. Attempt: " + str(attempts))
                        sqlCode = None
                        try:
                            sqlCode, results = executeSnowflakeQuery(prompt, user, password, account, warehouse,database, schema)
                            print("Query Result:")
                            print(sqlCode)
                            print(results.head(3))
                            if results.empty: raise ValueError("The DataFrame is empty, retrying...")
                            break  # If the function succeeds, exit the loop
                        except Exception as e:
                            attempts += 1
                            print(f"Query attempt {attempts} failed with error: {repr(e)}")
                            sqlCode_str = str(sqlCode) if sqlCode is not None else "None"
                            prompt += f"\nQUERY FAILED! Attempt {attempts} failed with error: {repr(e)}\nSQL Code: {sqlCode_str}"
                            if attempts == max_retries:
                                print("Max retries reached.")
                                break


                    try:
                        with st.expander(label="Code", expanded=False):
                            st.code(sqlCode, language="sql")
                        with st.expander(label="Result", expanded=True):
                            st.table(results)
                    except:
                        st.write(
                            "I tried a few different ways, but couldn't get a working solution. Rephrase the question and try again.")

                with st.spinner("Visualization and analysis in progress..."):
                    createChartsAndBusinessAnalysis(st.session_state["businessQuestion"], results, prompt)
        elif csvFile is not None:
            with tab1:
                with st.spinner("Processing data, see Explore tab for details..."):
                    with tab2:
                        # df = pd.read_csv(r"C:\Users\BrettOlmstead\PycharmProjects\DataAnalyst - Snowflake\DataAnalystGPT4oCustomAppSnowflakeDemo\DR_Demo_Employee_Attrition.csv")
                        df = pd.read_csv(csvFile)
                        # Display the dataframe
                        with st.expander(label="First 10 Rows", expanded=False):
                            st.dataframe(df.head(10))

                        try:
                            with st.expander(label="Column Descriptions", expanded=False):
                                st.dataframe(df.describe(include='all'))
                        except:
                            pass

                        try:
                            with st.expander(label="Unique and Frequent Values", expanded=False):
                                st.dataframe(get_top_frequent_values(df))
                        except Exception as e:
                            print(e)

                        try:
                            with st.expander(label="Data Dictionary", expanded=True):
                                with st.spinner("Making dictionary..."):
                                    # Initialize an empty list to hold the markdown strings
                                    dictionary_chunks = []
                                    # Define the chunk size
                                    chunk_size = 10
                                    total_columns = len(df.columns)

                                    # Initialize the progress bar
                                    progress_placeholder = st.empty()  # Placeholder for the progress bar

                                    for start in range(0, total_columns, chunk_size):
                                        # Update the progress bar and text
                                        current_chunk = start // chunk_size + 1
                                        total_chunks = (total_columns + chunk_size - 1) // chunk_size
                                        progress = current_chunk / total_chunks

                                        with progress_placeholder.container():
                                            st.progress(progress,
                                                        text=f'Processing {chunk_size} columns at a time in chunks. Currently working on chunk {current_chunk} of {total_chunks}')

                                        # Select the subset of columns
                                        end = min(start + chunk_size, total_columns)
                                        subset = df.iloc[:10, start:end]
                                        data = "First 10 Rows: \n" + str(
                                            subset) + "\n Unique and Frequent Values of Categorical Data: \n" + str(
                                            get_top_frequent_values(df))

                                        # Call the function and collect the result
                                        if openAImode:
                                            dictionary_chunk = getDataDictionary(data)
                                        else:
                                            dictionary_chunk = getDataDictionary2(data)


                                        dictionary_chunks.append(dictionary_chunk)

                                    # Remove the progress bar when complete
                                    progress_placeholder.empty()
                                with st.spinner("Putting it all together..."):
                                    if openAImode:
                                        dictionary = assembleDictionaryParts(dictionary_chunks)
                                    else:
                                        dictionary = assembleDictionaryParts2(dictionary_chunks)
                                    st.markdown(dictionary)
                        except:
                            pass
            with tab1:
                if openAImode:
                    suggestedQuestions = suggestQuestion(dictionary)
                else:
                    suggestedQuestions = suggestQuestion2(dictionary)
                print(suggestedQuestions)
                st.write(suggestedQuestions)
                # Initialize businessQuestion session state variable
                if 'businessQuestion' not in st.session_state:
                    st.session_state["businessQuestion"] = ""

                # Function to clear text input
                def clear_text():
                    st.session_state["businessQuestion"] = ""

                # st.session_state["businessQuestion"] = "How many customers from California had a bad loan?"
                st.session_state["businessQuestion"] = st.text_input(label="Question",value=st.session_state["businessQuestion"])

                # button columns
                buttonContainer = st.container()
                buttonCol1, buttonCol2, empty = buttonContainer.columns([1, 1, 8])
                submitQuestion = buttonCol1.button(label="Ask", use_container_width=True, type="primary")
                clearButton = buttonCol2.button(label="clear", use_container_width=True, type="secondary",on_click=clear_text)
                if submitQuestion:
                    with st.spinner("Analyzing... "):
                        print("------------")
                        print(st.session_state["businessQuestion"])
                        print("------------")
                        prompt = "Business Question: " + str(st.session_state["businessQuestion"]) +"\n Data Sample: \n" + str(df.head(3)) + "\n Unique and Frequent Values of Categorical Data: \n" + str(get_top_frequent_values(df)) + str("\n Data Dictionary: \n") + str(dictionary)
                        print(prompt)
                        print("------------")

                        attempts = 0
                        max_retries = 10
                        while attempts < max_retries:
                            print("Generating code to get the answer. Attempt: " + str(attempts))
                            pythonCode = None
                            try:
                                pythonCode, results = executePythonCode(prompt, df)
                                print("Query Result:")
                                print(pythonCode)
                                print(results.head(3))
                                if results.empty: raise ValueError("The DataFrame is empty, retrying...")
                                break  # If the function succeeds, exit the loop
                            except Exception as e:
                                attempts += 1
                                print(f"Query attempt {attempts} failed with error: {repr(e)}")
                                pythonCode_str = str(pythonCode) if pythonCode is not None else "None"
                                prompt += f"\nQUERY FAILED! Attempt {attempts} failed with error: {repr(e)}\nSQL Code: {pythonCode_str}"
                                if attempts == max_retries:
                                    print("Max retries reached.")
                                    break

                        try:
                            with st.expander(label="Code", expanded=False):
                                st.code(pythonCode, language="sql")
                            with st.expander(label="Result", expanded=True):
                                st.table(results)
                        except:
                            st.write(
                                "I tried a few different ways, but couldn't get a working solution. Rephrase the question and try again.")

                    with st.spinner("Visualization and analysis in progress..."):
                        createChartsAndBusinessAnalysis(st.session_state["businessQuestion"], results, prompt)

# Main app
def _main():
    hide_streamlit_style = """
    <style>
    # MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)  # This lets you hide the Streamlit branding

    mainPage()

if __name__ == "__main__":
    _main()
