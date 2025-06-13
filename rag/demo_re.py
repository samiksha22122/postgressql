# Installing Required Libraries
%pip install python-docx
%pip install python-pptx
%pip install PyPDF2
%pip install langchain
%pip install langchain_community
%pip install langchain_google_genai
%pip install langchain_text_splitters
%pip install sentence-transformers
%pip install faiss-cpu
%pip install cohere

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DPRQuestionEncoder, DPRContextEncoder, DPRRetriever, RagTokenizer, RagRetriever, RagSequenceForGeneration
import torch

# Load tokenizer and models
retriever_model_name = 'facebook/dpr-ctx_encoder-single-nq-base'
question_model_name = 'facebook/dpr-question_encoder-single-nq-base'
generation_model_name = 'facebook/rag-sequence-nq'

tokenizer_retriever = AutoTokenizer.from_pretrained(retriever_model_name)
model_retriever = AutoModelForSeq2SeqLM.from_pretrained(retriever_model_name)
tokenizer_question = AutoTokenizer.from_pretrained(question_model_name)
model_question = AutoModelForSeq2SeqLM.from_pretrained(question_model_name)
tokenizer_generation = RagTokenizer.from_pretrained(generation_model_name)
model_generation = RagSequenceForGeneration.from_pretrained(generation_model_name)

# Load data from CSV
csv_file_path = '/content/healthcare_dataset.csv'  # Path to your CSV file
data = pd.read_csv(csv_file_path)

# Function to encode text and get embeddings
def get_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
    return embeddings

# Process text and get embeddings
data['embeddings'] = data['text'].apply(lambda x: get_embeddings(x, tokenizer_question, model_question).tolist())

# Initialize retriever
retriever = RagRetriever.from_pretrained(generation_model_name, tokenizer=tokenizer_generation)

# Example query
query = "What are the latest trends in AI?"

# Encode the query and retrieve relevant documents
query_embeddings = get_embeddings(query, tokenizer_question, model_question)
retrieved_docs = retriever.retrieve(query_embeddings, data['embeddings'].tolist())

# Generate response using retrieved documents
input_ids = tokenizer_generation(query, return_tensors='pt').input_ids
generated_ids = model_generation.generate(input_ids, retriever_contexts=retrieved_docs)
generated_text = tokenizer_generation.decode(generated_ids[0], skip_special_tokens=True)

print(f"Generated response: {generated_text}")

# Save updated DataFrame
output_csv_file_path = 'data_with_embeddings.csv'
data.to_csv(output_csv_file_path, index=False)
# Databricks notebook source
# MAGIC %md
# MAGIC # 00. Introduction
# MAGIC This notebook implements a comprehensive Retrieval-Augmented Generation (RAG) pipeline and logs the resulting question-answer model. The key functionalities integrated into this notebook include:
# MAGIC
# MAGIC 1. **Definition of the `RetrieverModel` Class**:
# MAGIC    - The `RetrieverModel` class is the core of the end-to-end RAG pipeline, handling all necessary operations from data retrieval to answer generation.
# MAGIC
# MAGIC 2. **Query Filter Generation**:
# MAGIC    - Custom query filters are created to refine and narrow down the scope of data retrieval from Snowflake. These filters ensure that the search is focused on relevant and targeted data, improving the efficiency and accuracy of the retrieval process.
# MAGIC
# MAGIC 3. **Top-k Records Retrieval**:
# MAGIC    - After applying the custom filters, the pipeline fetches the top k records from the database. These records are selected based on their similarity scores with the user's query, ensuring that the most relevant data is retrieved for further processing.
# MAGIC
# MAGIC 4. **Descriptive Answer Generation**:
# MAGIC    - Leveraging a Large Language Model (LLM), the pipeline generates detailed and informative answers to the user's questions. The retrieved top k records provide the context and information needed for the LLM to produce accurate and descriptive responses.
# MAGIC
# MAGIC Overall, this notebook not only defines and executes the complete RAG pipeline but also logs the trained question-answer model for future use, facilitating efficient and accurate information retrieval and response generation.

# COMMAND ----------

# MAGIC %md
# MAGIC #01. Importing Libraries

# COMMAND ----------

# MAGIC %pip show openai

# COMMAND ----------

# MAGIC %pip install ragas==0.1.10

# COMMAND ----------

# MAGIC %pip show openai

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import os
import re
import json
import logging
from typing import List, Dict

import numpy as np
import pandas as pd
import snowflake.connector
from openai import OpenAI

# COMMAND ----------

import mlflow
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.types.schema import Schema, ColSpec

# COMMAND ----------

# MAGIC %md
# MAGIC #02. Importing Notebooks
# MAGIC - snowflake_credentials: To load snowflake credentials
# MAGIC - Load databricks user credentials from environment variables
# MAGIC - QueryFilterException: Custom exception class for errors raised by query filters

# COMMAND ----------

# MAGIC %run "../../../../../common/snowflake_credentials"

# COMMAND ----------

# Loading user credentials from notebook environment variables
sfUser = dbutils.secrets.get("commercial", "snowflake_comm_medical_datahub_etl_user")
sfPwd = dbutils.secrets.get("commercial", "snowflake_comm_medical_datahub_etl_pwd")

# COMMAND ----------

dbcEnv = os.getenv("dbc_environment")
if dbcEnv == 'DEV':
  url = 'devtest'
else:
  url = dbcEnv.lower()

# COMMAND ----------

class QueryFilterException(Exception):
    """
    Exception raised for errors in the query filtering process.

    Attributes:
        message (str): Explanation of the error.
    """
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC #03. Setting up configs for Notebook
# MAGIC

# COMMAND ----------

rag_config = {
    "model_name": "question-answering",
    "TOKEN": dbutils.secrets.get("commercial", "dbc_model_access_token"),
    "endpoint_url_base": f"https://sagerx-aws-{url}-comm.cloud.databricks.com/serving-endpoints",
    "response_model_settings": {
        "name": "databricks-meta-llama-3-70b-instruct",
        "max_tokens": 5000,
        "max_response_tokens": 2000,
    },
    "filter_model_settings": {
        "name": "databricks-meta-llama-3-70b-instruct",
        "max_tokens": 8000,
        "max_response_tokens": 1000,
        "temperature": 0.1,
    },
    "embedding_model_settings": {"name": "databricks-bge-large-en"},
    "filter_settings": {
        "prompt": "Based on the question, you have to create filter or condition clause for the query which will be used to fetch relevant records from the database. You have to provide only query filter in response, as any additional text will result in error. You have to create filter only on the following columns: VCRM_CREATED_DATE (TIMESTAMP_NTZ), THERAPEUTIC_FOCUS (VARCHAR), COMPOUND_PRODUCT (VARCHAR), PREDICTED_TOPIC (VARCHAR), PREDICTED_SENTIMENT (VARCHAR), SOURCE_DATA_PROVIDER (VARCHAR).",
        "unique_values": {
            "THERAPEUTIC_FOCUS": "None",
            "COMPOUND_PRODUCT": "None",
            "HCP_SPECIALTY": "None",
            "PREDICTED_TOPIC": "None",
            "PREDICTED_SENTIMENT": "None",
            "SOURCE_DATA_PROVIDER": "None",
        },
        "notes": "Filters should be based solely on the unique values in the specified columns. Ensure that the filter does not have an AND clause on the same column. Also, based on the question, try to assign closest topic from PREDICTED_TOPIC if predefined topics are not mentioned.",
    },
    "qa_settings": {
        "prompt": """
        You are a smart assistant specializing in the medical field. Answer questions based only on the provided medical context, using detailed bullet points.

        If you cannot answer, say "I don't know. Please ask a different question." If no context is provided, say "No record found for the question asked. Please ask a different question." If greeted, respond with a greeting. If the user asks out-of-domain questions, kindly respond with "Refer to the user guide and ask questions based on the provided guidelines."

        Do not repeat the user's question. Format your answers with bullet points with (*) that include a title and explanation, and follow with a brief summary of the key points discussed in a paragraph. Avoid using nested bullet points.
        """
    },
    "query_settings": {
        "db_name": f"SCUDE_MEDICAL_AFFAIRS_{dbcEnv}",
        "schema_name": "DEP_CONFORMED",
        "table_name": "VOKOL_SENTIMENT_ANALYSIS",
        "top_k": "10",
        "embedding_col": "VECTORIZED_VOKOL",
        "vokol_col": "VOICE_OF_CUSTOMER",
        "vokol_id_col": "VOKOL_NAME",
        "date_col": "VCRM_CREATED_DATE",
        "hcp_speciality_col": "HCP_SPECIALTY",
    },
    "default_context": "No relevant documents found. Please try another question.",
    "default_error_message": "Something went wrong, Please contact admin or support team.",
}

# COMMAND ----------

# MAGIC %md
# MAGIC #04. RetrieverModel class
# MAGIC The RetrieverModel class is a custom class designed to retrieve relevant documents from a Snowflake database based on a given question, and generate a response using the retrieved documents as context.
# MAGIC Methods:
# MAGIC - \_\_init__: Initializes the RetrieverModel instance with a configuration file path.
# MAGIC - load_context: Calls reload_client and _get_filter_prompt to complete the initialization of RetrieverModel instance.
# MAGIC - reload_client: Reloads the OpenAI client using the configuration settings.
# MAGIC - _get_filter_prompt: Retrieves a filter prompt from the configuration settings.
# MAGIC - _get_active_sfconn: Establishes a connection to Snowflake using the predefined credentials.
# MAGIC - _get_query_embeddings: Generates query embeddings for a given question using the OpenAI client.
# MAGIC - _generate_query_filter_from_question: Generates a query filter for a given question using the OpenAI client.
# MAGIC - _get_query_records: Retrieves relevant documents from Snowflake based on a given question and query filter.
# MAGIC - _generate_answer_from_context: Generates a response to a given question using the retrieved documents as context.
# MAGIC - predict: Generates a response to a given question based on the provided context and model input.

# COMMAND ----------

class RetrieverModel(mlflow.pyfunc.PythonModel):
    def __init__(self, config: Dict) -> None:
        """
        Initializes the instance with a configuration file.

        This method reads the configuration file at the specified path and stores its contents in the instance's `config` attribute.
        The `filter_prompt` attribute is also initialized to `None`, which will be updated later when the context is loaded.

        Args:
            config (dict): The path to the configuration file.

        Returns:
            None

        Notes:
            - The configuration file must be in a format that can be read by the `read_config` function.
            - The instance's attributes are not fully initialized until the `load_context` method is called.
        """
        self.config = config
        self.filter_prompt = None

    def load_context(self, context) -> None:
        """
        Initializes the instance's context by reloading the OpenAI client and constructing the filter prompt.

        This method calls the `reload_client` method to reinitialize the OpenAI client with the configured API token and base URL, and then calls the `_get_filter_prompt` method to construct the filter prompt based on the configuration settings.

        Args:
            context: A PythonModelContext instance containing artifacts that the model can use to perform inference. (currently not used in this method)

        Returns:
            None

        Notes:
            - This method modifies the `client` and `filter_prompt` attributes of the instance.
            - The `config` dictionary must be properly initialized before calling this method.
        """
        self.reload_client()
        self._get_filter_prompt()

    def reload_client(self) -> None:
        """
        Reinitializes the OpenAI client with the configured API token and base URL.

        This method updates the `client` attribute of the instance with a new OpenAI client, using the API token and base URL specified in the instance's `config` dictionary.

        Args:
            None

        Returns:
            None
        """
        self.client = OpenAI(
            api_key=self.config["TOKEN"], base_url=self.config["endpoint_url_base"]
        )

    def _get_filter_prompt(self) -> None:
        """
        Retrieves and constructs a filter prompt based on the configuration settings.

        This method retrieves the filter prompt from the instance's `config` dictionary and appends unique values for each column specified in the `filter_settings` to the prompt. The unique values are fetched from the database using the active Snowflake connection.

        Args:
            None

        Returns:
            None

        Notes:
            - This method modifies the `filter_prompt` attribute of the instance.
            - The `filter_settings` and `query_settings` must be properly configured in the instance's `config` dictionary.
            - The database connection is closed after fetching the unique values.
        """
        self.filter_prompt = self.config["filter_settings"]["prompt"]

        conn = self._get_active_sfconn()
        curr = conn.cursor()

        # Fetch unique values for each column in the table specified in the `query_settings` dictionary.
        unique_val_col = self.config["filter_settings"]["unique_values"]
        for key in unique_val_col:
            curr.execute(
                f"""
                select distinct {key}
                FROM {self.config['query_settings']['db_name']}.{self.config['query_settings']['schema_name']}.{self.config['query_settings']['table_name']};
            """
            )
            docs = curr.fetchall()
            unique_val_col[key] = [doc[0] for doc in docs if (doc[0] is not None) and  (doc[0].lower()!= 'other')]
        conn.close()

        # Append unique values to the filter prompt
        unique_val_prompt = (
            "\nBelow is the JSON Dump of Unique Values for each columns: \n"
            + json.dumps(unique_val_col)
        )

        # Append unique values prompt and notes to the filter prompt
        self.filter_prompt = self.filter_prompt + unique_val_prompt + "\n" + self.config["filter_settings"]["notes"]

    def get_environment_settings(self) -> Dict:
        """
        Retrieve environment settings from the configuration.

        This method extracts various settings from the configuration and returns them as a dictionary.

        Returns:
            Dict: A dictionary containing the environment settings.

        Notes:
            - The returned dictionary includes settings for token, endpoint URL base, response model, filter model, embedding model, query, and Snowflake connection.
        """
        env_dict = {}
        env_dict['token'] = self.config["TOKEN"]
        env_dict['endpoint_url_base'] = self.config["endpoint_url_base"]
        env_dict['response_model_settings'] = self.config["response_model_settings"]
        env_dict['filter_model_settings'] = self.config["filter_model_settings"]
        env_dict['embedding_model_settings'] = self.config["embedding_model_settings"]
        env_dict['query_settings'] = self.config["query_settings"]
        env_dict['snowflake_settings'] = {
            "user": sfUser,
            "password": sfPwd,
            "account": account,
            "warehouse": warehouse,
            "role": role
        }
        return env_dict

    def _get_active_sfconn(self):
        """
        Establishes a connection to Snowflake using the predefined credentials.

        This method creates a new connection to Snowflake using the `snowflake.connector` library, with the specified user, password, account, warehouse, and role.

        Args:
            None

        Returns:
            A Snowflake connection object.

        Notes:
            - The connection parameters (user, password, account, warehouse, and role) must be properly defined and accessible.
            - The connection is not closed by this method; it is the caller's responsibility to close the connection when it is no longer needed.
        """
        conn = snowflake.connector.connect(
            user=sfUser, password=sfPwd, account=account, warehouse=warehouse, role=role
        )
        return conn

    def _get_query_embedding(self, query: str) -> List:
        """
        Generates embeddings for a given query using the OpenAI client.

        This method uses the OpenAI client to create embeddings for the specified query,
        using the embedding model specified in the instance's `config` dictionary.

        Args:
            query (str): The query for which to generate embeddings.

        Returns:
            A list of embeddings for the query.

        Notes:
            - The `embedding_model_settings` must be properly configured in the instance's `config` dictionary.
            - The logging level is set to INFO to log a success message when embeddings are generated successfully.
        """
        embeddings = self.client.embeddings.create(
            input=[query], model=self.config["embedding_model_settings"]["name"]
        )
        logging.info(f"Embedding generated successfully.")
        return embeddings.data[0].embedding

    def _generate_query_filter_from_question(self, question: str) -> str:
        """
        Generates a query filter for a given question using the OpenAI client.

        This method uses the OpenAI client to create a chat completion that generates a query filter based on the provided question and the instance's filter prompt. If the filter prompt is not set, it is generated by calling the `_get_filter_prompt` method.

        Args:
            question (str): The question for which to generate a query filter.

        Returns:
            A string representing the generated query filter.

        Notes:
            - The `filter_model_settings` must be properly configured in the instance's `config` dictionary.
            - The `filter_prompt` is generated lazily, i.e. only when it is needed.
        """
        if self.filter_prompt is None:
            self._get_filter_prompt()
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.filter_prompt,
                },
                {"role": "user", "content": f"Question:\n {question} \n\n Filter:"},
            ],
            model=self.config["filter_model_settings"]["name"],
            max_tokens=self.config["filter_model_settings"]["max_tokens"],
            temperature=self.config["filter_model_settings"]["temperature"]
        )
        return chat_completion.choices[0].message.content

    def _get_query_records(self, question: str) -> (List, str):
        """
        Retrieves records from Snowflake that match the given question.

        This method uses the OpenAI client to generate a query vector from the provided question, and then uses this vector to query the Snowflake database for records that match the query. The records are filtered based on a query filter generated from the question, and are ordered by their similarity score.

        Args:
            question (str): The question for which to retrieve records.

        Returns:
            docs (list): A list of records from the Snowflake database that match the query.
            filter (str): A string representing the query filter that was used to filter the records.

        Notes:
            - The `query_settings` must be properly configured in the instance's `config` dictionary.
            - If an exception occurs during the execution of the query with the filter, the method will retry the query without the filter.
            - The Snowflake connection is closed after the records are retrieved.
        """
        try:
            conn = self._get_active_sfconn()

            query_vec = self._get_query_embedding(question)
            vector_dim = len(query_vec)  # get the dimension of the vector
            filters = self._generate_query_filter_from_question(question)

            curr = conn.cursor()
            try:
                curr.execute(
                    f"""
                    select {self.config['query_settings']['vokol_col']}, {self.config['query_settings']['vokol_id_col']}, {self.config['query_settings']['date_col']}, {self.config['query_settings']['hcp_speciality_col']}, VECTOR_COSINE_SIMILARITY({self.config['query_settings']['embedding_col']}::VECTOR(FLOAT, {vector_dim}), {query_vec}::VECTOR(FLOAT, {vector_dim}))
                    AS similarity_score
                    FROM {self.config['query_settings']['db_name']}.{self.config['query_settings']['schema_name']}.{self.config['query_settings']['table_name']} where {filters}
                    ORDER BY similarity_score DESC
                    limit {self.config['query_settings']['top_k']};
                """
                )
                docs = curr.fetchall()

                # Raise an exception if no records are found from the filter query.
                if len(docs) == 0:
                    raise QueryFilterException('No records found for the given question. query filter: ' + filters)

            except QueryFilterException as e:
                logging.warning(f"Warning: {e}")
                logging.warning(f"Warning: Removing the filter from the query and retrying the query...")

                filters = ""
                curr.execute(
                    f"""
                    select {self.config['query_settings']['vokol_col']}, {self.config['query_settings']['vokol_id_col']}, {self.config['query_settings']['date_col']}, {self.config['query_settings']['hcp_speciality_col']}, VECTOR_COSINE_SIMILARITY({self.config['query_settings']['embedding_col']}::VECTOR(FLOAT, {vector_dim}), {query_vec}::VECTOR(FLOAT, {vector_dim}))
                    AS similarity_score
                    FROM {self.config['query_settings']['db_name']}.{self.config['query_settings']['schema_name']}.{self.config['query_settings']['table_name']}
                    ORDER BY similarity_score DESC
                    limit {self.config['query_settings']['top_k']};
                """
                )
                docs = curr.fetchall()
            except Exception as e:
                logging.error(f"Error: {e}")
                logging.error(f"Error: Removing the filter from the query and retrying the query...")

                filters = ""
                curr.execute(
                    f"""
                    select {self.config['query_settings']['vokol_col']}, {self.config['query_settings']['vokol_id_col']}, {self.config['query_settings']['date_col']}, {self.config['query_settings']['hcp_speciality_col']}, VECTOR_COSINE_SIMILARITY({self.config['query_settings']['embedding_col']}::VECTOR(FLOAT, {vector_dim}), {query_vec}::VECTOR(FLOAT, {vector_dim}))
                    AS similarity_score
                    FROM {self.config['query_settings']['db_name']}.{self.config['query_settings']['schema_name']}.{self.config['query_settings']['table_name']}
                    ORDER BY similarity_score DESC
                    limit {self.config['query_settings']['top_k']};
                """
                )
                docs = curr.fetchall()
            conn.close()
            return docs, filters
        except Exception as e:
            raise e

    def _generate_answer_from_context(self, context: str, question: str) -> str:
        """
        Generates a response to a given question based on the provided context.

        This method uses the OpenAI client to create a chat completion that generates a response based on the provided context and question. The response is generated using a prompt specified in the instance's `config` dictionary.

        Args:
            context (str): The context in which to answer the question.
            question (str): The question to answer.

        Returns:
            A string representing the generated response.

        Notes:
            - The `qa_settings` and `response_model_settings` must be properly configured in the instance's `config` dictionary.
            - The response is generated based on the provided context and question, and may not always be accurate or relevant.
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": self.config["qa_settings"]["prompt"],
                },
                {
                    "role": "user",
                    "content": f"Context: \n {context} \n\n Question:\n {question} \n\n Answer:",
                },
            ],
            model=self.config["response_model_settings"]["name"],
            max_tokens=self.config["response_model_settings"]["max_tokens"],
        )
        return chat_completion.choices[0].message.content

    def predict(self, context, model_input) -> (str, str):
        """
        Generates a response to a given question based on the provided model input.

        This method reloads the OpenAI client, retrieves relevant documents from Snowflake based on the model input, and then generates a response using the retrieved documents as context.

        Args:
            context: Not used in this implementation.
            model_input (pd.DataFrame): A pandas DataFrame containing the input question.

        Returns:
            response (str): A string representing the generated response, including the IDs of the relevant documents.
            filter (str): A string representing the query filter that was used to filter the records.

        Notes:
            - The `model_input` is expected to be a pandas DataFrame with a single row and column.
            - If no relevant documents are found, the response will indicate that no relevant documents were found.
            - The response includes the IDs of the relevant documents, separated by commas.
        """
        response = {
                'result': '',
                'ref_vokol_data': {},
                'query_filter': '',
                'status_message': 'Success'
            }
        records_df_columns=[self.config['query_settings']['vokol_col'], self.config['query_settings']['vokol_id_col'], self.config['query_settings']['date_col'], self.config['query_settings']['hcp_speciality_col'], 'Score']
        try:
            self.reload_client()
            logging.info(f"Client reloaded successfully.")

            # Adding Zuranolone or Zurzuvae to the question to improve the results.
            question = re.sub(r'\b(Zuranolone|Zurzuvae)\b', 'Zuranolone or Zurzuvae', model_input[0].iloc[0], flags=re.IGNORECASE)

            # Retrieve relevant records from Snowflake based on the model input.
            docs, filter = self._get_query_records(question)

            # Convert the retrieved records to a pandas DataFrame.
            records_df = pd.DataFrame(docs, columns=records_df_columns)
            records_df.drop(['Score'], axis=1, inplace=True)
            records_df[self.config['query_settings']['date_col']] = pd.to_datetime(records_df[self.config['query_settings']['date_col']]).dt.strftime('%d-%m-%Y').astype(str)

            if len(docs) > 0:
                vokols = [doc[0] for doc in docs]
                context = " ".join(vokols)
            else:
                context = self.config['default_context']

            # Generate a result_answer based on the retrieved records.
            result_answer = self._generate_answer_from_context(context, question).replace('**', '')
            logging.info(f"Answer generated!")
            response['result'] = result_answer
            response['ref_vokol_data'] = records_df.to_json(orient="records")
            response['query_filter'] = filter
        except Exception as e:
            logging.error(f"Error: {e}")
            response['result'] = self.config['default_error_message']
            response['status_message'] = 'Failure'
        return response

# COMMAND ----------

# MAGIC %md
# MAGIC #05. Log/Save Model
# MAGIC - Load config file
# MAGIC - Create RetrieverModel using config
# MAGIC - Set inference signature
# MAGIC - Log model and get run_id

# COMMAND ----------

user_name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment(f'/Users/{user_name}/RAG_PIPELINE')

# COMMAND ----------

model = RetrieverModel(rag_config)

# COMMAND ----------

signature = infer_signature('Hi', {'context': 'string', 'filter': 'string'})
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        rag_config['model_name'], python_model=model, signature = signature
    )
    run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC #06. Load/Test Model
# MAGIC - Load logged model using run_id
# MAGIC - make prediction using the model

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/{rag_config['model_name']}")

# COMMAND ----------

loaded_model.unwrap_python_model().get_environment_settings()

# COMMAND ----------

response = loaded_model.predict('What is the opinion of KOLs on insurance coverage for Zuranolone in 2024?')

print(response['result'])
print(response['query_filter'])
print(response['ref_vokol_data'])
print(response['status_message'])

# COMMAND ----------

