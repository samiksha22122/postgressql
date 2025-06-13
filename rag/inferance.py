# Databricks notebook source
# MAGIC %md
# MAGIC #01. Importing Libraries

# COMMAND ----------

import logging
import yaml
import os
import datetime
import uuid
import json
import pytz
from typing import Dict, List, Optional

import pandas as pd
import snowflake.connector
from openai import OpenAI

import mlflow
from mlflow import MlflowClient

from pyspark.sql.types import TimestampType, DecimalType, StringType, BooleanType
from pyspark.sql import SparkSession

import re
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC #02. Importing Notebooks
# MAGIC - snowflake_credentials: To load snowflake credentials
# MAGIC - Load databricks user credentials from environment variables
# MAGIC - SEMANTIC_CHUNKING: Importing Semantic_Chunking class for splitting vokols based on matching intent of sentences.

# COMMAND ----------

# MAGIC %run ../../../../common/snowflake_credentials

# COMMAND ----------

sfUser = dbutils.secrets.get("commercial", "snowflake_comm_medical_datahub_etl_user")
sfPwd = dbutils.secrets.get("commercial", "snowflake_comm_medical_datahub_etl_pwd")

# COMMAND ----------

dbcEnv = os.getenv("dbc_environment")
if dbcEnv == 'DEV':
  url = 'devtest'
else:
  url = dbcEnv.lower()

# COMMAND ----------

session = SparkSession.builder.appName("VoKOL Inference").getOrCreate()

# COMMAND ----------

# MAGIC %run ./utils/SEMANTIC_CHUNKING

# COMMAND ----------

# MAGIC %md
# MAGIC #03. Setting up configs for Notebook
# MAGIC

# COMMAND ----------

inference_config = {
    "data_source": {
        "db_name": f"SCUDE_MEDICAL_AFFAIRS_{dbcEnv}",
        "schema_name": "DEP_CONFORMED",
        "table_name": "VOICE_OF_KOL",
    },
    "data_destination": {
        "db_name": f"SCUDE_MEDICAL_AFFAIRS_{dbcEnv}",
        "schema_name": "DEP_CONFORMED",
        "table_name": "VOKOL_SENTIMENT_ANALYSIS",
    },
    "column_ref": {"date_col": "VCRM_CREATED_DATE", "start_date": "1-1-2023"},
    "model_source": {
        "db_name": f"commercial_{dbcEnv}",
        "schema_name": "medical_affair_core",
        "atepc_model": "aspect_sentiment_classification_4",
        "classification_model": "vokol_classfication_model",
    },
    "sentence_embedding_model": "cambridgeltl/BioRedditBERT-uncased",
    "embedding_model_name": "databricks-bge-large-en",
    "serving_endpoint_base_url": f"https://sagerx-aws-{url}-comm.cloud.databricks.com/serving-endpoints",
}

# COMMAND ----------

theme_topic = {"BS#Avaibility": ('Barriers and Solutions', 'Availability Barriers'),
"BS#Breastfeeding": ('Barriers and Solutions', 'Breastfeeding'),
"BS#Cost": ('Barriers and Solutions', 'Cost and Coverage Barriers'),
"BS#Durability": ('Barriers and Solutions', 'Durability'),
"BS#Patient": ('Barriers and Solutions', 'Patient Barriers'),
"CPD#Monoaminergic": ('Competitor Products and Data', 'Pharmacological Treatments'),
"Education": ('Barriers and Solutions', 'Education'),
"PTUC#Comorbidities": ('Patient Journey', 'Comorbidities'),
"PTUC#Diagnosis": ('Patient Journey', 'Screening and Diagnosis'),
"SBPD#Administration": ('Sage and Biogen: Products & Data', 'Drug Dosing and Administration'),
"SBPD#Efficacy": ('Sage and Biogen: Products & Data', 'Efficacy'),
"SBPD#Safety": ('Sage and Biogen: Products & Data', 'Safety/Side Effects'),
"SBPD#Study": ('Sage and Biogen: Products & Data', 'Clinical Trials - Study'),
"SBSM#GABA": ('Sage and Biogen: Science and Mechanisms - MOA', 'GABA & Neuroactive Steroids'),
"other": ('Other', 'Other')}

# COMMAND ----------

# MAGIC %md
# MAGIC #04. VokolInference class
# MAGIC The VokolInference class is a custom class designed to extract Aspect, Sentiment, Theme and Topic from the VoKOLs for the previous month data sourced from the snowflake.
# MAGIC Methods:
# MAGIC - \_\_init__: Initializes the VokolInference instance with a configuration dictionary.
# MAGIC - _is_initial_run: Check if this is the initial run of the process by verifying the existence of data in the destination table.
# MAGIC - _load_mlflow_models: Load an MLflow model from the Unity Catalog using the specified model name.
# MAGIC - _load_champion_models: Load the champion models for aspect extraction, sentiment analysis, and text classification.
# MAGIC - _previous_month: Returns the year and month of the previous month based on the given query date.
# MAGIC - _get_active_sfconn: Establishes a connection to Snowflake using the predefined credentials.
# MAGIC - _get_source_data: Retrieves data from the source table based on the given query date.
# MAGIC - _get_cleaned_aspects: Cleans symbols and stop words from aspects.
# MAGIC - _get_sentiment: Determine the overall sentiment from a list of sentiment classifications.
# MAGIC - _get_aspect_sentiment_theme_topic: Extract aspect, sentiment, theme, and topic information from a given text.
# MAGIC - predict: Predict aspect, sentiment, theme, and topic information for customer feedback data on a given date.

# COMMAND ----------

class VokolInference:
    def __init__(self, config: Dict) -> None:
        """
        Initialize the class instance with the provided configuration.

        This method loads the champion models, checks if this is the initial run, and sets up the instance variables.

        Args:
            config (Dict): A dictionary containing the configuration settings.

        Returns:
            None
        """
        self.config = config
        self._load_champion_models()
        self.stop_words = set(stopwords.words('english'))
        self.initial_run = self._is_initial_run()
        logging.info(f"Initial run: {self.initial_run}")

    def _is_initial_run(self) -> bool:
        """
        Check if this is the initial run of the process by verifying the existence of data in the destination table.

        This function queries the destination table to check if it contains any data. If the table is empty, it returns True, indicating that this is the initial run. Otherwise, it returns False.

        Returns:
            bool: True if this is the initial run, False otherwise.

        Raises:
            Exception: If an error occurs while checking the destination table.
        """
        destination_table_name = f"{self.config['data_destination']['db_name']}.{self.config['data_destination']['schema_name']}.{self.config['data_destination']['table_name']}"
        get_col_names_query = f"select * from {destination_table_name} limit 5"

        try:
            conn = self._get_active_sfconn()
            df = pd.read_sql(get_col_names_query,conn)
            if df.shape[0]==0:
                return True
            return False
        except Exception as e:
            raise Exception(f"Error while checking for the summary table: {e}")

    def _load_mlflow_models(self, model_name:str):
        """
        Load an MLflow model from the Unity Catalog using the specified model name.

        This function loads the champion version of the model, as denoted by the "@Champion" alias.

        Args:
            model_name (str): The name of the model to load, in the format "database.schema.model_name".

        Returns:
            mlflow.pyfunc.PyFuncModel: The loaded model as a PyFuncModel.

        Raises:
            mlflow.exceptions.MlflowException: If the model cannot be loaded.
        """
        # Specify the model URI using the Unity Catalog format
        model_uri = f"models:/{model_name}@Champion"

        # Load the model as a PyFuncModel
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        return loaded_model

    def _load_champion_models(self) -> None:
        """
        Load the champion models for aspect extraction, sentiment analysis, and text classification.

        This function uses the MLflow client to load the models specified in the configuration file.
        The loaded models are then assigned to instance variables for use in other methods.

        The following models are loaded:
        - Aspect extraction model (ATEPC)
        - Text classification model

        The instance variables populated by this method are:
        - `self.chunking`: An instance of the `Semantic_Chunking` class, used for sentence embedding.
        - `self.aspect_extractor`: The loaded ATEPC model.
        - `self.label_classifier`: The loaded text classification model.

        Returns:
            None
        """
        client = MlflowClient()

        atepc_model_name = f"{self.config['model_source']['db_name']}.{self.config['model_source']['schema_name']}.{self.config['model_source']['atepc_model']}"
        classification_model_name = f"{self.config['model_source']['db_name']}.{self.config['model_source']['schema_name']}.{self.config['model_source']['classification_model']}"

        self.chunking = Semantic_Chunking(self.config["sentence_embedding_model"])
        self.aspect_extractor = self._load_mlflow_models(atepc_model_name)
        self.label_classifier = self._load_mlflow_models(classification_model_name)

    def _previous_month(self, query_date: datetime.date) -> (int, int):
        """
        Returns the year and month of the previous month based on the given query date.

        Args:
            query_date: The date for which to find the previous month.

        Returns:
            A tuple containing the year and month (1-12) of the previous month.

        Note:
            If the query date is in January, the previous month is considered to be December of the previous year.
        """
        if query_date.month == 1:
            req_date = datetime.date(query_date.year - 1, 12, 31)
        else:
            req_date = datetime.date(query_date.year, query_date.month - 1, 1)
        return req_date.year, req_date.month

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

    def _get_source_data(self, query_date: datetime.date) -> Optional[pd.DataFrame]:
        """
        Retrieves data from the source table based on the given query date.

        Args:
            query_date: The date for which to retrieve data.

        Returns:
            A pandas DataFrame containing the retrieved data, or None if an error occurs.

        Raises:
            Exception: If an error occurs while reading data from the source table.

        Note:
            The data is retrieved using a SQL query constructed by the `_get_source_data_query` method. The query is executed using a Snowflake connection obtained from the `_get_active_sfconn` method.
        """
        try:
            source_table_name = f"{self.config['data_source']['db_name']}.{self.config['data_source']['schema_name']}.{self.config['data_source']['table_name']}"

            date_col = self.config["column_ref"]["date_col"]
            start_date = self.config["column_ref"]["start_date"]

            year, month = self._previous_month(query_date)
            data_query = f"select * from {source_table_name} where year({date_col}) = {year} and month({date_col}) = {month}"

            if self.initial_run:
                data_query = f"select * from {source_table_name} where {date_col}  >= TO_DATE('{start_date}', 'DD-MM-YYYY') and ((year({date_col}) < {year}) or (year({date_col}) = {year} and month({date_col}) <= {month}))"

            conn = self._get_active_sfconn()
            df = pd.read_sql(data_query, conn)
            return df
        except Exception as e:
            raise Exception(
                f"Error reading data from {source_table_name} table. The following exception raised {e}"
            )

    def _get_cleaned_aspects(self, aspects: List[str], sentiments: List[str]) -> (List[str],List[str]):
        """
        Cleans aspects and removes stop words from it and returns cleanded aspects and sentiments

        Args:
            aspects (List[str]): A list of aspects corresponding to the vokols.
            sentiments (List[str]): A list of sentiments corresponding to the vokols.

        Returns:
            Cleaned aspects and sentiments
        """
        cleaned_aspects=[]
        cleaned_sentiments=[]

        for j in range(len(aspects)):

            cleaned_sub_aspects=[]
            cleaned_sub_sentiments=[]

            sub_list_aspects=aspects[j]
            sub_list_sentiments=sentiments[j]

            for i in range(len(sub_list_aspects)):

                x= re.sub(r'[^a-zA-Z0-9\s]', '', sub_list_aspects[i]).lower()

                x = ' '.join(word for word in x.split() if word not in self.stop_words)

                if x:
                    cleaned_sub_aspects.append(x)
                    cleaned_sub_sentiments.append(sub_list_sentiments[i])

            cleaned_aspects.append(cleaned_sub_aspects)
            cleaned_sentiments.append(cleaned_sub_sentiments)

        return cleaned_aspects,cleaned_sentiments


    def _get_sentiment(self, sentiments: List[str]) -> str:
        """
        Determine the overall sentiment from a list of sentiment classifications.

        Args:
            sentiments (List[str]): A list of sentiment classifications, where each classification is one of "Negative", "Neutral", or "Positive".

        Returns:
            str: The overall sentiment, which is "Positive" if the number of positive classifications exceeds the number of negative classifications, "Negative" if the number of negative classifications exceeds the number of positive classifications, and "Neutral" otherwise.
        """
        sentiments = str(sentiments)
        negative = sentiments.count("Negative")
        neutral = sentiments.count("Neutral")
        positive = sentiments.count("Positive")
        if positive > negative:
            return "Positive"
        elif positive < negative:
            return "Negative"
        else:
            return "Neutral"

    def _get_aspect_sentiment_theme_topic(
        self, text: str
    ) -> (List[str], List[str], List[str], List[str]):
        """
        Extract aspect, sentiment, theme, and topic information from a given text.

        This function performs the following steps:
        1. Chunk the input text into smaller units.
        2. Extract aspect and sentiment information from each chunk.
        3. Merge chunks based on aspect information.
        4. Classify each chunk into a theme (label).
        5. Merge chunks based on label information.
        6. Determine the overall sentiment for each chunk.

        Args:
            text (str): The input text to extract information from.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: A tuple containing four lists:
                - chunks (List[str]): The text chunks after merging.
                - labels (List[str]): The theme (label) for each chunk.
                - aspects (List[str]): The aspect information for each chunk.
                - sentiments (List[str]): The overall sentiment for each chunk.
        """
        chunks = self.chunking.create_chunks(text)
        atepc = self.aspect_extractor.predict(chunks)
        aspects = atepc["Aspect"]
        sentiments = atepc["Sentiment"]

        aspects,sentiments=self._get_cleaned_aspects(
            aspects,sentiments
        )

        chunks, aspects, sentiments = self.chunking.aspect_based_merge_chunks(
            chunks, aspects, sentiments
        )

        labels = self.label_classifier.predict(pd.DataFrame(chunks))
        labels = labels["label"].tolist()

        chunks, labels, aspects, sentiments = self.chunking.label_based_merge_chunks(
            chunks, labels, aspects, sentiments
        )

        sentiments = [self._get_sentiment(i) for i in sentiments]

        return chunks, labels, aspects, sentiments

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df['THERAPEUTIC_FOCUS'] = df['THERAPEUTIC_FOCUS'].apply(lambda x: 'MDD/Bipolar/Sleep' if x =='MDD' else x)
        df['VOKOL_NAME'] = df.apply(lambda x: x['VOKOL_ID'] if x['VOKOL_NAME'] is None else x['VOKOL_NAME'], axis = 1)

        allowed_therapeutic_focus = ['MDD/Bipolar/Sleep', 'PPD']
        df = df[df['THERAPEUTIC_FOCUS'].isin(allowed_therapeutic_focus)]

        df['COMPOUND_PRODUCT'] = df['COMPOUND_PRODUCT'].apply(lambda x: 'Zuranolone' if x =='Zurzuvae' else x)
        df['COMPOUND_PRODUCT'] = df['COMPOUND_PRODUCT'].apply(lambda x: 'Brexanolone' if x =='217 and Brexanolone' else x)

        return df

    def predict(self, day: int, month: int, year: int) -> pd.DataFrame:
        """
        Predict aspect, sentiment, theme, and topic information for customer feedback data on a given date.

        This function performs the following steps:
        1. Retrieves the source data for the given date.
        2. Extracts aspect, sentiment, theme, and topic information from the customer feedback text using the `_get_aspect_sentiment_theme_topic` method.
        3. Explodes the resulting DataFrame to create separate rows for each chunk of text.
        4. Maps the predicted labels to theme and topic categories.
        5. Assigns a unique prediction ID to each row.

        Args:
            day (int): The day of the month (1-31) for which to retrieve data.
            month (int): The month (1-12) for which to retrieve data.
            year (int): The year for which to retrieve data.

        Returns:
            pd.DataFrame: A DataFrame containing the predicted aspect, sentiment, theme, and topic information for each chunk of customer feedback text.

        Raises:
            AssertionError: If no data is found for the given date and grain.
        """
        query_date = datetime.date(year, month, day)
        vokol_df = self._get_source_data(query_date)

        vokol_df = self._preprocess_data(vokol_df)
        ids_with_empty_voice_of_customer = vokol_df[vokol_df["VOICE_OF_CUSTOMER"].isna()]['VOKOL_NAME'].to_list()
        logging.warning(f'IDs with empty voice of customer: {ids_with_empty_voice_of_customer} \nAbove list of vokols are not used for inference.')
        vokol_df = vokol_df[~vokol_df["VOICE_OF_CUSTOMER"].isna()].copy()

        if len(vokol_df) == 0:
            dbutils.notebook.exit("No data found for the given date.")

        vokol_df[
            ["VOKOL_CHUNK", "labels", "PREDICTED_ASPECT", "PREDICTED_SENTIMENT"]
        ] = vokol_df.apply(
            lambda x: self._get_aspect_sentiment_theme_topic(x["VOICE_OF_CUSTOMER"]),
            axis=1,
            result_type="expand",
        )

        vokol_df = vokol_df.explode(
            ["VOKOL_CHUNK", "labels", "PREDICTED_ASPECT", "PREDICTED_SENTIMENT"]
        )

        vokol_df[["PREDICTED_THEME", "PREDICTED_TOPIC"]] = vokol_df.apply(
            lambda x: theme_topic[x["labels"]], axis="columns", result_type="expand"
        )
        vokol_df = vokol_df.drop(columns=["labels"])

        vokol_df["PREDICTION_ID"] = vokol_df.apply(lambda _: uuid.uuid4(), axis=1)
        vokol_df["PREDICTION_ID"] = vokol_df["PREDICTION_ID"].astype(str)

        return vokol_df

# COMMAND ----------

# Create a VokolInference instance to use for inference
vokol_inference = VokolInference(inference_config)

# COMMAND ----------

run_date = datetime.datetime.today()

# COMMAND ----------

vokol_sentiment_analysis_df = vokol_inference.predict(run_date.day,run_date.month,run_date.year)

# COMMAND ----------

# MAGIC %md
# MAGIC #05. Post-processing and Ingesting to Snowflake
# MAGIC Below code has the following functionalities:
# MAGIC - Generate Vokol Embeddings
# MAGIC - Add metadata columns
# MAGIC - Fetch required columns from the destination table
# MAGIC - Write data to the snowflake

# COMMAND ----------

DATABRICKS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# COMMAND ----------

client = OpenAI(
  api_key=DATABRICKS_TOKEN,
  base_url= inference_config['serving_endpoint_base_url']
)

# COMMAND ----------

def get_embeddings(text, model_name):
  embeddings = client.embeddings.create(
  input=text,
  model=model_name
  )
  return embeddings.data[0].embedding

# COMMAND ----------

vec_length = len(get_embeddings('hello', inference_config['embedding_model_name']))
print(vec_length)

# COMMAND ----------

vokol_sentiment_analysis_df['VECTORIZED_VOKOL'] = vokol_sentiment_analysis_df['VOICE_OF_CUSTOMER'].apply(lambda x : get_embeddings(x, inference_config['embedding_model_name']))

# COMMAND ----------

# Add metadata columns to the dataframe
widget_values = dbutils.widgets.getAll()

vokol_sentiment_analysis_df['RECORD_ID'] = vokol_sentiment_analysis_df['PREDICTION_ID'].astype(str)
vokol_sentiment_analysis_df['DATABRICKS_RUNID'] = widget_values.get('databricks_run_id', '')
vokol_sentiment_analysis_df['BATCH_ID'] = widget_values.get('batch_id', '')
vokol_sentiment_analysis_df['JOB_ID'] = widget_values.get('job_id', '')
vokol_sentiment_analysis_df['INSERT_DATE_TIME'] = pd.Timestamp('now', tz=pytz.utc)
vokol_sentiment_analysis_df['UPDATE_DATE_TIME'] = None

# COMMAND ----------

# Fetch required columns from the destination table
destination_table_name = f"{inference_config['data_destination']['db_name']}.{inference_config['data_destination']['schema_name']}.{inference_config['data_destination']['table_name']}"
conn = snowflake.connector.connect(
            user=sfUser, password=sfPwd, account=account, warehouse=warehouse, role=role
        )
get_col_names_query = f"show columns in table {destination_table_name}"
df = pd.read_sql(get_col_names_query,conn)
required_cols = df['column_name'].to_list()

# COMMAND ----------

vokol_sentiment_analysis_df.shape

# COMMAND ----------

# Filter the dataframe based on the required columns
vokol_sentiment_analysis_df = vokol_sentiment_analysis_df[required_cols]

# COMMAND ----------

vokol_sentiment_analysis_df.head(10)

# COMMAND ----------

# Conver the Pandas dataframe to Spark DataFrame, and set UPDATE_DATE_TIME column as timestamp
vokol_sentiment_analysis_spark_df = session.createDataFrame(vokol_sentiment_analysis_df)

vokol_sentiment_analysis_spark_df = vokol_sentiment_analysis_spark_df.withColumn("UPDATE_DATE_TIME", vokol_sentiment_analysis_spark_df["UPDATE_DATE_TIME"].cast(TimestampType()))
vokol_sentiment_analysis_spark_df = vokol_sentiment_analysis_spark_df.withColumn("VEEVA_NETWORK_ID", vokol_sentiment_analysis_spark_df["VEEVA_NETWORK_ID"].cast(DecimalType(38,0)))

# COMMAND ----------

vokol_sentiment_analysis_spark_df = vokol_sentiment_analysis_spark_df.withColumn("HCP_SPECIALTY", vokol_sentiment_analysis_spark_df["HCP_SPECIALTY"].cast(StringType()))
vokol_sentiment_analysis_spark_df = vokol_sentiment_analysis_spark_df.withColumn("COMPOUND_PRODUCT", vokol_sentiment_analysis_spark_df["COMPOUND_PRODUCT"].cast(StringType()))
vokol_sentiment_analysis_spark_df = vokol_sentiment_analysis_spark_df.withColumn("ADVERSE_EVENT_RELATED", vokol_sentiment_analysis_spark_df["ADVERSE_EVENT_RELATED"].cast(StringType()))

vokol_sentiment_analysis_spark_df = vokol_sentiment_analysis_spark_df.withColumn("TIER_1_CONGRESS_VO_KOL", vokol_sentiment_analysis_spark_df["TIER_1_CONGRESS_VO_KOL"].cast(BooleanType()))

# COMMAND ----------

vokol_sentiment_analysis_spark_df.dtypes

# COMMAND ----------

# Write the data to Snowflake
options = {
  "sfUrl": f"https://{account}.snowflakecomputing.com",
  "sfUser": sfUser,
  "sfPassword": sfPwd,
  "sfDatabase": inference_config['data_destination']['db_name'],
  "sfSchema":inference_config['data_destination']['schema_name'],
  "sfWarehouse": warehouse,
  "sfrole":role,
  "dbtable": inference_config['data_destination']['table_name'],
  "column_mapping": 'name'
}

vokol_sentiment_analysis_spark_df.write.format("snowflake") \
.options(**options)\
.mode('append')\
.save()
