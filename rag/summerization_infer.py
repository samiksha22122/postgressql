# Databricks notebook source
# MAGIC %md
# MAGIC #01. Importing Libraries

# COMMAND ----------

import configparser
import logging
import snowflake.connector
import yaml
import os
import datetime
import pandas as pd
import numpy as np
import mlflow
from typing import Dict, List, Optional
import uuid
import json
import pytz
from pyspark.sql.types import TimestampType
from pyspark.sql import SparkSession

# COMMAND ----------

# MAGIC %md
# MAGIC #02. Importing Notebooks
# MAGIC - setup spark session to ingest data to snowflake
# MAGIC - snowflake_credentials: To load snowflake credentials
# MAGIC - Load databricks user credentials from environment variables
# MAGIC - ORCHASTRATOR: Importing Orchastrator class for generating summaries.

# COMMAND ----------

session = SparkSession.builder.appName("Summarization Inference").getOrCreate()

# COMMAND ----------

logging.basicConfig(level=logging.WARNING)

# COMMAND ----------

# MAGIC %run "./ORCHESTRATOR"

# COMMAND ----------

# MAGIC %run "../../../../common/snowflake_credentials"

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

# MAGIC %md
# MAGIC #03. Setting up configs for Notebook

# COMMAND ----------

llm_config = {
    "endpoint_url_base": f"https://sagerx-aws-{url}-comm.cloud.databricks.com/serving-endpoints",
    "model_settings": {
        "name": "databricks-meta-llama-3-70b-instruct",
        "max_tokens": 8000,
        "max_response_tokens": 1000,
    },
}
general_configs = {
    "max_retires": 5,
    "classifier_model_uri": f"models:/commercial_{dbcEnv.lower()}.medical_affair_core.bart-large-mnli-pipe_1@champion",
    "spacy_model_uri": f"models:/commercial_{dbcEnv.lower()}.medical_affair_core.spacy-english-model_1@champion",
    "evaluate_summary" : False
}

# COMMAND ----------

summarization_inference_config = {
    "data_source": {
        "db_name": f"SCUDE_MEDICAL_AFFAIRS_{dbcEnv}",
        "schema_name": "DEP_CONFORMED",
        "table_name": "VOKOL_SENTIMENT_ANALYSIS",
    },
    "data_destination": {
        "db_name": f"SCUDE_MEDICAL_AFFAIRS_{dbcEnv}",
        "schema_name": "DEP_CONFORMED",
        "summary_table_name": "VOKOL_SUMMARY",
        "xref_table_name": "VOKOL_SA_SUMMARY_XREF",
    },
    "column_ref": {"date_col": "VCRM_CREATED_DATE",
                   "start_date": "1-1-2023"},
}

# COMMAND ----------

# MAGIC %md
# MAGIC #04. Summary Generation Prompt

# COMMAND ----------

map_prompt = """Your a Medical Consultant who reports to MD of medical company.Your job is to get actions and insights from the medical opinion  of a drug provided by a key opinion leader(KOL).The opinion given may contain past feedback which are called insights and requests,suggestions and asks by the KOLs for the company to imporve which are called actions.Find out what are the actions and insights from the below opinion.And follow the guidelines.

"{text}"

Guidelines:
1.The person giving the opinion is called KOL so use words like 'KOL Suggested', 'KOL feels', the language should be more like a suggestion or polite mentioning. It must be in professional language and executive level.
2.There is no rule like that opinion contains both actions and insights, both actions and insigths might related and might not be.
3. Topic gives you the context for this discussion.
4.Pass the id along witht the summary at the beginning ID:id in a new line.
5.Avoid introducing unrelated information or fabrications.
6.Actions are those related to sage/biogen, those which can be taken care by sage/biogen and not those related to paitents, these actions must be a request for sage to do it in future. Focus on actionable items like requests specifically given by the KOLs and they cannot be any past things.Actions must be specifically asked by KOL.Actions must be directed sage/biogen.
7.Do not create any actions based on the insights, actions must be rquests mentioned by the KOLs and not be created by you,these will help you to find actions "it would be nice","please provide", "please look into" ,etc.
8.A point cannot simultaneously be mentioned as an action and  as an insight,it can inclued in only one of them.
9.Opinion sent is ANONYMIZED, it is sent between <>,example of ANONYMIZED data <ANONYMIZED> and <ANONYMIZED_PERSON_NAME>, never inclued these in the output
9. There are place holders like <ANONYMIZED_PERSON_NAME>,<ANONYMIZED> for sensitive data, dont inclued them in the output.

Additional Context:
Sage/biogen is a health care provider working on drugs and treatments for maternal mental health and postpartum depression.

SUMMARY:"""

# Define the reduce step prompt
reduce_prompt= """Your a Medical Consultant who reports to MD of medical company.Your job is to summarize all these insights into once consie paragraph and all the actions into another consie paragrah. You have to decide what to show,you can make this decision by knowing what's most important and what's mentioned frequently. Try to send as consie as possible summary.

"{text}"

Guidelines:
1.The persons giving the opinion are called KOLs so use words like 'KOLs suggested', 'KOLs feel', the language should be more like a suggestion or polite mentioning.Actions must be directed to sage.
2.Send the ids of the insights and actions, at the end as a list.
3.Group the related actions/insigths together.
4.Do not create any actions based on the insights, actions must be mentioned by the KOLs and not be created.
5.Avoid introducing unrelated information or fabrications.
6.A point cannot simultaneously be mentioned as an action and  as an insight,it can inclued in only one of them.
7.Send the output in json format, with labels Insights,Actions,Insights_ID,Actions_ID,in between ``` ```.
8.If there are no insights or actions, send "" for for that.and not ids then send [].
9.Avoid providing obvious and generic insight and actions that are applicalbe to any drug manufacturing company.Incase of no sutabile insights and actions send "" for insights/actions keys.

Additional Context:
Sage/biogen  is a health care provider working on drugs and treatments for maternal mental health and postpartum depression.

FINAL SUMMARY:"""

# COMMAND ----------

# MAGIC %md
# MAGIC #05. SummarizationInference class
# MAGIC The SummarizationInference class is a custom class designed to generate monthly or quarterly Insights and Actions based on the grain, from the data sourced from the snowflake.
# MAGIC Methods:
# MAGIC - \_\_init__: Initialize the class instance with the provided configuration, grain, and other settings.
# MAGIC - _is_initial_run: Check if this is the initial run of the process by verifying the existence of data in the summary table.
# MAGIC - _previous_quarter: Returns the year and quarter of the previous quarter based on the given query date.
# MAGIC - _previous_month: Returns the year and month of the previous month based on the given query date.
# MAGIC - _get_source_data_query: Returns a SQL query string to retrieve data from the source table based on the given query date and grain.
# MAGIC - _get_active_sfconn: Establishes a connection to Snowflake using the predefined credentials.
# MAGIC - _get_source_data: Retrieves data from the source table based on the given query date.
# MAGIC - _generate_summary: Generates a summary of the given DataFrame using the Orchestrator class.
# MAGIC - _get_vokol_summary_df: Combines the insights and actions DataFrames into a single summary DataFrame.
# MAGIC - _restructure_df_based_on_schema: Restructures the given DataFrames based on the schema and grain.
# MAGIC - predict: generate monthly or quarterly summaries based on the grain provided for a given period of time.

# COMMAND ----------

class SummarizationInference:
    def __init__(self, config: Dict, grain: str, llm_config:Dict, general_configs:Dict) -> None:
        """
        Initialize the class instance with the provided configuration, grain, and other settings.

        Args:
            config (Dict): A dictionary containing the configuration settings.
            grain (str): The grain of the data, either "M" for monthly or "Q" for quarterly.
            llm_config (Dict): A dictionary containing the configuration settings for the LLM.
            general_configs (Dict): A dictionary containing general configuration settings.

        Raises:
            ValueError: If the grain is not either "M" or "Q".

        Returns:
            None
        """
        self.config = config
        if grain not in ["M", "Q"]:
            raise ValueError(
                "Incorrect grain passed. Grain needs to be either M for monthly or Q for Quartely"
            )
        self.grain = grain
        self.llm_config = llm_config
        self.general_configs = general_configs
        self.initial_run = self._is_initial_run()
        logging.info(f"Initial run: {self.initial_run}")

    def _is_initial_run(self) -> bool:
        """
        Check if this is the initial run of the process by verifying the existence of data in the summary table.

        This method checks if the summary table exists and if it contains data for the specified grain (monthly or quarterly).
        If the table does not exist or does not contain data for the specified grain, it returns True, indicating that this is the initial run.

        Returns:
            bool: True if this is the initial run, False otherwise.

        Raises:
            Exception: If an error occurs while checking the summary table.
        """
        destination_table_name = f"{self.config['data_destination']['db_name']}.{self.config['data_destination']['schema_name']}.{self.config['data_destination']['summary_table_name']}"
        get_col_names_query = f"show columns in table {destination_table_name}"

        try:
            conn = self._get_active_sfconn()

            df = pd.read_sql(get_col_names_query,conn)
            cols_to_keep = df['column_name'].to_list()

            if "SUMMARY_TIME_FRAME" in cols_to_keep:
                get_unique_time_frame_query = f"select distinct(SUMMARY_TIME_FRAME) from {destination_table_name}"
                df = pd.read_sql(get_unique_time_frame_query,conn)['SUMMARY_TIME_FRAME'].to_list()
                if self.grain=='M' and 'Monthly' not in df:
                    return True
                elif self.grain=='Q' and 'Quarterly' not in df:
                    return True
        except Exception as e:
            raise Exception(f"Error while checking for the summary table: {e}")
        return False

    def _previous_quarter(self, query_date: datetime.date) -> (int, int):
        """
        Returns the year and quarter of the previous quarter based on the given query date.

        Args:
            query_date: The date for which to find the previous quarter.

        Returns:
            A tuple containing the year and quarter (1-4) of the previous quarter.

        Note:
            Quarters are defined as:
                Q1: January 1 - March 31
                Q2: April 1 - June 30
                Q3: July 1 - September 30
                Q4: October 1 - December 31
        """
        if query_date.month < 4:
            req_date = datetime.date(query_date.year - 1, 12, 31)
        elif query_date.month < 7:
            req_date = datetime.date(query_date.year, 3, 31)
        elif query_date.month < 10:
            req_date = datetime.date(query_date.year, 6, 30)
        else:
            req_date = datetime.date(query_date.year, 9, 30)
        return req_date.year, req_date.month // 3

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

    def _get_source_data_query(
        self, query_date: datetime.date, source_table_name: str
    ) -> Optional[str]:
        """
        Returns a SQL query string to retrieve data from the source table based on the given query date and grain.

        Args:
            query_date: The date for which to retrieve data.
            source_table_name: The name of the source table.

        Returns:
            A SQL query string to retrieve data from the source table, or None if an error occurs.

        Raises:
            Exception: If the grain is not either 'M' for monthly or 'Q' for quarterly.

        Note:
            The grain determines the time period for which data is retrieved. If the grain is 'M', data is retrieved for the previous month. If the grain is 'Q', data is retrieved for the previous quarter.
        """
        date_col = self.config["column_ref"]["date_col"]
        start_date = self.config["column_ref"]["start_date"]
        if self.grain == "M":
            year, month = self._previous_month(query_date)
            data_query = f"select * from {source_table_name} where year({date_col}) = {year} and month({date_col}) = {month}"
            if self.initial_run:
                data_query = f"select * from {source_table_name} where {date_col}  >= TO_DATE('{start_date}', 'DD-MM-YYYY') and ((year({date_col}) < {year}) or (year({date_col}) = {year} and month({date_col}) <= {month}))"
            return data_query
        elif self.grain == "Q":
            year, quarter = self._previous_quarter(query_date)
            data_query = f"select * from {source_table_name} where year({date_col}) = {year} and quarter({date_col}) = {quarter}"
            if self.initial_run:
                data_query = f"select * from {source_table_name} where {date_col} >= TO_DATE('{start_date}', 'DD-MM-YYYY') and ((year({date_col}) < {year}) or (year({date_col}) = {year} and quarter({date_col}) <= {quarter}))"
            return data_query

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
            data_query = self._get_source_data_query(query_date, source_table_name)

            conn = self._get_active_sfconn()
            df = pd.read_sql(data_query, conn)
            df["year_month"] = df[self.config["column_ref"]["date_col"]].apply(
                lambda x: f"{x.year}-M{x.month}"
            )
            df["year_quarter"] = df[self.config["column_ref"]["date_col"]].apply(
                lambda x: f"{x.year}-Q{x.quarter}"
            )
            return df
        except Exception as e:
            raise Exception(
                f"Error reading data from {source_table_name} table. The following exception raised {e}"
            )

    def _generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a summary of the given DataFrame using the Orchestrator class.

        Args:
            df: The DataFrame to summarize.

        Returns:
            A new DataFrame containing the summary of the input DataFrame.

        Note:
            The summary is generated using the Orchestrator class, which takes into account the grain specified in the class instance.
        """

        orchestrator = Orchestrator(map_prompt, reduce_prompt, "text", self.grain, self.llm_config, self.general_configs)
        result = orchestrator.summary_of_df(df)
        return result

    def _get_vokol_summary_df(
        self, insights_df: pd.DataFrame, actions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combines the insights and actions DataFrames into a single summary DataFrame.

        Args:
            insights_df: The DataFrame containing insights data.
            actions_df: The DataFrame containing actions data.

        Returns:
            A new DataFrame containing the combined summary data.

        Note:
            The resulting DataFrame is created by concatenating the insights and actions DataFrames along the rows (axis=0). A unique SUMMARY_ID is generated for each row using a UUID. The NAME column is processed to remove any unnecessary characters and split into a list of values.

            Any rows with missing values in the SUMMARY_TEXT column are dropped from the resulting DataFrame.
        """
        vokol_summary_df = pd.concat(
            [insights_df, actions_df], ignore_index=True, sort=False, axis=0
        )

        vokol_summary_df["SUMMARY_ID"] = vokol_summary_df.apply(
            lambda _: str(uuid.uuid4()), axis=1
        )

        vokol_summary_df["VOKOL_NAME"] = vokol_summary_df["VOKOL_NAME"].apply(
            lambda x: list(
                map(
                    str.strip,
                    str(x).strip("][").replace('"', "").replace("'", "").split(","),
                )
            )
        )
        vokol_summary_df["SUMMARY_TEXT"] = vokol_summary_df["SUMMARY_TEXT"].apply(lambda x: x if x != '' else np.nan)
        vokol_summary_df = vokol_summary_df.dropna(subset=["SUMMARY_TEXT"])
        if self.grain == "M":
            vokol_summary_df["SUMMARY_TIME_FRAME"] = 'Monthly'
        elif self.grain == "Q":
            vokol_summary_df["SUMMARY_TIME_FRAME"] = 'Quarterly'
        return vokol_summary_df

    def _restructure_df_based_on_schema(
        self, gen_summary_df: pd.DataFrame, sentiment_df: pd.DataFrame
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Restructures the given DataFrames based on the schema and grain.

        Args:
            gen_summary_df: The DataFrame containing the generated summary data.
            sentiment_df: The DataFrame containing the sentiment data.

        Returns:
            A tuple of two DataFrames: the restructured summary DataFrame and the summary-sentiment cross-reference DataFrame.

        Note:
            The function first checks the grain and sets the corresponding column name. It then creates two separate DataFrames for insights and actions, and adds a SUMMARY_TYPE column to each. The DataFrames are then merged and restructured to match the schema.

            The function also creates a cross-reference DataFrame that maps the summary IDs to the prediction IDs.

            If the grain is not either 'M' for monthly or 'Q' for quarterly, an error is logged and an exception is raised.
        """
        if self.grain == "M":
            grain_col = "year_month"
        elif self.grain == "Q":
            grain_col = "year_quarter"

        insight_df = gen_summary_df[
            ["Insights", "Insights_ID", "PREDICTED_THEME", "PREDICTED_TOPIC", "SOURCE_DATA_PROVIDER", grain_col]
        ].copy()
        action_df = gen_summary_df[
            ["Actions", "Actions_ID", "PREDICTED_THEME", "PREDICTED_TOPIC", "SOURCE_DATA_PROVIDER", grain_col]
        ].copy()

        insight_df["SUMMARY_TYPE"] = "INSIGHTS"
        action_df["SUMMARY_TYPE"] = "ACTIONS"

        action_df.rename(
            columns={"Actions": "SUMMARY_TEXT", "Actions_ID": "VOKOL_NAME"}, inplace=True
        )
        insight_df.rename(
            columns={"Insights": "SUMMARY_TEXT", "Insights_ID": "VOKOL_NAME"}, inplace=True
        )

        vokol_summary_df = self._get_vokol_summary_df(insight_df, action_df)

        temp_df = vokol_summary_df[
            ["SUMMARY_ID", "VOKOL_NAME", "PREDICTED_THEME", "PREDICTED_TOPIC", "SOURCE_DATA_PROVIDER", grain_col]
        ]
        temp_df = temp_df.explode("VOKOL_NAME")

        left_join_df = temp_df.merge(
            sentiment_df,
            on=["VOKOL_NAME", "PREDICTED_THEME", "PREDICTED_TOPIC", "SOURCE_DATA_PROVIDER", grain_col],
            how="left",
            indicator=True,
        )

        sa_summary_xref_df = left_join_df[["SUMMARY_ID", "PREDICTION_ID"]].dropna()
        vokol_summary_df = vokol_summary_df[
            [
                "SUMMARY_ID",
                "SUMMARY_TYPE",
                "SUMMARY_TEXT",
                "PREDICTED_THEME",
                "PREDICTED_TOPIC",
                "SUMMARY_TIME_FRAME",
                "SOURCE_DATA_PROVIDER"
            ]
        ]

        return vokol_summary_df, sa_summary_xref_df

    def predict(self, day: int, month: int, year: int) -> (pd.DataFrame, pd.DataFrame):
        """
        Generates predictions for the given day.

        Args:
            day: The day of the month (1-31).
            month: The month of the year (1-12).
            year: The year.

        Returns:
            A tuple of two DataFrames: the summary DataFrame and the summary-sentiment cross-reference DataFrame.

        Note:
            The function first retrieves the source data for the given date using the `_get_source_data` method. It then generates a summary of the data using the `_generate_summary` method. The summary is then restructured to match the schema using the `_restructure_df_based_on_schema` method.

            If no data is found for the given date and grain, an AssertionError is raised.

            The function returns two DataFrames: the summary DataFrame and the summary-sentiment cross-reference DataFrame.
        """
        query_date = datetime.date(year, month, day)
        sentiment_df = self._get_source_data(query_date)
        sentiment_df.sort_values(by=[self.config["column_ref"]["date_col"]])

        if len(sentiment_df) == 0:
            dbutils.notebook.exit("No data found for the given date and grain.")

        gen_summary_df = self._generate_summary(sentiment_df)
        vokol_summary_df, sa_summary_xref_df = self._restructure_df_based_on_schema(
            gen_summary_df, sentiment_df
        )
        return vokol_summary_df, sa_summary_xref_df

# COMMAND ----------

widget_values = dbutils.widgets.getAll()

# COMMAND ----------

# SummarizationInference class to call the Summarization Inference model
summarization_inference = SummarizationInference(summarization_inference_config, widget_values.get('grain','M'), llm_config, general_configs)

# COMMAND ----------

# Fetch todays date
run_date = datetime.datetime.today()

# COMMAND ----------

# Generating summaries and xref table for the given date and grain
vokol_summary_df, sa_summary_xref_df = summarization_inference.predict(run_date.day,run_date.month,run_date.year)

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC #05. Post-processing and Ingesting to Snowflake
# MAGIC Below code has the following functionalities:
# MAGIC - Add metadata columns
# MAGIC - Write data to the snowflake

# COMMAND ----------

def add_system_columns(df, widget_values):
  """
    Add system columns to a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to which system columns will be added.
        widget_values (dict): A dictionary containing widget values.

    Returns:
        pd.DataFrame: The input DataFrame with the added system columns.

    Notes:
        The following system columns are added:
            - DATABRICKS_RUNID
            - BATCH_ID
            - JOB_ID
            - INSERT_DATE_TIME (current timestamp in UTC)
            - UPDATE_DATE_TIME (initially set to None)
  """
  df['DATABRICKS_RUNID'] = widget_values.get('databricks_run_id', '')
  df['BATCH_ID'] = widget_values.get('batch_id', '')
  df['JOB_ID'] = widget_values.get('job_id', '')
  df['INSERT_DATE_TIME'] = pd.Timestamp('now', tz=pytz.utc)
  df['UPDATE_DATE_TIME'] = None
  return df

# COMMAND ----------

# Add metadata columns to the summary dataframe
vokol_summary_df['RECORD_ID'] = vokol_summary_df['SUMMARY_ID'].astype(str)
vokol_summary_df = add_system_columns(vokol_summary_df, widget_values)

# COMMAND ----------

# Add metadata columns to the sa_summary_xref dataframe
sa_summary_xref_df["RECORD_ID"] = sa_summary_xref_df.apply(lambda _: uuid.uuid4(), axis=1)
sa_summary_xref_df["RECORD_ID"] = sa_summary_xref_df["RECORD_ID"].astype(str)
sa_summary_xref_df = add_system_columns(sa_summary_xref_df, widget_values)

# COMMAND ----------

# Write the summary data to Snowflake
options = {
  "sfUrl": f"https://{account}.snowflakecomputing.com",
  "sfUser": sfUser,
  "sfPassword": sfPwd,
  "sfDatabase": summarization_inference_config['data_destination']['db_name'],
  "sfSchema": summarization_inference_config['data_destination']['schema_name'],
  "sfWarehouse": warehouse,
  "sfrole":role,
  "dbtable":summarization_inference_config['data_destination']['summary_table_name'],
  "column_mapping": 'name'
}

vokol_summary_spark_df = session.createDataFrame(vokol_summary_df)
vokol_summary_spark_df = vokol_summary_spark_df.withColumn("UPDATE_DATE_TIME", vokol_summary_spark_df["UPDATE_DATE_TIME"].cast(TimestampType()))

vokol_summary_spark_df.write.format("snowflake") \
.options(**options)\
.mode('append')\
.save()

# COMMAND ----------

# Write the sa_summary_xref data to Snowflake
options = {
  "sfUrl": f"https://{account}.snowflakecomputing.com",
  "sfUser": sfUser,
  "sfPassword": sfPwd,
  "sfDatabase": summarization_inference_config['data_destination']['db_name'],
  "sfSchema": summarization_inference_config['data_destination']['schema_name'],
  "sfWarehouse": warehouse,
  "sfrole":role,
  "dbtable":summarization_inference_config['data_destination']['xref_table_name'],
  "column_mapping": 'name'
}

sa_summary_xref_spark_df = session.createDataFrame(sa_summary_xref_df)
sa_summary_xref_spark_df = sa_summary_xref_spark_df.withColumn("UPDATE_DATE_TIME", sa_summary_xref_spark_df["UPDATE_DATE_TIME"].cast(TimestampType()))

sa_summary_xref_spark_df.write.format("snowflake") \
.options(**options)\
.mode('append')\
.save()

# COMMAND ----------
