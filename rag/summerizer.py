# Databricks notebook source
# MAGIC %md
# MAGIC #01.Importing Libraries

# COMMAND ----------

import langchain
from langchain.chains import (
    StuffDocumentsChain,
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    LLMChain,
)
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.chains.mapreduce import MapReduceChain
from langchain.schema import Document
import re
import ast
import logging

# COMMAND ----------

# MAGIC %md
# MAGIC #02.Summarizer class
# MAGIC The Summarizer class provides functionality to generate summaries from a list of texts using a map-reduce approach with LLMs.
# MAGIC
# MAGIC - \_\_init__: Initializes the Summarizer with prompts, document variable, and configurations.
# MAGIC - get_summary: Returns a summary of the provided list of strings, with an option for detailed output.

# COMMAND ----------

class Summarizer:
    def __init__(self, map_prompt: str, reduce_prompt: str, doc_var: str,config:dict,general_configs:dict):
        """
        Initialize the Summarizer class with prompts and document variable.

        Parameters
            ----------
            map_prompt : str
                Prompt for map chain
            reduce_prompt : str
                Prompt for reduce chain
            doc_var : str
                Name of the variable where documnet must be inserted.
            config : dict
                LLM config
            genreal_configs : dict
                general configs

        """
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.doc_var = doc_var

        self.llm = DatabricksLLMCall(
            endpoint_url_base=config["endpoint_url_base"],
            model_name=config["model_settings"]["name"],
            max_tokens=config["model_settings"]["max_tokens"],
            max_response_tokens=config["model_settings"]["max_response_tokens"],
            access_token=dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .apiToken()
            .get(),
        )

        self.max_retries = general_configs["max_retires"]

        self.logger = logging.getLogger("summarizer")
        self.logger.setLevel(logging.WARNING)

    def get_summary(self, lst: list[str], full_output: bool = False) -> str | list:
        """
        Returns summary of the passed lsit of strings using combined map_reduce llm chain

        Parameters
            ----------
              lst:list[str]:
                List of texts to be summarized
              full_output: bool:
                Returns full_ouput with intermediate steps and all if true else returns only generated summary,
                 Defaults to False

        Returns
            ----------
            str:
              Generated summary in string format
        """
        chunks = [Document(page_content=text) for text in lst]

        # Map prompt template
        map_prompt_template = PromptTemplate.from_template(self.map_prompt)

        # Reduce prompt template
        reduce_prompt_template = PromptTemplate.from_template(self.reduce_prompt)

        summary_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt_template,
            combine_prompt=reduce_prompt_template,
            verbose=False,
            return_intermediate_steps=True,
        )

        full_summary = summary_chain.invoke(chunks)
        final_summary = full_summary["output_text"]

        if full_output:
            return full_summary
        else:
            return final_summary
