# Databricks notebook source
from ragas import evaluate
from ragas.metrics import summarization_score,context_recall
import datasets
import pandas as pd

import nest_asyncio
nest_asyncio.apply()

# COMMAND ----------

class Evaluator:
  def __init__(self,config:dict):
    """
    Intializes the evaluator class.

    Parameters
    ----------
      config : dict
        llm config
    """

    self.llm = DatabricksLLMCall(
    endpoint_url_base = config['endpoint_url_base'],
    model_name = config['model_settings']['name'],
    max_tokens = config['model_settings']['max_tokens'],
    max_response_tokens= config['model_settings']['max_response_tokens'],
    access_token= dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    )

  def summarization_score(self,out_dict) -> float:
    """
    Returns summarization score calculated using RAGAS framework.

    Parameters
    ----------
    out_dict : dict
      output dictionary from the model with input and summary.

    Returns:
      ragas_score : float
        summarization score,returns -2 incase of any error
    """
    data = {}
    data['contexts']=[out_dict['input']]
    data['summary']=[out_dict['summary']]
    try:
        # Initialize the RAGAS object
        ragas_score = evaluate(datasets.Dataset.from_dict(data),
            metrics=[
                summarization_score
            ],llm=self.llm,is_async=False
        )['summary_score']
    except:
        ragas_score=-2
    return ragas_score
