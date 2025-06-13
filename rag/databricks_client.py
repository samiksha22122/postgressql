# Databricks notebook source
import logging
import yaml
from openai import OpenAI
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

# COMMAND ----------

class DatabricksLLMCall(LLM):
    """
    Creates a connection to the LLM hosted as Databricks serving endpoints

    Attributes:
      endpoint_url_base : str
                        The base url of the serving endpoints. Will be in the form of https://xxxxx.cloud.databricks.com/serving-endpoints"
      model_name : str
                  The name of the deployed model
      max_tokens : int
                  The maximum context length for the LLM
      max_response_tokens: int
                  The maximum response tokens for the response

    e.g. :
        llm = DatabricksLLMCall(
          'endpoint_url_base' = 'https://<your-workspace>.cloud.databricks.com/serving-endpoints'
          'model_name' = 'llama-3',
          'max_tokens' = 8000,
          'max_response_tokens' = 1000
        )

    """

    endpoint_url_base = ""
    model_name = ""
    max_tokens = 2048
    max_response_tokens = 256
    access_token = ""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> str:
        """
        Invokes the LLM call based on the given prompt
        Parameters:
          prompt : str
              The input to be passed to the LLM

        Returns:
          str
          Output from the LLM

        e.g. :
         llm.invoke('hi') , hi being the prompt

        """
        try:
            client = OpenAI(api_key=self.access_token, base_url=self.endpoint_url_base)
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_response_tokens,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(
                f"Error at time of making LLM Call\n. Following exception raised : {e.message}"
            )
            return None

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"model_name": self.model_name, "max_context_length": self.max_tokens}

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return self.model_name
