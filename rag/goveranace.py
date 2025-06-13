# Databricks notebook source
from presidio_analyzer import AnalyzerEngine, PatternRecognizer
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_analyzer.nlp_engine import SpacyNlpEngine
import guardrails as gd
import torch
from guardrails.validators import ToxicLanguage
from typing import Union, Callable, Any, Tuple ,Optional, Dict
import mlflow
import logging
import mlflow.spacy
import re

# COMMAND ----------

class LoadedSpacyNlpEngine(SpacyNlpEngine):
    def __init__(self, loaded_spacy_model):
        super().__init__()
        self.nlp = {"en": loaded_spacy_model}

# COMMAND ----------

class Masker:
  """
    A class to identify and mask Personally Identifiable Information (PII).

    Methods:
        mask(text: str) -> str:
            Identifies and masks PII in the given text, returning the masked version.
  """

  def __init__(self,general_configs:dict):
    """
    Intilaizing masker class

    Parameters
    ----------
      general_configs: dict
        general configs for the pipeline

    """
    mlflow.set_registry_uri('databricks-uc')

    model_uri=general_configs['spacy_model_uri']

    nlp = mlflow.spacy.load_model(model_uri)

    loaded_nlp_engine = LoadedSpacyNlpEngine(loaded_spacy_model = nlp)

    # Pass the engine to the analyzer
    self.analyzer = AnalyzerEngine(nlp_engine = loaded_nlp_engine)
    medical_title_recognizer = PatternRecognizer(supported_entity="MEDICAL_TITLE", deny_list=["Dr.","Mr.","Mrs."])

    # Add custom recognizers to the analyzer
    self.analyzer.registry.add_recognizer(medical_title_recognizer)

    self.anonymizer = AnonymizerEngine()

  def mask(self,text: str) -> str:
    """
    Identifies and masks PII in the given text, returning the masked version.

    This method scans the input text for any Personally Identifiable Information (PII) such as phone numbers,
    social security numbers, and other sensitive data. Once identified, it replaces the PII with a masked
    placeholder to protect the information.

    Parameters
    ----------
        text (str): The input text containing potential PII.

    Returns:
    ----------
        str: The text with all identified PII masked.

    """

    analyzer_results = self.analyzer.analyze(text=text, entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "URL","MEDICAL_TITLE"], language='en')

    anonymized_results = self.anonymizer.anonymize(
      text=text,
      analyzer_results=analyzer_results,
      operators={"DEFAULT": OperatorConfig("replace", {"new_value": "<ANONYMIZED>"}),
                          "PERSON": OperatorConfig("replace", {"new_value": "<ANONYMIZED_PERSON_NAME>"}),
                          "MEDICAL_TITLE": OperatorConfig("redact", {"new_value":""})}
    )
    return anonymized_results.text


# COMMAND ----------

class Validate:
  """
    A class to validate the tone and toxicity of text.

    Methods:
        tone(text: str,label_mapping: Optional[Dict[str, str]] = None) -> str:
            Validates the tone of the text and returns a flag indicating 'pass' or 'fail'.

        toxicity(input_text: str,threshold: float = 0.5,validation_method: str = "sentence",on_fail: Union[Callable[..., Any], None] = "fix") -> str :
            Validates and updates the text to reduce toxicity, returning the updated text.
  """
  def __init__(self,general_configs:dict):
        """
        Initialize the Validate class

        Parameters
        ----------
            genereal_configs: dict
                general configs for the pipeline
        """
        mlflow.set_registry_uri('databricks-uc')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model_uri=general_configs['classifier_model_uri']

        self.classifier=mlflow.transformers.load_model(model_uri,return_typr='pipeline')
        self.logger = logging.getLogger('validate')
        self.logger.setLevel(logging.DEBUG)

  def tone(self,
        text: str,
        label_mapping: Optional[Dict[str, str]] = None
        ) -> bool:
        """
        Classifies the given text using zero-shot classification.

        Parameters
        ----------
        text : str
            The input text to classify.
        label_mapping : Dict[str, str], optional
            A dictionary mapping candidate labels to output values.
            If not provided, the default mapping is {"suggestion": "pass", "demand": "Sorry, I can't answer to that"}.
            If the user want to provide any input then the labels and outputs can be given this way
            Ex : {
            "suggestion": "Accept",
            "demand": "Reject",
            "rude": "Reject",
            "grateful": "Accept",
            "excited": "Accept",
            "apologetic": "Accept"
            }

        Returns
        -------
        bool:
            False if the tone check fails,
            otherwise returns True
        """
        candidate_labels = (
            list(label_mapping.keys()) if label_mapping else ["Unprofessional", "Professional"]
            )
        result = self.classifier(text, candidate_labels, multi_label=True)
        label = result["labels"][0]
        output = (
            label_mapping[label]
            if label_mapping
            else {"Professional": "Pass", "Unprofessional": "Fail"}[label]
        )
        label_probs = {
            label: prob for label, prob in zip(result["labels"], result["scores"])
        }
        self.logger.info(f"{output}ed the tone check.")
        if output=="Pass":
            return True
        else:
            return False

  def toxicity(self,
        input_text: str,
        threshold: float = 0.5,
        validation_method: str = "sentence",
        on_fail: Union[Callable[..., Any], None] = "fix"
        ) -> bool:
        """
        Perform toxic language check on input text and .

        Parameters
        ----------
        input_text : str
            Text to pefrome toxic check and modifications
        threshold : float, optional
            Threshold for toxicity. Defaults to 0.5.
        validation_method : str, optional
            Validation method. Defaults to "sentence".
        on_fail : Union[Callable[..., Any], None], optional
            Action to take on validation failure. Defaults to "fix".

        Returns
        -------
        bool:
            False if the toxic check fails,
            otherwise returns True.
        """
        guard = gd.Guard.from_string(
            validators=[
                ToxicLanguage(
                    threshold=threshold,
                    validation_method=validation_method,
                    on_fail=on_fail,
                )
            ],
            description="testmeout",
            )

        raw_llm_output, validated_output, *rest = guard.parse(llm_output=input_text)

        if len(raw_llm_output) != len(validated_output):
            self.logger.info("Failed the toxicity check.")
            return False
        else:
            self.logger.info("Passed the toxicity check.")
            return True


  def match_pattern(self,input_text: str)-> bool:
    """
    Checks the input text for code blocks(using triple backticks) and returns True if found.
    Parameters
    """
    pattern = re.compile(r"```(.*?)```", re.DOTALL)
    matches = pattern.findall(input_text)
    if matches:
        return True
    else:
        return False
