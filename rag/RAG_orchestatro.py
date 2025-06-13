# Databricks notebook source
# MAGIC %md
# MAGIC #01.Importing Libraries

# COMMAND ----------

import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC #02.Importing Notebooks
# MAGIC - UTILS:To load utility functions
# MAGIC - LLM_MODEL_CALL: To load DatabricksLLMCall class
# MAGIC - GUARD_RAILS : To load Masker and Validate class
# MAGIC - EVALUATION : To load Evaluator class
# MAGIC - SUMMARIZATION : To load Summarizer class

# COMMAND ----------

# MAGIC %run "./utils/UTILS"

# COMMAND ----------

# MAGIC %run "./utils/LLM_MODEL_CALL"

# COMMAND ----------

# MAGIC %run "./utils/GUARD_RAILS"

# COMMAND ----------

# MAGIC %run "./utils/EVALUATION"

# COMMAND ----------

# MAGIC %run "./SUMMARIZATION"

# COMMAND ----------

# MAGIC %md
# MAGIC #03.Jsonproccessor Class
# MAGIC
# MAGIC The `Jsonproccessor` class is responsible for processing sting into JSON data, including validation and transformation. It interacts with a language model to ensure the JSON data meets specific requirements.
# MAGIC
# MAGIC Methods:
# MAGIC
# MAGIC - \_\_init__: Initializes the class with configuration parameters.
# MAGIC - string_to_dict: Converts a summary string into a dictionary.
# MAGIC - get_validate_dict_summary: Validates the tone and toxicity of actions and insights.
# MAGIC - get_validated_ids: Validates the IDs of actions and insights.

# COMMAND ----------

class Jsonproccessor:
    def __init__(self, config: dict, general_configs: dict):
        """
        Initializes the class with the required parameters.
        Parameters
        ----------
        config : dict
            llm config
        general_configs: dict
            general configs for the pipeline

        """

        self.validate = Validate(general_configs)

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

        self.logger = logging.getLogger("jsonproccessor")
        self.logger.setLevel(logging.DEBUG)

        self.expected_keys = ["Actions", "Insights", "Actions_ID", "Insights_ID"]

    def string_to_dict(self, summary: str) -> Dict[str, any]:
        """
        Returns summary in the format of dict of the passed summary in the string format.
        Parameters
            ----------
              summary:str:
                List of texts to be summarized

        Retunrs
            ----------
            Dict:
              Summary in dict with excpected key ['Actions','Insights','Actions_Id','Insights_Id'], the key will be based on the prompt passed.

        """
        pattern = re.compile(r"```(.*?)```", re.DOTALL)
        attempts = 0

        output_dict = {}
        continue_execution = True

        while continue_execution:
            if attempts <= self.max_retries:
                attempts += 1
                # Find all matches for raw_summary
                try:
                    matches = pattern.findall(summary)
                    json_str = matches[0]
                    self.logger.info(f'Json str {json_str}')
                    output_dict = ast.literal_eval(json_str)

                    if validate_dict_keys(output_dict, self.expected_keys):
                        continue_execution = False
                    else:
                        self.logger.info(
                            "Expected keys not found in the json,running key fix prompt"
                        )
                        key_fix_prompt = """The json must contian these keys: {expected_keys}, please fix the output and send again.Dont change any contents inside the json,these are the expected keys: {expected_keys}.For these two keys({value_keys}) the values must be in string format and for these two keys({id_keys}) the values must be in list format.

                        Output:
                        {summary}
                        """
                        fix_prompt = key_fix_prompt.format(
                            expected_keys=self.expected_keys, summary=summary,value_keys=self.expected_keys[0:2],id_keys=self.expected_keys[2:4]
                            )

                        self.logger.info(
                            f"Input Summary on {attempts} attempt: {summary}"
                        )
                        summary = self.llm.invoke(fix_prompt)

                        self.logger.info(
                            f"Output Summary on {attempts} attempt: {summary}"
                        )

                except IndexError:
                    match_fix_prompt = """The below output must have a json between ``` and ```,but not able to find this pattern, please fix the format and send the json in between ``` and ```. Dont change any contents inside the json,these are the expected keys: {expected_keys}.
                    For these two keys({value_keys}) the values must be in string format and for these two keys({id_keys}) the values must be in list format.

                    Output:
                    {summary}
                    """
                    fix_prompt = match_fix_prompt.format(summary=summary,expected_keys=self.expected_keys,value_keys=self.expected_keys[0:2],id_keys=self.expected_keys[2:4])
                    self.logger.info(
                            f"Input Summary on {attempts} attempt: {summary}"
                        )
                    summary = self.llm.invoke(fix_prompt)
                    self.logger.info(
                        f"Output Summary on {attempts} attempt: {summary}"
                    )
                except SyntaxError:
                    json_stynax_fix_prompt = """fix the format of this json and send it in between ``` and ```.Dont change any contents inside the json,these are the expected keys: {expected_keys}.
                    For these two keys({value_keys}) the values must be in string format and for these two keys({id_keys}) the values must be in list format.

                    Output:
                    {summary}"""

                    fix_prompt = json_stynax_fix_prompt.format(summary=summary,expected_keys=self.expected_keys,value_keys=self.expected_keys[0:2],id_keys=self.expected_keys[2:4])

                    self.logger.info(f"Input Summary on {attempts} attempt: {summary}")
                    summary = self.llm.invoke(fix_prompt)
                    self.logger.info(
                        f"Output Summary on {attempts} attempt: {summary}"
                    )
                except ValueError:
                    summary = {'Actions': '', 'Insights': '', 'Actions_ID': [], 'Insights_ID': []}
                    self.logger.info(
                    f"Json structure failed.Forcing an empty string"
                    )

            else:
                self.logger.info(
                    f"Failed to convert output into json format after {self.max_retries} retries"
                )

                output_dict = {}
                continue_execution = False

        return output_dict

    def get_validate_dict_summary(self, out_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates the tone and toxicity of actions and insights and returns the validated output dict.
        Parameters:
        ----------
         out_dict:Dict[str]:
           Output dict with actions and insights
        Returns:
        -------
          out_dict:Dict[str]:
            Output dictionary with validated actions and insights
        """
        attempts = 0

        for i in ["Actions", "Insights"]:
            value = out_dict[i]
            if value != "":
                while not (self.validate.toxicity(value)) or (
                    not self.validate.tone(value)
                ):
                    if attempts <= self.max_retries:
                        if not self.validate.toxicity(value):
                            toxicity_fix_prompt = """The below output contains toxic word and harmful language, please remove those and send the output again,only send the output and nothing else
                            {summary}
                            """
                            fix_prompt = toxicity_fix_prompt.format(summary=value)
                            value = self.llm.invoke(fix_prompt)
                            attempts += 1
                            continue

                        if not self.validate.tone(value):
                            tone_fix_prompt = """The below output is not in a proffesional tone, please change it to a proffesional tone and send the output again,only send the output and nothing else
                            {summary}
                            """
                            fix_prompt = tone_fix_prompt.format(summary=value)
                            value = self.llm.invoke(fix_prompt)
                            attempts += 1
                            continue
                    else:
                        self.logger.info(
                            f"Failed to validate output after {self.max_retries} retries."
                        )
                        value = ""
                    out_dict[i] = value
                    break

        return out_dict

    def get_validated_ids(
        self, out_dict: Dict[str, Any], id_lst: List[str]
    ) -> Dict[str, Any]:
        """
        Validates the ids of actions and insights and returns the validated output dict.

        Parameters:
        ----------
         out_dict:Dict[str]:
           Output dict with Actions_ID and Insights_ID

        Returns:
        -------
          out_dict:Dict[str]:
            Output dictionary with validated Actions_ID and Insights_ID
        """

        if not is_subset(out_dict["Actions_ID"], id_lst) or not is_subset(
            out_dict["Insights_ID"], id_lst
        ):
            self.logger.warning("The output ids are not a subset of the input ids")
            out_dict["Actions_ID"] = id_lst
            out_dict["Insights_ID"] = id_lst

        return out_dict

# COMMAND ----------

# MAGIC %md
# MAGIC #03.Orchestrator Class
# MAGIC
# MAGIC The `Orchestrator` class manages the end-to-end process of summarizing and validating data. It integrates various components like summarization, JSON processing, masking, and evaluation.
# MAGIC
# MAGIC Methods:
# MAGIC
# MAGIC - \_\_init\_\_: Initializes the class with required parameters.
# MAGIC - process: Processes input data, masks opinions, runs summarization, and outputs a dictionary.
# MAGIC - summary_of_df: Generates output for the entire dataframe after grouping by specified columns and calculates summarization scores.

# COMMAND ----------

class Orchestrator:
    def __init__(self, map_prompt: str, reduce_prompt: str, text_var: str, grain: str,config:dict,general_configs:dict):
        """
        Intializing the class
        Parameters
        ----------
          map_prompt : str
            prompt for the map stage
          reduce_prompt : str
            prompt for the reduce stage
          text_var : str
            Name of the variable where documnet must be inserted.
          grain : str
            M if montlhy and Q if quarterly
          config : dict
            llm config
          general_configs: dict
            general configs for the pipeline
        """
        self.summarizer = Summarizer(map_prompt, reduce_prompt, text_var,config,general_configs)
        self.jsonproccessor = Jsonproccessor(config,general_configs)
        self.masker = Masker(general_configs)
        self.evaluator = Evaluator(config)
        self.evaluate_summary=general_configs['evaluate_summary']

        if grain == "M":
            self.group_columns = ["year_month", "PREDICTED_THEME", "PREDICTED_TOPIC", "SOURCE_DATA_PROVIDER"]
        elif grain == "Q":
            self.group_columns = ["year_quarter", "PREDICTED_THEME", "PREDICTED_TOPIC", "SOURCE_DATA_PROVIDER"]
        else:
            logging.error(f"grain {grain} not supported")

        self.vokol_column = "VOICE_OF_CUSTOMER"

        self.required_columns = self.group_columns + [self.vokol_column]

        self.col_name = {
            "id": "VOKOL_NAME",
            "drug": "COMPOUND_PRODUCT",
            "topic": "PREDICTED_TOPIC",
            "opinion": "VOICE_OF_CUSTOMER",
        }

        self.structrer = """ID:{id}
    Topic:{topic}
    Opnion:{opinion}
    """

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Creates input data, maskes the opinion and runs summarization function and outputs dictionary.
        Parameters :
          ----------
          data : pd.DataFrame
          input dataframe
        Returns :
          -------
          out_dict : dict
        """
        col_name = self.col_name
        ids = data[col_name["id"]].to_list()

        data["masked_opinion"] = data[col_name["opinion"]].apply(
            lambda x: self.masker.mask(x)
        )

        data["combined_input"] = data.apply(
            lambda x: self.structrer.format(
                id=x[col_name["id"]],
                topic=x[col_name["topic"]],
                opinion=x["masked_opinion"],
            ),
            axis=1,
        )

        raw_summary = self.summarizer.get_summary(data["combined_input"].to_list())

        out_dict = self.jsonproccessor.string_to_dict(raw_summary)

        if isinstance(out_dict.get('Actions'), list):
          out_dict['Actions'] = ''.join(out_dict['Actions'])

        if isinstance(out_dict.get('Insights'), list):
            out_dict['Insights'] = ''.join(out_dict['Insights'])

        if not out_dict:
            out_dict["raw_summary"] = raw_summary
            return out_dict

        out_dict = self.jsonproccessor.get_validated_ids(out_dict, ids)
        out_dict = self.jsonproccessor.get_validate_dict_summary(out_dict)
        out_dict["raw_summary"] = raw_summary
        return out_dict

    def summary_of_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates output of whole dataframe, after grouping by given columns,also generate summarization score.
        Parameters :
          ----------
          df : pd.DataFrame
            input dataframe
        Returns :
          --------
          final_df : pd.DataFrame"""
        validate_columns(df, self.required_columns)

        gropued_df = df.groupby(self.group_columns)
        final_dict = {}
        for group, sub_df in gropued_df:
            summary_dict = self.process(sub_df)

            if len(summary_dict.keys()) > 1:
                summary_dict["input"] = list(sub_df[self.col_name["opinion"]])
                summary_dict["summary"] = (
                    summary_dict["Actions"] + " " + summary_dict["Insights"]
                )
                if self.evaluate_summary:
                  summary_dict["score"] = self.evaluator.summarization_score(summary_dict)
            else:
                summary_dict["input"] = list(sub_df[self.col_name["opinion"]])

            summary_dict["count"] = len(sub_df)

            final_dict[group] = summary_dict

        final_df = pd.DataFrame(final_dict)

        final_df = final_df.transpose().reset_index()
        n = len(self.group_columns)
        final_df.columns = self.group_columns + final_df.columns[n:].tolist()

        return final_df
