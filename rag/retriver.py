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
                    select {self.config['query_settings']['vokol_col']}, {self.config['query_settings']['vokol_id_col']}, {self.config['query_settings']['date_col']}, {self.config['query_settings']['hcp_speciality_col']}, VECTOR_COSINE_SIMILARITY({self.config['query_settings']['embedding_col']}, {query_vec}::VECTOR(FLOAT, {vector_dim}))
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
                    select {self.config['query_settings']['vokol_col']}, {self.config['query_settings']['vokol_id_col']}, {self.config['query_settings']['date_col']}, {self.config['query_settings']['hcp_speciality_col']}, VECTOR_COSINE_SIMILARITY({self.config['query_settings']['embedding_col']}, {query_vec}::VECTOR(FLOAT, {vector_dim}))
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
                    select {self.config['query_settings']['vokol_col']}, {self.config['query_settings']['vokol_id_col']}, {self.config['query_settings']['date_col']}, {self.config['query_settings']['hcp_speciality_col']}, VECTOR_COSINE_SIMILARITY({self.config['query_settings']['embedding_col']}, {query_vec}::VECTOR(FLOAT, {vector_dim}))
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
        return response--
