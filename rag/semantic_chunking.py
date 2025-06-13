

import re
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

from typing import List


class Semantic_Chunking:
    def __init__(self, model_name: str) -> None:
        """
        Initialize the class instance with a pre-trained language model.

        Args:
            model_name (str): The name of the pre-trained language model to use.

        Notes:
            - The `tokenizer` and `model` attributes are initialized with the specified
            pre-trained model using the `AutoTokenizer` and `AutoModel` classes from
            the Hugging Face Transformers library.
            - The `breakpoint_threshold` attribute is set to a default value of 0.3, which
            is used to determine the threshold for identifying breakpoints in the text.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.breakpoint_threshold = 0.3

    def merge_short_sentences(
        self, sentences: List[str], min_words: int = 5
    ) -> List[str]:
        """
        Merge short sentences into longer ones.

        This function takes a list of sentences and merges any sentences that have fewer
        than a specified minimum number of words with the previous or next sentence.

        Args:
            sentences (List[str]): A list of sentences to be merged.
            min_words (int, optional): The minimum number of words required for a sentence
                to be considered long enough not to be merged. Defaults to 5.

        Returns:
            List[str]: A list of merged sentences.

        Notes:
            - The function iterates through the input sentences, merging any short sentences
            with the previous or next sentence.
            - The merged sentences are then returned as a list.
        """
        merged_sentences = []
        current_sentence = ""

        for sentence in sentences:
            words = current_sentence.split()
            if len(words) < min_words:
                if current_sentence:
                    current_sentence += " " + sentence
                else:
                    current_sentence = sentence
            else:
                if current_sentence:
                    merged_sentences.append(current_sentence)
                    current_sentence = ""
                merged_sentences.append(sentence)

        if current_sentence:
            merged_sentences.append(current_sentence)

        return merged_sentences

    def calculate_cosine_distances(
        self, sentences_embeddings: List[np.ndarray]
    ) -> List[float]:
        """
        Calculate the cosine distances between consecutive sentence embeddings.

        This function takes a list of sentence embeddings and calculates the cosine
        distances between each pair of consecutive embeddings.

        Args:
            sentences_embeddings (List[np.ndarray]): A list of sentence embeddings.

        Returns:
            List[float]: A list of cosine distances between consecutive sentence embeddings.

        Notes:
            - The cosine similarity is calculated using the `cosine_similarity` function.
            - The cosine distance is calculated as 1 - cosine similarity.
            - The function returns a list of cosine distances, where each distance corresponds
            to the distance between two consecutive sentence embeddings.
        """
        distances = []
        for i in range(len(sentences_embeddings) - 1):
            embedding_current = sentences_embeddings[i]
            embedding_next = sentences_embeddings[i + 1]

            # Calculate cosine similarity
            similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

            # Convert to cosine distance
            distance = 1 - similarity

            # Append cosine distance to the list
            distances.append(distance)
        return distances

    def merge_similar_sentences(
        self, sentences: List[str], distances: List[float]
    ) -> List[str]:
        """
        Merge similar sentences based on their cosine distances.

        This function takes a list of sentences and their corresponding cosine distances,
        and groups the sentences into chunks based on the distances. The distances are
        used to identify breakpoints, where sentences with distances above a certain
        threshold are considered to be part of a new chunk.

        Args:
            sentences (List[str]): A list of sentences to be merged.
            distances (List[float]): A list of cosine distances corresponding to the sentences.

        Returns:
            List[str]: A list of merged sentence chunks.

        Notes:
            - The `breakpoint_threshold` attribute is used to determine the threshold for
            identifying breakpoints.
            - The function iterates through the breakpoints to slice the sentences into chunks.
            - The last group of sentences, if any remain, is appended to the list of chunks.
        """
        indices_above_thresh = [
            i for i, x in enumerate(distances) if x > self.breakpoint_threshold
        ]  # The indices of those breakpoints on your list
        # Initialize the start index
        start_index = 0

        # Create a list to hold the grouped sentences
        chunks = []

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join(group)
            chunks.append(combined_text)

            # Update the start index for the next group
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join(sentences[start_index:])
            chunks.append(combined_text)

        return chunks

    def create_chunks(self, sentence: str) -> List[str]:
        """
        Split a sentence into semantic chunks.

        This function performs the following steps:
        1. Splits the input sentence into individual sentences using punctuation.
        2. Merges short sentences to form more coherent chunks.
        3. Tokenizes the merged sentences using a transformer model.
        4. Calculates the cosine distances between the sentence embeddings.
        5. Merges similar sentences based on their cosine distances.

        Args:
            sentence (str): The input sentence to be chunked.

        Returns:
            List[str]: A list of semantic chunks.

        Notes:
            - The `merge_short_sentences` method is used to merge short sentences.
            - The `tokenizer` and `model` attributes are used for tokenization and sentence embedding calculation, respectively.
            - The `calculate_cosine_distances` method is used to calculate the cosine distances between sentence embeddings.
            - The `merge_similar_sentences` method is used to merge similar sentences based on their cosine distances.
        """
        single_sentences_list = re.split(r"(?<=[.?!])\s+", sentence)
        # Merge short sentences
        merged_sentences = self.merge_short_sentences(single_sentences_list)

        tokenized_sentences = self.tokenizer(
            merged_sentences, return_tensors="pt", padding=True
        )
        sentence_embedding = (
            self.model(**tokenized_sentences)["pooler_output"].detach().numpy()
        )

        distances = self.calculate_cosine_distances(sentence_embedding)
        semantic_chunks = self.merge_similar_sentences(merged_sentences, distances)
        return semantic_chunks

    def aspect_based_merge_chunks(
        self, chunks: List[str], aspects: List[str], sentiments: List[str]
    ) -> (List[str], List[str], List[str]):
        """
        Merge chunks of text based on their corresponding aspects.

        This function iterates through the input lists of chunks, aspects, and sentiments, and merges adjacent chunks that have empty aspects. The merged chunks, aspects, and sentiments are then returned.

        Args:
            chunks (List[str]): A list of text chunks.
            aspects (List[str]): A list of aspects corresponding to the chunks.
            sentiments (List[str]): A list of sentiments corresponding to the chunks.

        Returns:
            Tuple[List[str], List[str], List[str]]: A tuple containing the merged chunks, aspects, and sentiments.

        Notes:
            - Chunks with empty aspects are merged with adjacent chunks.
            - If the input lists of aspects or sentiments are empty, a single empty list is returned for each.
        """
        merged_chunks = []
        current_chunk = ""

        for chunk, aspect in zip(chunks, aspects):
            if len(aspect) == 0:
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk
            else:
                if current_chunk:
                    current_chunk += " " + chunk
                    merged_chunks.append(current_chunk)
                    current_chunk = ""
                else:
                    current_chunk = chunk
                    merged_chunks.append(current_chunk)
                    current_chunk = ""

        if current_chunk:
            if merged_chunks and len(aspects[-1]) == 0:
                merged_chunks[-1] += " " + current_chunk
            else:
                merged_chunks.append(current_chunk)

        aspects = [i for i in aspects if len(i) > 0]
        sentiments = [i for i in sentiments if len(i) > 0]

        if len(aspects) == 0:
            aspects = [[]]

        if len(sentiments) == 0:
            sentiments = [[]]

        return merged_chunks, aspects, sentiments

    def label_based_merge_chunks(
        self,
        chunks: List[str],
        labels: List[str],
        aspects: List[str],
        sentiments: List[str],
    ) -> (List[str], List[str], List[str], List[str]):
        """
        Merge chunks of text based on their corresponding labels.

        This function iterates through the input lists of chunks, labels, aspects, and sentiments, and merges adjacent chunks that have the same label. The merged chunks, labels, aspects, and sentiments are then returned.

        Args:
            chunks (List[str]): A list of text chunks.
            labels (List[str]): A list of labels corresponding to the chunks.
            aspects (List[str]): A list of aspects corresponding to the chunks.
            sentiments (List[str]): A list of sentiments corresponding to the chunks.

        Returns:
            Tuple[List[str], List[str], List[str], List[str]]: A tuple containing the merged chunks, labels, aspects, and sentiments.

        Notes:
            - Chunks with the label "other" are treated as separators and are not merged with adjacent chunks.
            - If the lengths of the input lists do not match, an error message is printed.
        """
        merged_chunks, merged_labels, merged_aspects, merged_sentiments = [], [], [], []
        current_chunk = ""
        previous_label = "other"
        previous_aspect, previous_sentiment = [], []

        for i in range(len(chunks)):
            chunk = chunks[i]
            label = labels[i]
            aspect = aspects[i]
            sentiment = sentiments[i]

            if label == previous_label or label == "other":
                current_chunk += " " + chunk
                previous_aspect += aspect
                previous_sentiment += sentiment
            else:
                if current_chunk and previous_label != "other":
                    merged_chunks.append(current_chunk)
                    merged_labels.append(previous_label)
                    merged_aspects.append(previous_aspect)
                    merged_sentiments.append(previous_sentiment)
                current_chunk = chunk
                previous_aspect = aspect
                previous_sentiment = sentiment
            if label != "other":
                previous_label = label

        if current_chunk:
            merged_chunks.append(current_chunk)
            merged_labels.append(label)
            merged_aspects.append(previous_aspect)
            merged_sentiments.append(previous_sentiment)

        if (len(merged_chunks) != len(merged_labels)) or (
            len(merged_chunks) != len(merged_aspects)
        ):
            print(len(chunks), len(labels), len(aspects), len(sentiments))
            print((chunks), (labels), (aspects), (sentiments))

        return merged_chunks, merged_labels, merged_aspects, merged_sentiments



