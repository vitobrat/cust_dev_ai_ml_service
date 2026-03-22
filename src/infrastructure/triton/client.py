"""Triton Inference Server async gRPC client for embedding generation.

Pipeline: raw text → tokenization → Triton inference → mean pooling → L2 norm → embedding vector.

The ONNX model (multilingual-e5-small) expects three INT64 tensors:
    - input_ids:      token indices from the vocabulary
    - attention_mask:  1 for real tokens, 0 for padding
    - token_type_ids:  segment IDs (all zeros for single-sentence input)

The model returns last_hidden_state of shape [batch, seq_len, 384] —
one vector per token. We apply mean pooling over real (non-padding)
tokens and L2-normalize the result to get a single 384-dim embedding
per input text.
"""

import numpy as np
from numpy import typing as npt
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from tritonclient.grpc import aio as triton_grpc

from src.configs.config import TritonConfigs
from src.configs.consts import EPSILON
from src.configs.log.logger import get_logger
from src.infrastructure.triton.schema import EmbeddingsModelInput


class TritonClient:
    """Async gRPC client for text embedding via Triton Inference Server.

    Handles the full pipeline: tokenization (client-side) → inference
    (server-side ONNX model) → post-processing (client-side pooling).

    Attributes:
        _client:     Underlying async gRPC Triton client.
        _model_name: Name of the deployed ONNX model on Triton.
        _tokenizer:  HuggingFace tokenizer matching the deployed model.
        _max_length: Maximum token sequence length (model limit is 512).
    """

    def __init__(self, configs: TritonConfigs) -> None:
        """Initialize the async Triton gRPC client.

        Args:
            configs: Triton server connection and model configuration.
        """
        self._logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        url = f"{configs.host}:{configs.grpc_port}"

        self._client = triton_grpc.InferenceServerClient(url=url)
        self._model_name = configs.model_name
        self._model_output_name = configs.output_name
        self._max_length = configs.max_length
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            configs.tokenizer_name,
        )

    async def close(self) -> None:
        """Close the underlying gRPC channel and release all resources."""
        await self._client.close()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts via Triton inference.

        Args:
            texts: Non-empty list of texts to embed.

        Returns:
            Embedding vectors in the same order as the input texts.
            Each vector has dimensionality defined by the deployed model.

        Raises:
            triton_grpc.InferenceServerException: If the model is unavailable
                or inference fails on the server side.
        """
        tokens = self._tokenize(texts)
        self._logger.debug("Tokenization processed")

        inputs = self._build_inputs(tokens)
        self._logger.debug("Inputs built")

        outputs = [triton_grpc.InferRequestedOutput(self._model_output_name)]
        triton_result = await self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
        )

        hidden_state = triton_result.as_numpy(self._model_output_name)
        self._logger.debug("Infer processed, output shape: %s", hidden_state.shape)

        attention_mask = tokens.attention_mask
        embeddings = self._mean_pool_and_normalize(hidden_state, attention_mask)
        self._logger.debug("Mean pooling and normalization processed, shape: %s", embeddings.shape)

        return embeddings.tolist()

    def _tokenize(self, texts: list[str]) -> EmbeddingsModelInput:
        """Tokenize a batch of texts into padded integer arrays.

        The tokenizer does the following for each text:
          1. Splits text into subword tokens using the learned vocabulary.
          2. Maps each subword to its integer ID (input_ids).
          3. Adds special tokens: [CLS] at start, [SEP] at end.
          4. Pads shorter sequences with zeros to match the longest one.
          5. Creates attention_mask: 1 for real tokens, 0 for padding.
          6. Creates token_type_ids: all zeros (single sentence).

        Args:
            texts: Raw strings (should already have "query: " / "passage: " prefix).

        Returns:
            Dict with three keys, each mapping to an int64 numpy array
            of shape [batch_size, padded_sequence_length]:
                - "input_ids"
                - "attention_mask"
                - "token_type_ids"
        """
        encoded = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="np",
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        token_type_ids = encoded.get(
            "token_type_ids",
            np.zeros_like(input_ids),
        )

        return EmbeddingsModelInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    @staticmethod
    def _build_inputs(
        tokenized_tensores: EmbeddingsModelInput,
    ) -> list[triton_grpc.InferInput]:
        """Pack tokenized arrays into Triton InferInput objects.

        Each of the three arrays (input_ids, attention_mask, token_type_ids)
        becomes a separate InferInput with datatype INT64 and shape
        [batch_size, sequence_length].

        Args:
            tokenized: Output of _tokenize() — three int64 numpy arrays.

        Returns:
            List of three InferInput objects ready for triton_grpc.infer().
        """
        inputs: list[triton_grpc.InferInput] = []

        for tensor_name in EmbeddingsModelInput.model_fields:
            tensor_data: npt.NDArray[np.int64] = getattr(tokenized_tensores, tensor_name)

            infer_input = triton_grpc.InferInput(
                tensor_name,
                shape=list(tensor_data.shape),
                datatype="INT64",
            )
            infer_input.set_data_from_numpy(tensor_data)

            inputs.append(infer_input)

        return inputs

    @staticmethod
    def _mean_pool_and_normalize(
        hidden_state: npt.NDArray[np.float32],
        attention_mask: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float32]:
        """Apply mean pooling over tokens and L2-normalize the result.

        The ONNX model returns one 384-dim vector per token.  We need
        one vector per text.  Mean pooling averages the token vectors,
        but only over real tokens (not padding).

        Step by step:
          1. attention_mask has shape [batch, seq_len].
             Expand it to [batch, seq_len, 1] so we can multiply
             element-wise with hidden_state [batch, seq_len, 384].

          2. Multiply: padding token vectors become all-zeros.

          3. Sum along the seq_len axis → [batch, 384].

          4. Divide by the count of real tokens (sum of mask)
             to get the mean → [batch, 384].

          5. L2-normalize: divide each vector by its Euclidean length
             so that cosine similarity = dot product.

        Args:
            hidden_state:  Model output, shape [batch, seq_len, 384].
            attention_mask: Token mask, shape [batch, seq_len].

        Returns:
            Normalized embeddings, shape [batch, 384].
        """
        # [batch, seq_len] → [batch, seq_len, 1]
        mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(np.float32)

        # [batch, seq_len, 384] * [batch, seq_len, 1] → sum → [batch, 384]
        sum_embeddings = np.sum(hidden_state * mask_expanded, axis=1)

        # Count real tokens per sample: [batch, seq_len, 1] → sum → [batch, 1]
        sum_mask = np.sum(mask_expanded, axis=1)
        # clamp to minimum 1e-9 to avoid division by zero
        sum_mask = np.clip(sum_mask, a_min=EPSILON, a_max=None)

        # Mean pooling
        embeddings = sum_embeddings / sum_mask  # [batch, 384]

        # L2 normalization
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=EPSILON, a_max=None)
        embeddings /= norms

        return embeddings
