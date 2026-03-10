"""Triton Inference Server async gRPC client for embedding generation."""

import numpy as np
from numpy import typing as npt
from tritonclient.grpc import aio as triton_grpc

from src.configs.config import TritonConfigs


class TritonClient:
    """Async gRPC client for Triton Inference Server embedding requests.

    Wraps tritonclient.grpc.aio to provide a simple interface for
    generating text embeddings via a deployed Triton ensemble model.

    Input texts are sent as raw UTF-8 bytes (BYTES datatype). The model
    returns float32 embedding vectors of fixed dimensionality.

    Attributes:
        _client: Underlying async gRPC Triton client.
        _model_name: Name of the deployed embedding model on Triton.
        _input_name: Name of the model's input tensor.
        _output_name: Name of the model's output tensor.
    """

    def __init__(self, configs: TritonConfigs) -> None:
        """Initialize the async Triton gRPC client.

        Args:
            configs: Triton server connection and model configuration.
        """
        url = f"{configs.host}:{configs.grpc_port}"
        self._client = triton_grpc.InferenceServerClient(url=url)
        self._model_name = configs.model_name
        self._input_name = configs.input_name
        self._output_name = configs.output_name

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
        inputs = self._build_inputs(texts)
        outputs = [triton_grpc.InferRequestedOutput(self._output_name)]

        triton_result = await self._client.infer(
            model_name=self._model_name,
            inputs=inputs,
            outputs=outputs,
        )

        return self._parse_output(triton_result)

    def _build_inputs(self, texts: list[str]) -> list[triton_grpc.InferInput]:
        """Convert a batch of texts into a Triton InferInput tensor.

        Each text is encoded as UTF-8 bytes and packed into a numpy object
        array compatible with Triton's BYTES datatype.

        Args:
            texts: Raw text strings to embed.

        Returns:
            Single-element list containing the prepared InferInput.
        """
        input_data: npt.NDArray[np.object_] = np.array(
            [text.encode("utf-8") for text in texts],
            dtype=np.object_,
        )
        infer_input = triton_grpc.InferInput(self._input_name, [len(input_data)], "BYTES")
        infer_input.set_data_from_numpy(input_data)
        return [infer_input]

    def _parse_output(self, triton_result: triton_grpc.InferResult) -> list[list[float]]:
        """Extract embedding vectors from a Triton inference result.

        Args:
            result: Raw inference result returned by triton_grpc.infer().

        Returns:
            List of embedding vectors, one float32 vector per input text.
        """
        return triton_result.as_numpy(self._output_name).tolist()
