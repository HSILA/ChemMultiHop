import argparse
import time
import json
import tqdm
import math
import os
import re

import ollama
import boto3
from botocore.config import Config
from openai import OpenAI, NotGiven

from enum import Enum
from typing import Any, Union, Callable, Optional
from pathlib import Path
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import concurrent.futures
from threading import Lock
from datetime import datetime
import requests
import yaml

load_dotenv()


def read_json(file_path: str) -> dict:
    """Read the JSON file from the given file path."""
    with open(file_path, "r") as file:
        return json.load(file)


class Answer(BaseModel):
    answer: str = Field(..., description="A short answer to the question")


class AreSimilar(BaseModel):
    are_the_same: bool = Field(..., description="Whether the two answers are the same.")


JSON_ENFORCE = (
    "Please return only the raw JSON string that strictly conforms to the following JSON schema, with no additional text: {json_schema}"
    "Example: {example}"
)


DEFAULT_VERIFIER_PROMPT = (
    "Expected answer: {expected}\n"
    "Candidate answer: {candidate}\n"
    "Are these two answers referring to the same entity? Answer with true or false."
)


class Provider(str, Enum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    NVIDIA = "nvidia"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"


class ModelRegistry:
    """Registry to manage provider-model relationships and benchmarking state."""

    PROVIDER_MODELS = {
        Provider.OPENAI: [
            "gpt-4o",
            "gpt-4o-mini",
            # "o1",
            "o1-mini",
            "o3-mini",
            "gpt-5-mini",
            "gpt-5",
        ],
        Provider.BEDROCK: [
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            "us.meta.llama3-3-70b-instruct-v1:0",
            "mistral.mistral-large-2402-v1:0",
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0-reasoning",
            "us.deepseek.r1-v1:0-reasoning",
        ],
        Provider.OLLAMA: [
            # "gemma3:27b",
            # "deepseek-r1:32b-reasoning",
        ],
        Provider.OPENROUTER: [
            "google/gemma-3-27b-it",
            "deepseek/deepseek-r1-distill-qwen-32b-reasoning",
            "qwen/qwq-32b-reasoning",
        ],
        Provider.NVIDIA: [
            # "deepseek-ai/deepseek-r1",
        ],
    }

    def __init__(self):
        self.completed_benchmarks = set()

    def get_models_for_provider(self, provider: Provider) -> list[str]:
        """Get all models supported by a specific provider."""
        return self.PROVIDER_MODELS.get(provider, [])

    def is_valid_model(self, provider: Provider, model: str) -> bool:
        """Check if a model is valid for a given provider."""
        return model in self.PROVIDER_MODELS.get(provider, [])

    def get_all_provider_model_combinations(self) -> list[tuple[Provider, str]]:
        """Get all valid provider-model combinations."""
        combinations = []
        for provider in Provider:
            for model in self.get_models_for_provider(provider):
                combinations.append((provider, model))
        return combinations

    def mark_as_completed(self, provider: Provider, model: str):
        """Mark a provider-model combination as completed."""
        self.completed_benchmarks.add((provider, model))


class ModelSpec(BaseModel):
    provider: Provider
    model: str

    @field_validator("provider", mode="before")
    @classmethod
    def _validate_provider(cls, value: Union[Provider, str]) -> Provider:
        if isinstance(value, Provider):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            try:
                return Provider(normalized)
            except ValueError as exc:
                raise ValueError(f"Unsupported provider '{value}'.") from exc
        raise TypeError("Provider must be provided as a string or Provider enum.")

    @model_validator(mode="after")
    def _validate_model(self) -> "ModelSpec":
        registry = ModelRegistry()
        if not registry.is_valid_model(self.provider, self.model):
            raise ValueError(
                f"Model '{self.model}' is not registered for provider '{self.provider.value}'."
            )
        return self


class EvaluationConfig(BaseModel):
    responses_folder: str
    results_summary: str
    records_path: str
    verifier_prompt: str
    models: list[ModelSpec]
    verifier_model: ModelSpec
    num_workers: int = 4
    bedrock_cooldown: float = 0.5

    @model_validator(mode="before")
    @classmethod
    def _preprocess(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data is None:
            raise ValueError("Configuration data cannot be empty.")

        processed = dict(data)
        processed["models"] = cls._normalize_models(processed.get("models"))
        processed["verifier_model"] = cls._normalize_model_spec(
            processed.get("verifier_model")
        )

        return processed

    @staticmethod
    def _normalize_models(raw: Any) -> list[dict[str, str]]:
        if raw is None:
            raise ValueError("'models' must be provided.")

        if isinstance(raw, str) and raw.strip().lower() == "all":
            registry = ModelRegistry()
            return [
                {"provider": provider.value, "model": model}
                for provider, model in registry.get_all_provider_model_combinations()
            ]

        if isinstance(raw, dict):
            normalized: list[dict[str, str]] = []
            for provider_name, models in raw.items():
                if isinstance(models, str):
                    model_names = [models]
                elif isinstance(models, list):
                    model_names = models
                else:
                    raise TypeError(
                        "Model lists must be provided as strings or lists of strings."
                    )

                for model_name in model_names:
                    normalized.append(
                        {"provider": str(provider_name), "model": str(model_name)}
                    )

            if not normalized:
                raise ValueError("At least one model must be specified under 'models'.")
            return normalized

        raise TypeError(
            "'models' must be a mapping of provider to model names or the string 'all'."
        )

    @staticmethod
    def _normalize_model_spec(raw: Any) -> dict[str, str]:
        if raw is None:
            raise ValueError("'verifier_model' must be provided.")

        if isinstance(raw, ModelSpec):
            return {"provider": raw.provider.value, "model": raw.model}

        if isinstance(raw, dict):
            provider = raw.get("provider")
            model = raw.get("model")
            if not provider or not model:
                raise ValueError(
                    "'verifier_model' must include both 'provider' and 'model' entries."
                )
            return {"provider": str(provider), "model": str(model)}

        raise TypeError(
            "'verifier_model' must be a mapping with 'provider' and 'model' keys."
        )

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EvaluationConfig":
        config_path = Path(path)
        with config_path.open("r") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)


class StructuredLLM:
    def __init__(
        self,
        model: Union[ModelSpec, dict[str, Any]],
        output_format: type[BaseModel],
        temperature: float = 0.2,
        max_completion_tokens: int = 8192,
    ):
        if isinstance(model, ModelSpec):
            self.model_spec = model
        elif isinstance(model, dict):
            self.model_spec = ModelSpec(**model)
        else:
            raise TypeError(
                "model must be provided as a ModelSpec or mapping with 'provider' and 'model'."
            )

        self.provider = self.model_spec.provider
        self.model_id = self.model_spec.model
        self.output_format = output_format
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.is_reasoning = False
        self.thinking_params = None

        openai_reasoning_models = {
            "o1",
            "o1-preview",
            "o1-mini",
            "o3-mini",
            "gpt-5",
            "gpt-5-mini",
        }

        if "reasoning" in self.model_id:
            self.model_id = self.model_id.replace("-reasoning", "")
            self.is_reasoning = True
            if "claude-3-7" in self.model_id:
                self.temperature = 1.0
                self.thinking_params = {
                    "thinking": {
                        "type": "enabled",
                        "budget_tokens": self.max_completion_tokens - 256,
                    }
                }
            else:
                self.temperature = 0.6

        if self.provider == Provider.OPENAI and self.model_id in openai_reasoning_models:
            self.temperature = NotGiven()

        if self.provider in [Provider.OPENAI, Provider.NVIDIA]:
            self.api_key = self._get_api_key()
        self.client = self._initialize_client()
        if self.provider == Provider.BEDROCK:
            self.bedrock_llm = self._get_bedrock_llm()

    def _get_api_key(self) -> str:
        """Get API key from environment variables based on the provider."""
        key_mapping = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.NVIDIA: "NVIDIA_API_KEY",
            Provider.OPENROUTER: "OPENROUTER_API_KEY",
        }

        if self.provider in key_mapping:
            env_var = key_mapping[self.provider]
            api_key = os.environ.get(env_var)
            if not api_key:
                raise ValueError(f"{env_var} environment variable is not set")
            return api_key

    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == Provider.OPENAI:
            return OpenAI(api_key=self.api_key)
        elif self.provider == Provider.NVIDIA:
            return OpenAI(
                base_url="https://integrate.api.nvidia.com/v1", api_key=self.api_key
            )
        elif self.provider == Provider.BEDROCK:
            session = boto3.session.Session()
            configured_region = session.region_name
            return boto3.client(
                "bedrock-runtime",
                region_name=configured_region,
                config=Config(
                    connect_timeout=300,
                    read_timeout=1000,
                    retries={"max_attempts": 3},
                ),
            )
        elif self.provider == Provider.OLLAMA:
            return ollama.Client(host="http://localhost:11434")
        elif self.provider == Provider.OPENROUTER:
            return None
        else:
            raise ValueError(f"Invalid provider: {self.provider}")

    def _parse_json_from_text(self, text_to_parse: str) -> BaseModel:
        """Extract and parse JSON from text string."""
        try:
            parsed_json = JsonOutputParser().invoke(text_to_parse)
            parsed_output = self.output_format.model_validate(parsed_json)
        except Exception as e:
            regex_match = re.search(r"(\{.*\})", text_to_parse, re.DOTALL)
            if regex_match:
                cleaned_text = regex_match.group(1)
                try:
                    parsed_json = JsonOutputParser().invoke(cleaned_text)
                    parsed_output = self.output_format.model_validate(parsed_json)
                except Exception as e:
                    print(f"Error parsing JSON: {e}")
                    parsed_output = self._generate_empty_output()
            else:
                parsed_output = self._generate_empty_output()

        return parsed_output

    def _extract_from_content(
        self, content: Union[str, list[AIMessage]]
    ) -> tuple[str, str]:
        """Extract raw_response and reasoning from model response content."""
        raw_response, reason = None, None
        try:
            if isinstance(content, str):
                raw_response = content
            elif isinstance(content, list):
                # We only have reasoning content if `content` is a list
                reason = next(
                    (
                        item["reasoning_content"]["text"]
                        for item in content
                        if item.get("type") == "reasoning_content"
                    ),
                    None,
                )
                # We don't use tool call for these models, just raw json output
                if self.model_id in [
                    "us.deepseek.r1-v1:0",
                    "mistral.mistral-large-2402-v1:0",
                ]:
                    raw_response = next(
                        (
                            item["text"]
                            for item in content
                            if item.get("type") == "text"
                        ),
                        None,
                    )
                else:
                    raw_response = next(
                        (
                            item["input"]
                            for item in content
                            if item.get("type") == "tool_use"
                        ),
                        None,
                    )
        except Exception as e:
            raw_response = None

        return raw_response, reason

    @staticmethod
    def _parse_raw_reasoning_output(raw_output: str) -> tuple[str, str]:
        """Parse the raw reasoning output from the AI model."""
        pattern = r"<think>\s*(.*?)\s*</think>\s*(.*)"
        match = re.search(pattern, raw_output, re.DOTALL)
        if match:
            reasoning_tokens = match.group(1).strip()
            final_output = match.group(2).strip()
        else:
            reasoning_tokens = None
            final_output = raw_output

        return reasoning_tokens, final_output

    def _get_bedrock_llm(self):
        """Get the Bedrock LLM model based on the model name and temperature."""
        llm = ChatBedrockConverse(
            client=self.client,
            model_id=self.model_id,
            max_tokens=self.max_completion_tokens,
            temperature=self.temperature,
            additional_model_request_fields=self.thinking_params,
        )
        llm = (
            llm.with_structured_output(self.output_format, include_raw=True)
            if self.model_id
            not in ["us.deepseek.r1-v1:0", "mistral.mistral-large-2402-v1:0"]
            else llm
        )
        return llm

    def _call_bedrock(self, messages: list[dict]) -> dict[str, Any]:
        """Call the Bedrock LLM model with the given message."""
        try:
            reason = None
            response = self.bedrock_llm.invoke(messages)
            if self.model_id in [
                "us.deepseek.r1-v1:0",
                "mistral.mistral-large-2402-v1:0",
            ]:
                content = response.content
                raw_response, reason = self._extract_from_content(content)
                parsed_output = self._parse_json_from_text(raw_response)
            else:
                parsed_output = response["parsed"]
                response = response["raw"]
                raw_response, reason = self._extract_from_content(response.content)

            usage_metadata = response.usage_metadata
            latency = response.response_metadata["metrics"]["latencyMs"][0]

            output = {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": latency,
                "input_tokens": usage_metadata["input_tokens"],
                "output_tokens": usage_metadata["output_tokens"],
                "reasoning": reason,
            }
            return output
        except Exception as e:
            print(f"Error calling Bedrock LLM: {e}")
            return {
                "raw_response": None,
                "parsed_output": None,
                "date": datetime.now(),
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning": None,
                "error": str(e),
            }

    def _call_ollama(self, messages: str) -> dict[str, Any]:
        """Call the Ollama API with the given messages."""
        reason = None
        response = self.client.chat(
            model=self.model_id,
            messages=messages,
            format=self.output_format.model_json_schema()
            if not self.is_reasoning
            else None,
            options={
                "temperature": self.temperature,
                "num_predicr": self.max_completion_tokens,
            },
        )
        raw_response = response.message.content
        if self.is_reasoning:
            reason, raw_response = self._parse_raw_reasoning_output(raw_response)
        parsed_output = self._parse_json_from_text(raw_response)
        latency = int(
            (response.prompt_eval_duration / 1e6) + (response.eval_duration / 1e6)
        )

        output = {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": latency,
            "input_tokens": response.prompt_eval_count,
            "output_tokens": response.eval_count,
            "reasoning": reason,
        }
        return output

    def _call_openai(self, messages: str) -> dict[str, Any]:
        """Call the OpenAI API with the given messages."""
        try:
            now = time.time()
            if self.model_id == "o1-mini":
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_completion_tokens=self.max_completion_tokens,
                )
                raw_response = response.choices[0].message.content
                parsed_output = self._parse_json_from_text(raw_response)
            else:
                response = self.client.beta.chat.completions.parse(
                    model=self.model_id,
                    messages=messages,
                    response_format=self.output_format,
                    temperature=self.temperature,
                    max_completion_tokens=self.max_completion_tokens,
                )
                raw_response = response.choices[0].message.content
                parsed_output = response.choices[0].message.parsed
            elapsed_ms = (time.time() - now) * 1000
            output = {
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "date": datetime.now(),
                "latency": elapsed_ms,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "reasoning_tokens": response.usage.completion_tokens_details.reasoning_tokens,
            }
            return output
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {
                "raw_response": None,
                "parsed_output": None,
                "date": datetime.now(),
                "latency": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "reasoning_tokens": 0,
                "error": str(e),
            }

    def _call_openrouter(self, messages: str) -> dict[str, Any]:
        """Call the OpenRouter API with the given messages."""
        reason = None
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": self.output_format.__name__,
                "strict": True,
                "schema": self.output_format.model_json_schema(),
            },
        }

        payload = {
            "model": self.model_id,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": response_format,
            "max_tokens": self.max_completion_tokens,
        }

        if self.model_id == "qwen/qwq-32b":
            payload["top_k"] = 40
            payload["top_p"] = 0.95

        if self.is_reasoning:
            payload.pop("response_format")

        max_retries = 3
        for retry in range(max_retries):
            try:
                now = time.time()
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._get_api_key()}",
                    },
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                response_json = response.json()
                elapsed_ms = (time.time() - now) * 1000

                raw_response = response_json["choices"][0]["message"]["content"]
                parsed_output = self._parse_json_from_text(raw_response)

                if "reasoning" in response_json["choices"][0]["message"]:
                    reason = response_json["choices"][0]["message"]["reasoning"]

                return {
                    "raw_response": raw_response,
                    "parsed_output": parsed_output,
                    "date": datetime.now(),
                    "latency": elapsed_ms,
                    "input_tokens": response_json["usage"]["prompt_tokens"],
                    "output_tokens": response_json["usage"]["completion_tokens"],
                    "reasoning": reason,
                }
            except Exception as e:
                if retry < max_retries - 1:
                    time.sleep(2 * (retry + 1))  # exponential backoff
                else:
                    return {
                        "raw_response": None,
                        "parsed_output": self._generate_empty_output(),
                        "date": datetime.now(),
                        "latency": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "reasoning": None,
                        "error": str(e),
                    }

    def _call_nvidia(self, messages: str) -> dict[str, Any]:
        now = time.time()
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.6,
            top_p=0.7,
            max_tokens=self.max_completion_tokens,
        )
        elapsed_ms = (time.time() - now) * 1000
        response_text = response.choices[0].message.content

        reason, raw_response = self._parse_raw_reasoning_output(response_text)

        parsed_output = self._parse_json_from_text(raw_response)

        output = {
            "raw_response": raw_response,
            "parsed_output": parsed_output,
            "date": datetime.now(),
            "latency": elapsed_ms,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "reasoning": reason,
        }
        return output

    def _generate_empty_output(self):
        """Create an empty instance of the output format."""
        field_types = self.output_format.__annotations__
        fields = {}
        for field_name, field_type in field_types.items():
            if field_type == str:
                fields[field_name] = ""
            elif field_type == bool:
                fields[field_name] = False
            elif field_type == int:
                fields[field_name] = 0
            elif field_type == float:
                fields[field_name] = 0.0
            else:
                fields[field_name] = None
        return self.output_format(**fields)

    def __call__(self, prompt: str) -> dict[str, Any]:
        """Evaluate the given messages using the appropriate LLM model."""
        # TODO: Add support for dynamic example, instead of hardcoding
        json_schema = JSON_ENFORCE.format(
            json_schema=self.output_format.model_json_schema(),
            example='{"answer": "Aspirin"}',
        )
        prompt = f"{prompt}\n{json_schema}"

        if self.provider == Provider.BEDROCK:
            messages = [{"role": "user", "content": [{"text": prompt}]}]
            return self._call_bedrock(messages)
        elif self.provider == Provider.OPENAI:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self._call_openai(messages)
        elif self.provider == Provider.NVIDIA:
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            return self._call_nvidia(messages)
        elif self.provider == Provider.OLLAMA:
            messages = [{"role": "user", "content": prompt}]
            return self._call_ollama(messages)
        elif self.provider == Provider.OPENROUTER:
            messages = [{"role": "user", "content": prompt}]
            return self._call_openrouter(messages)


class Evaluate:
    def __init__(
        self,
        qa_llm: StructuredLLM,
        records: list[dict],
        use_context: bool,
        verifier_model: ModelSpec,
        responses_save_path: Optional[str] = None,
        verifier_prompt: Optional[str] = None,
        num_workers: int = 4,
        bedrock_cooldown: float = 0.5,
    ):
        self.qa_llm = qa_llm
        self.verifier_model = verifier_model
        self.verifier_llm = StructuredLLM(
            model=self.verifier_model.model_copy(),
            output_format=AreSimilar,
        )
        self.records = records
        self.responses_save_path = responses_save_path
        self.num_workers = num_workers
        self.bedrock_cooldown = bedrock_cooldown
        self.use_context = use_context
        self.verifier_prompt = verifier_prompt or DEFAULT_VERIFIER_PROMPT

        self.file_lock = Lock()

        records_per_worker = len(records) / num_workers
        self.batch_size = max(1, math.ceil(records_per_worker))

        self.qa_llm_config = {
            "model": qa_llm.model_spec.model_copy(),
            "output_format": qa_llm.output_format,
            "temperature": qa_llm.temperature,
            "max_completion_tokens": qa_llm.max_completion_tokens,
        }

        self.verifier_llm_config = {
            "model": self.verifier_model.model_copy(),
            "output_format": AreSimilar,
            "temperature": self.verifier_llm.temperature,
            "max_completion_tokens": self.verifier_llm.max_completion_tokens,
        }

    def _save_result_to_jsonl(self, result: dict):
        """Save a single result to the JSONL file incrementally"""
        if not self.responses_save_path:
            return
        with self.file_lock:
            with open(self.responses_save_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

    def _verify_entity(
        self, expected: str, candidate: str, worker_verifier_llm=None
    ) -> bool:
        verifier_llm = worker_verifier_llm or self.verifier_llm

        if candidate.strip().lower() == expected.strip().lower():
            return True
        else:
            try:
                prompt = self.verifier_prompt.format(
                    expected=expected, candidate=candidate
                )
            except KeyError:
                prompt = DEFAULT_VERIFIER_PROMPT.format(
                    expected=expected, candidate=candidate
                )
            response = verifier_llm(prompt)
            return response["parsed_output"].are_the_same

    def _create_worker_llms(self):
        """Create new LLM instances for workers to avoid thread safety issues"""
        qa_config = dict(self.qa_llm_config)
        qa_config["model"] = qa_config["model"].model_copy()
        qa_llm = StructuredLLM(**qa_config)

        verifier_config = dict(self.verifier_llm_config)
        verifier_config["model"] = verifier_config["model"].model_copy()
        verifier_llm = StructuredLLM(**verifier_config)
        return qa_llm, verifier_llm

    def _process_record_without_context(self, record, worker_llms=None):
        """Process a single record without context (thread-safe)"""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        expected = record["expected"]

        NO_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question:\n"
            f"Question: {question}"
        )

        response = qa_llm(NO_CONTEXT_PROMPT)

        if response["parsed_output"] is not None:
            candidate = response["parsed_output"].answer
            is_correct = (
                self._verify_entity(expected, candidate, verifier_llm)
                if expected
                else False
            )
        else:
            candidate = None
            is_correct = False

        result = {
            "candidate": candidate,
            "is_correct": is_correct,
            "raw_response": response.get("raw_response", None),
            "context_used": False,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "raw": record,
            "error": response.get("error", None),
        }

        self._save_result_to_jsonl(result)

        return result

    def _process_record_with_context(self, record, worker_llms=None):
        """Process a single record with context (thread-safe)"""
        qa_llm, verifier_llm = worker_llms or (self.qa_llm, self.verifier_llm)

        question = record["question"]
        context = record["context"]
        expected = record["expected"]

        WITH_CONTEXT_PROMPT = (
            "Provide a single entity name as an answer to the following question based on the context:\n"
            f"Question: {question}\n\n"
            f"Context: {context}"
        )

        response = qa_llm(WITH_CONTEXT_PROMPT)

        if response["parsed_output"] is not None:
            candidate = response["parsed_output"].answer
            is_correct = (
                self._verify_entity(expected, candidate, verifier_llm)
                if expected
                else False
            )
        else:
            candidate = None
            is_correct = False

        result = {
            "candidate": candidate,
            "is_correct": is_correct,
            "raw_response": response.get("raw_response", None),
            "context_used": True,
            "input_tokens": response.get("input_tokens", 0),
            "output_tokens": response.get("output_tokens", 0),
            "reasoning_tokens": response.get("reasoning_tokens", 0),
            "latency": round(response.get("latency", 0), 2),
            "date": response.get("date", datetime.now()).strftime("%Y-%m-%d %H:%M"),
            "reasoning": response.get("reasoning", None),
            "raw": record,
            "error": response.get("error", None),
        }

        self._save_result_to_jsonl(result)

        return result

    def _process_batch(self, batch, process_fn, progress_callback=None):
        """Process a batch of records with the given processing function"""
        # Create worker-specific LLM instances to avoid thread safety issues
        worker_llms = self._create_worker_llms()
        results = []

        for record in batch:
            result = process_fn(record, worker_llms)
            results.append(result)

            if progress_callback:
                progress_callback(1)

            if worker_llms[0].provider == Provider.BEDROCK:
                time.sleep(self.bedrock_cooldown)

        return results

    def _parallel_process(self, process_fn: Callable):
        """Process all records in parallel using the provided function"""
        results = []
        total_records = len(self.records)

        batches = [
            self.records[i : i + self.batch_size]
            for i in range(0, len(self.records), self.batch_size)
        ]

        with tqdm.tqdm(total=total_records, desc="Processing records") as pbar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.num_workers
            ) as executor:
                futures = []

                def update_progress(n):
                    pbar.update(n)

                for batch in batches:
                    futures.append(
                        executor.submit(
                            self._process_batch,
                            batch=batch,
                            process_fn=process_fn,
                            progress_callback=update_progress,
                        )
                    )

                for future in concurrent.futures.as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)

        assert len(results) == len(self.records), "Some records were not processed"

        return results

    def _eval_without_context(self):
        """Evaluate records without context using parallel processing"""
        self._parallel_process(self._process_record_without_context)

    def _eval_with_context(self):
        """Evaluate records with context using parallel processing"""
        self._parallel_process(self._process_record_with_context)

    def evaluate(self):
        """
        Evaluate the records based on the use_context flag.
        Results are saved to the JSONL file as they are generated.
        """
        if not self.use_context:
            self._eval_without_context()
        else:
            self._eval_with_context()


class BenchmarkRunner:
    """Class to run benchmarks across multiple models and providers."""

    def __init__(
        self,
        records: list[dict],
        verifier_model: ModelSpec,
        responses_dir: str = "responses",
        results_file: str = "results.csv",
        verifier_prompt: str = DEFAULT_VERIFIER_PROMPT,
        num_workers: int = 4,
        bedrock_cooldown: float = 0.5,
    ):
        self.records = records
        self.responses_dir = responses_dir
        self.results_file = results_file
        self.model_registry = ModelRegistry()
        self.verifier_model = verifier_model
        self.verifier_prompt = verifier_prompt or DEFAULT_VERIFIER_PROMPT
        self.num_workers = num_workers
        self.bedrock_cooldown = bedrock_cooldown
        os.makedirs(responses_dir, exist_ok=True)

        if not os.path.exists(results_file):
            with open(results_file, "w") as f:
                f.write(
                    "Model,Run,Accuracy (%),Avg Duration (s),Avg Input Tokens,Avg Output Tokens,Total Input Tokens,Total Output Tokens,Total Samples\n"
                )

    def _get_processed_questions(self, responses_path: str):
        """Get questions that have already been processed for a model"""
        questions_without_context = set()
        questions_with_context = set()

        if not os.path.exists(responses_path):
            return questions_without_context, questions_with_context

        try:
            with open(responses_path, "r") as f:
                for line in f:
                    try:
                        response = json.loads(line.strip())
                        if "raw" in response and "question" in response["raw"]:
                            question = response["raw"]["question"]
                            if response.get("context_used", False):
                                questions_with_context.add(question)
                            else:
                                questions_without_context.add(question)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading response file {responses_path}: {e}")

        return questions_without_context, questions_with_context

    def _filter_records_for_evaluation(self, responses_path: str):
        """Filter records that need to be evaluated"""
        questions_without_context, questions_with_context = (
            self._get_processed_questions(responses_path)
        )

        records_for_no_context = []
        records_for_context = []

        for record in self.records:
            question = record["question"]
            if question not in questions_without_context:
                records_for_no_context.append(record)
            if question not in questions_with_context and "context" in record:
                records_for_context.append(record)

        return records_for_no_context, records_for_context

    def _calculate_metrics_from_responses(self, responses_path, provider, model):
        """Calculate metrics from a response file and update the results CSV"""
        if not os.path.exists(responses_path):
            return None

        with_context_results = []
        without_context_results = []

        try:
            with open(responses_path, "r") as f:
                for line in f:
                    try:
                        response = json.loads(line.strip())
                        if response.get("context_used", False):
                            with_context_results.append(response)
                        else:
                            without_context_results.append(response)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading response file {responses_path}: {e}")
            return None

        model_name = f"{provider.value}-{model}"
        results = []

        if without_context_results:
            total = len(without_context_results)
            correct = sum(
                1 for r in without_context_results if r.get("is_correct", False)
            )
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_latency = (
                sum(r.get("latency", 0) for r in without_context_results) / total
                if total > 0
                else 0
            )
            avg_in_tokens = (
                sum(r.get("input_tokens", 0) for r in without_context_results) / total
                if total > 0
                else 0
            )
            avg_out_tokens = (
                sum(r.get("output_tokens", 0) for r in without_context_results) / total
                if total > 0
                else 0
            )
            total_in_tokens = sum(
                r.get("input_tokens", 0) for r in without_context_results
            )
            total_out_tokens = sum(
                r.get("output_tokens", 0) for r in without_context_results
            )

            results.append(
                {
                    "Model": model_name,
                    "Run": "Without Context",
                    "Accuracy (%)": round(accuracy, 2),
                    "Avg Duration (s)": round(avg_latency, 2),
                    "Avg Input Tokens": round(avg_in_tokens, 2),
                    "Avg Output Tokens": round(avg_out_tokens, 2),
                    "Total Input Tokens": total_in_tokens,
                    "Total Output Tokens": total_out_tokens,
                    "Total Samples": total,
                }
            )

        if with_context_results:
            total = len(with_context_results)
            correct = sum(1 for r in with_context_results if r.get("is_correct", False))
            accuracy = (correct / total) * 100 if total > 0 else 0
            avg_latency = (
                sum(r.get("latency", 0) for r in with_context_results) / total
                if total > 0
                else 0
            )
            avg_in_tokens = (
                sum(r.get("input_tokens", 0) for r in with_context_results) / total
                if total > 0
                else 0
            )
            avg_out_tokens = (
                sum(r.get("output_tokens", 0) for r in with_context_results) / total
                if total > 0
                else 0
            )
            total_in_tokens = sum(
                r.get("input_tokens", 0) for r in with_context_results
            )
            total_out_tokens = sum(
                r.get("output_tokens", 0) for r in with_context_results
            )

            results.append(
                {
                    "Model": model_name,
                    "Run": "With Context",
                    "Accuracy (%)": round(accuracy, 2),
                    "Avg Duration (s)": round(avg_latency, 2),
                    "Avg Input Tokens": round(avg_in_tokens, 2),
                    "Avg Output Tokens": round(avg_out_tokens, 2),
                    "Total Input Tokens": total_in_tokens,
                    "Total Output Tokens": total_out_tokens,
                    "Total Samples": total,
                }
            )

        if results and self.results_file:
            self._update_results_csv(pd.DataFrame(results), model_name)
            return
        else:
            return pd.DataFrame(results)

    def _update_results_csv(self, new_df, model_name):
        """Update the results CSV file with new results, replacing existing entries for the same model"""
        if os.path.exists(self.results_file) and os.path.getsize(self.results_file) > 0:
            try:
                existing_df = pd.read_csv(self.results_file)
                existing_df = existing_df[existing_df["Model"] != model_name]
                if existing_df.empty:
                    combined_df = new_df
                else:
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                combined_df.to_csv(self.results_file, index=False)
            except Exception as e:
                print(f"Error updating results CSV: {e}")
                new_df.to_csv(self.results_file, index=False)
        else:
            new_df.to_csv(self.results_file, index=False)

    def run_benchmark_for_model(self, model_spec: ModelSpec):
        """Run benchmark for a specific provider and model"""
        model_filename = model_spec.model.replace("/", "__")
        responses_path = (
            f"{self.responses_dir}/responses_{model_spec.provider.value}_{model_filename}.jsonl"
        )

        try:
            records_no_context, records_with_context = (
                self._filter_records_for_evaluation(responses_path)
            )

            if not records_no_context and not records_with_context:
                print(
                    "All records already processed for "
                    f"{model_spec.provider.value} model: {model_spec.model}, skipping..."
                )
                self._calculate_metrics_from_responses(
                    responses_path, model_spec.provider, model_spec.model
                )
                return

            structured_llm = StructuredLLM(
                model=model_spec.model_copy(),
                output_format=Answer,
            )

            if records_no_context:
                print(f"Evaluating {len(records_no_context)} records without context")
                evaluator = Evaluate(
                    qa_llm=structured_llm,
                    records=records_no_context,
                    responses_save_path=responses_path,
                    use_context=False,
                    verifier_model=self.verifier_model,
                    verifier_prompt=self.verifier_prompt,
                    num_workers=self.num_workers,
                    bedrock_cooldown=self.bedrock_cooldown,
                )
                evaluator.evaluate()

            if records_with_context:
                print(f"Evaluating {len(records_with_context)} records with context")
                evaluator = Evaluate(
                    qa_llm=structured_llm,
                    records=records_with_context,
                    responses_save_path=responses_path,
                    use_context=True,
                    verifier_model=self.verifier_model,
                    verifier_prompt=self.verifier_prompt,
                    num_workers=self.num_workers,
                    bedrock_cooldown=self.bedrock_cooldown,
                )
                evaluator.evaluate()

            self._calculate_metrics_from_responses(
                responses_path, model_spec.provider, model_spec.model
            )

        except Exception as e:
            print(
                f"Error evaluating {model_spec.provider.value} model {model_spec.model}: {e}"
            )

    def run_selected_benchmarks(self, models: list[ModelSpec]):
        """Run benchmarks for the specified provider-model combinations."""
        for spec in models:
            print(f"\nEvaluating {spec.provider.value} model: {spec.model}")
            self.run_benchmark_for_model(spec)

    def run_all_benchmarks(self):
        """Run benchmarks for all provider-model combinations."""
        all_specs = [
            ModelSpec(provider=provider, model=model)
            for provider, model in self.model_registry.get_all_provider_model_combinations()
        ]
        self.run_selected_benchmarks(all_specs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM benchmarks from YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    config = EvaluationConfig.from_yaml(args.config)

    records = read_json(config.records_path)

    benchmark_runner = BenchmarkRunner(
        records=records,
        responses_dir=config.responses_folder,
        results_file=config.results_summary,
        verifier_model=config.verifier_model,
        verifier_prompt=config.verifier_prompt,
        num_workers=config.num_workers,
        bedrock_cooldown=config.bedrock_cooldown,
    )
    now = datetime.now()
    benchmark_runner.run_selected_benchmarks(config.models)
    elapsed = datetime.now() - now
    elapsed_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed.total_seconds()))
    print(f"Completed benchmarking in {elapsed_formatted}")
