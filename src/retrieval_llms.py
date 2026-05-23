"""Small Hugging Face LLM drivers for retrieval evaluation notebooks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping


QUERY_GENERATION_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RELEVANCE_JUDGE_MODEL_ID = "gpt-5-nano"

QUERY_GENERATION_ALLOWED_QUERY_TYPES = [
    "factual_lookup",
    "entity_centric",
    "event_centric",
    "relationship_centric",
    "summary_style",
    "temporal_contextual",
    "ocr_document_oriented",
    "video_caption_oriented",
]
QUERY_GENERATION_ALLOWED_DIFFICULTIES = ["easy", "medium", "hard"]


@dataclass(frozen=True)
class GenerationConfig:
    """Default decoding settings for deterministic-ish notebook generation."""

    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = True


class HuggingFaceChatLLM:
    """Minimal chat-style wrapper around a local Hugging Face causal LM."""

    def __init__(
        self,
        model_id: str,
        *,
        token: str | None = None,
        device_map: str | Mapping[str, Any] = "auto",
        torch_dtype: str | torch.dtype = "auto",
        load_in_4bit: bool = False,
        require_cuda: bool = False,
        allow_cpu_offload: bool = False,
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.token = token or os.getenv("HUGGING_FACE_TOKEN")
        self.generation_config = generation_config or GenerationConfig()

        import torch
        from huggingface_hub import login
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required for this LLM, but torch.cuda.is_available() is False. "
                "Reinstall a CUDA-enabled PyTorch build and restart the kernel."
            )

        if require_cuda and load_in_4bit and device_map == "auto":
            # bitsandbytes 4-bit modules should stay on GPU. With a small GPU,
            # device_map="auto" may try CPU/disk offload and fail before our
            # post-load device checks can run.
            device_map = {"": 0}

        if self.token:
            login(token=self.token, add_to_git_credential=False)

        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16
                if torch.cuda.is_available()
                else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=self.token,
            trust_remote_code=True,
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=self.token,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                trust_remote_code=True,
            )
        except Exception as exc:
            message = str(exc)
            if "torchvision::nms" in message or "Qwen2ForCausalLM" in message:
                raise RuntimeError(
                    "Transformers failed while importing the text model because an "
                    "incompatible torchvision package is installed. This retrieval "
                    "notebook does not need torchvision; uninstall it and restart the "
                    "kernel, then reload the LLM. If CUDA was replaced by a CPU-only "
                    "Torch build, reinstall the CUDA Torch build before loading models."
                ) from exc
            raise
        self.model.eval()

        if require_cuda:
            self._assert_cuda_dispatch(allow_cpu_offload=allow_cpu_offload)

    @property
    def device(self) -> str:
        return str(getattr(self.model, "device", "device_map:auto"))

    @property
    def device_map(self) -> Mapping[str, Any] | None:
        return getattr(self.model, "hf_device_map", None)

    def device_summary(self) -> dict[str, Any]:
        device_map = self.device_map
        if not device_map:
            return {
                "model_id": self.model_id,
                "device": self.device,
                "hf_device_map": None,
                "has_cpu_or_disk_offload": self.device in {"cpu", "disk"},
            }

        devices = sorted({str(device) for device in device_map.values()})
        return {
            "model_id": self.model_id,
            "device": self.device,
            "hf_device_map": dict(device_map),
            "devices": devices,
            "has_cpu_or_disk_offload": any(
                device in {"cpu", "disk"} for device in devices
            ),
        }

    def _assert_cuda_dispatch(self, *, allow_cpu_offload: bool) -> None:
        summary = self.device_summary()
        device_map = summary.get("hf_device_map")
        has_cpu_or_disk_offload = bool(summary.get("has_cpu_or_disk_offload"))

        if device_map:
            devices = {str(device) for device in device_map.values()}
            has_cuda = any(device.startswith(("cuda", "0")) for device in devices)
        else:
            has_cuda = str(summary.get("device", "")).startswith("cuda")

        if not has_cuda:
            raise RuntimeError(
                f"{self.model_id} was not dispatched to CUDA. Device summary: {summary}"
            )

        if has_cpu_or_disk_offload and not allow_cpu_offload:
            raise RuntimeError(
                f"{self.model_id} was partially offloaded to CPU/disk. "
                f"Device summary: {summary}. Use 4-bit quantization, a smaller model, "
                "or allow_cpu_offload=True if slow CPU/disk offload is acceptable."
            )

    def chat(
        self,
        messages: list[Mapping[str, str]],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> str:
        """Generate a response from OpenAI-style chat messages."""

        config = self.generation_config
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            if "System role not supported" not in str(exc):
                raise
            prompt = self.tokenizer.apply_chat_template(
                self._fold_system_prompt_into_user(messages),
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
                if max_new_tokens is not None
                else config.max_new_tokens,
                temperature=temperature if temperature is not None else config.temperature,
                top_p=top_p if top_p is not None else config.top_p,
                do_sample=do_sample if do_sample is not None else config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def prompt(self, system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
        return self.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )

    @staticmethod
    def _fold_system_prompt_into_user(
        messages: list[Mapping[str, str]],
    ) -> list[dict[str, str]]:
        """Adapt OpenAI-style system prompts for chat templates without system role."""

        system_parts: list[str] = []
        folded_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            else:
                folded_messages.append({"role": role, "content": content})

        if system_parts:
            system_text = "\n\n".join(part for part in system_parts if part).strip()
            if folded_messages and folded_messages[0]["role"] == "user":
                folded_messages[0]["content"] = (
                    f"{system_text}\n\n{folded_messages[0]['content']}"
                ).strip()
            else:
                folded_messages.insert(0, {"role": "user", "content": system_text})
        return folded_messages


class QueryGenerationLLM(HuggingFaceChatLLM):
    """Driver for generating retrieval evaluation queries and qrels."""

    def __init__(self, model_id: str = QUERY_GENERATION_MODEL_ID, **kwargs: Any) -> None:
        super().__init__(model_id, **kwargs)

    def generate_query_record(
        self,
        chunk: Mapping[str, Any],
        *,
        prompt_version: str = "v1",
        max_new_tokens: int = 768,
        temperature: float | None = None,
        top_p: float | None = None,
        do_sample: bool | None = None,
    ) -> str:
        system_prompt = (
            "You generate high-quality retrieval evaluation qrels from public archive "
            "content. Your job is to create realistic user information needs, not "
            "quiz questions about a supplied passage. The qrel must be grounded only "
            "in the provided chunk metadata and masked text. Return exactly one valid "
            "JSON object. Do not wrap it in Markdown. Do not add commentary."
        )
        user_prompt = (
            "Create one retrieval query/qrel record from this public archive item.\n\n"
            "Evaluation purpose:\n"
            "- The query will later be used to compare dense, sparse, hybrid, and graph-enhanced retrieval.\n"
            "- The source chunk is one known relevant result, but other chunks may also be "
            "relevant if they contain the same expected information.\n\n"
            "Strict grounding rules:\n"
            "- Use only facts supported by the provided metadata and masked_text.\n"
            "- Do not invent names, dates, locations, relationships, conclusions, or causes.\n"
            "- Do not use raw/unmasked text if it is not present.\n"
            "- If the text is too noisy, too short, metadata-only, or not answerable, return "
            "a JSON object with reject=true and explain why.\n\n"
            "Query quality rules:\n"
            "- Write a natural user query that someone might type into an archive search system.\n"
            "- The query must be answerable from the provided text.\n"
            "- The query must not mention 'chunk', 'passage', 'document', 'metadata', "
            "'excerpt', 'provided text', or similar source-framing words.\n"
            "- Prefer queries that require useful retrieval signals: entities, events, "
            "relationships, temporal context, aliases, or wording that is not just a long "
            "copy of the source sentence.\n"
            "- Avoid overly broad prompts such as 'What happened?' or 'Summarize this'.\n"
            "- Avoid yes/no questions unless the expected information is specific.\n\n"
            "Expected-information rules:\n"
            "- expected_relevant_information must be self-contained and include the core "
            "entity, event, relationship, or topic from the query.\n"
            "- Do not write only a bare value such as a year, name, or location.\n"
            "- Good: 'The Riverside Public Library opened its digital archive in 2024.'\n"
            "- Bad: 'The year 2024.'\n"
            "- The reference_answer may be shorter, but expected_relevant_information must "
            "state what the answer is about.\n\n"
            "Output schema:\n"
            "{\n"
            "  \"reject\": false,\n"
            "  \"query\": \"...\",\n"
            "  \"expected_relevant_information\": \"The minimal information a retrieved chunk must contain to be relevant.\",\n"
            "  \"reference_answer\": \"A concise answer grounded in the source text.\",\n"
            "  \"query_type\": \"one allowed query type\",\n"
            "  \"difficulty\": \"easy|medium|hard\",\n"
            "  \"grounding_evidence\": \"Short source-supported phrase or sentence fragment, not a long quote.\",\n"
            "  \"quality_notes\": \"Brief note on why this is a good retrieval query.\"\n"
            "}\n\n"
            "Allowed query_type values:\n"
            f"{json.dumps(QUERY_GENERATION_ALLOWED_QUERY_TYPES)}\n\n"
            "Allowed difficulty values:\n"
            f"{json.dumps(QUERY_GENERATION_ALLOWED_DIFFICULTIES)}\n\n"
            "If rejecting, use this schema:\n"
            "{\n"
            "  \"reject\": true,\n"
            "  \"rejection_reason\": \"...\",\n"
            "  \"query\": \"\",\n"
            "  \"expected_relevant_information\": \"\",\n"
            "  \"reference_answer\": \"\",\n"
            "  \"query_type\": \"\",\n"
            "  \"difficulty\": \"\",\n"
            "  \"grounding_evidence\": \"\",\n"
            "  \"quality_notes\": \"\"\n"
            "}\n\n"
            f"Prompt version: {prompt_version}\n"
            f"Chunk metadata and masked text:\n{json.dumps(dict(chunk), ensure_ascii=False)[:12000]}"
        )
        return self.prompt(
            system_prompt,
            user_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )


class OpenAIRelevanceJudgeLLM:
    """OpenAI-backed driver for judging retrieved chunks against generated qrels."""

    def __init__(
        self,
        model_id: str = RELEVANCE_JUDGE_MODEL_ID,
        *,
        api_key: str | None = None,
        env_path: str = ".env",
        **_: Any,
    ) -> None:
        self.model_id = model_id
        self.api_key = api_key or self._load_api_key(env_path=env_path)
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for the OpenAI relevance judge. "
                "Set it in .env, the environment, or pass api_key=..."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "Install the OpenAI SDK first: pip install openai"
            ) from exc

        self.client = OpenAI(api_key=self.api_key)

    @staticmethod
    def _load_api_key(*, env_path: str = ".env") -> str | None:
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path)
        except Exception:
            pass

        try:
            from google.colab import userdata

            value = userdata.get("OPENAI_API_KEY")
            if value:
                return value
        except Exception:
            pass

        return os.getenv("OPENAI_API_KEY")

    @property
    def device(self) -> str:
        return "openai_api"

    def device_summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": "openai",
            "device": self.device,
            "has_cpu_or_disk_offload": False,
            "api_key_configured": bool(self.api_key),
        }

    def judge_relevance(
        self,
        query: str,
        expected_relevant_information: str,
        retrieved_chunk_text: str,
        *,
        max_new_tokens: int = 128,
        temperature: float | None = 0.0,
        top_p: float | None = 1.0,
        do_sample: bool | None = False,
    ) -> str:
        system_prompt = (
            "You are a strict retrieval evaluator for an archive search system. "
            "Judge only retrieval relevance: whether the retrieved chunk contains "
            "information that satisfies the query. Do not judge writing style or "
            "whether the final answer is beautifully phrased. Return exactly one "
            "valid JSON object. Do not wrap it in Markdown. Do not add commentary."
        )
        user_prompt = (
            "Judge whether the retrieved chunk satisfies the query and contains the "
            "expected relevant information.\n\n"
            "Relevance rubric:\n"
            "3 = Perfectly relevant: directly answers the query and contains the expected information.\n"
            "2 = Highly relevant: contains useful answer information, but it is incomplete, indirect, "
            "or mixed with extraneous content.\n"
            "1 = Related: about the same topic, entity, event, or context, but does not answer the query.\n"
            "0 = Irrelevant: does not provide useful information for the query.\n\n"
            "Judging rules:\n"
            "- Base the score only on the retrieved chunk text below.\n"
            "- Do not reward a chunk merely because it shares a dataset, title, entity name, or broad topic.\n"
            "- Give score 3 only when the expected information is explicitly present or unambiguously entailed.\n"
            "- Give score 2 when the chunk would help answer the query but misses part of the expected information.\n"
            "- Give score 1 when it is topically related but cannot answer the query.\n"
            "- Give score 0 when it is unrelated or too vague/noisy to support the query.\n"
            "- If the retrieved text contains masked/redacted spans, judge the visible evidence only.\n\n"
            "Return this JSON schema:\n"
            "{\n"
            "  \"relevance_score\": 0,\n"
            "  \"relevance_label\": \"irrelevant|related|highly_relevant|perfectly_relevant\",\n"
            "  \"contains_expected_information\": false,\n"
            "  \"missing_information\": \"What is missing, or empty string if nothing is missing.\",\n"
            "  \"supporting_evidence\": \"Short evidence from the retrieved chunk, or empty string.\",\n"
            "  \"rationale\": \"One concise explanation of the score.\"\n"
            "}\n\n"
            f"Query: {query}\n\n"
            f"Expected relevant information: {expected_relevant_information}\n\n"
            f"Retrieved chunk:\n{retrieved_chunk_text[:12000]}"
        )
        request = {
            "model": self.model_id,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "retrieval_relevance_judgment",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "relevance_score": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 3,
                            },
                            "relevance_label": {
                                "type": "string",
                                "enum": [
                                    "irrelevant",
                                    "related",
                                    "highly_relevant",
                                    "perfectly_relevant",
                                ],
                            },
                            "contains_expected_information": {"type": "boolean"},
                            "missing_information": {"type": "string"},
                            "supporting_evidence": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": [
                            "relevance_score",
                            "relevance_label",
                            "contains_expected_information",
                            "missing_information",
                            "supporting_evidence",
                            "rationale",
                        ],
                    },
                    "strict": True,
                }
            },
            "max_output_tokens": max_new_tokens,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if top_p is not None:
            request["top_p"] = top_p

        try:
            response = self.client.responses.create(**request)
        except Exception as exc:
            message = str(exc).lower()
            if "temperature" not in message and "top_p" not in message:
                raise
            request.pop("temperature", None)
            request.pop("top_p", None)
            response = self.client.responses.create(**request)
        return response.output_text.strip()


RelevanceJudgeLLM = OpenAIRelevanceJudgeLLM


def extract_first_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object from an LLM response."""

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in LLM response.")

    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("The first JSON value in the LLM response is not an object.")
    return parsed
