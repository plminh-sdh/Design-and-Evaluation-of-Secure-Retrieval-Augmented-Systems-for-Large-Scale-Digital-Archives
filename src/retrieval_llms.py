"""Small Hugging Face LLM drivers for retrieval evaluation notebooks."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


QUERY_GENERATION_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RELEVANCE_JUDGE_MODEL_ID = "google/gemma-2-9b-it"


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
        generation_config: GenerationConfig | None = None,
    ) -> None:
        self.model_id = model_id
        self.token = token or os.getenv("HUGGING_FACE_TOKEN")
        self.generation_config = generation_config or GenerationConfig()

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

    @property
    def device(self) -> str:
        return str(getattr(self.model, "device", "device_map:auto"))

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
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

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


class QueryGenerationLLM(HuggingFaceChatLLM):
    """Driver for generating retrieval evaluation queries and qrels."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(QUERY_GENERATION_MODEL_ID, **kwargs)

    def generate_query_record(
        self,
        chunk: Mapping[str, Any],
        *,
        prompt_version: str = "v1",
        max_new_tokens: int = 512,
    ) -> str:
        system_prompt = (
            "You generate realistic retrieval evaluation queries from public archive "
            "chunks. Return one strict JSON object and do not add commentary."
        )
        user_prompt = (
            "Create one retrieval query record from this public archive chunk.\n"
            "The query must be answerable from the chunk, must not mention 'chunk', "
            "'passage', or 'document', and must avoid inventing facts.\n\n"
            "Return keys: query, expected_relevant_information, reference_answer, "
            "query_type, difficulty.\n\n"
            f"Prompt version: {prompt_version}\n"
            f"Chunk metadata: {json.dumps(dict(chunk), ensure_ascii=False)[:12000]}"
        )
        return self.prompt(system_prompt, user_prompt, max_new_tokens=max_new_tokens)


class RelevanceJudgeLLM(HuggingFaceChatLLM):
    """Driver for judging retrieved chunks against generated qrels."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(RELEVANCE_JUDGE_MODEL_ID, **kwargs)

    def judge_relevance(
        self,
        query: str,
        expected_relevant_information: str,
        retrieved_chunk_text: str,
        *,
        max_new_tokens: int = 256,
    ) -> str:
        system_prompt = (
            "You are a strict retrieval evaluator. Return one JSON object with "
            "relevance_score from 0 to 3 and a short rationale."
        )
        user_prompt = (
            "Judge whether the retrieved chunk satisfies the query.\n\n"
            f"Query: {query}\n\n"
            f"Expected relevant information: {expected_relevant_information}\n\n"
            f"Retrieved chunk:\n{retrieved_chunk_text[:12000]}"
        )
        return self.prompt(system_prompt, user_prompt, max_new_tokens=max_new_tokens)


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
