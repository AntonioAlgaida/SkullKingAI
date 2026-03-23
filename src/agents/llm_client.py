# src/agents/llm_client.py

import re
import logging
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """
    OpenAI-compatible client for vLLM / LM Studio / Ollama.

    All generation settings are fixed at construction time so each run script
    (wake / sleep / pruning) can instantiate the client with the right profile
    without any per-call overrides leaking through the codebase.

    Supports Qwen3 hybrid thinking mode and DeepSeek-R1 reasoning models via
    the `enable_thinking` flag (passed as extra_body to vLLM).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "lm-studio",
        model_name: str = "Qwen/Qwen3-14B-AWQ",
        temperature: float = 0.35,
        top_p: float = 0.90,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
    ):
        self.client  = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model_name      = model_name
        self.temperature     = temperature
        self.top_p           = top_p
        self.max_tokens      = max_tokens
        self.enable_thinking = enable_thinking

    # ------------------------------------------------------------------ #
    # Synchronous (used by reflector + pruner in the sleep/pruning cycles) #
    # ------------------------------------------------------------------ #

    def get_move_with_content(self, prompt: str, system_prompt: str = None) -> tuple:
        """Returns (full_text, parsed_action_id).
        Pass system_prompt to move the static rule bundle into the system role
        so vLLM prefix caching can reuse it across all concurrent calls."""
        system = system_prompt or "You are a Skull King expert. Always end your response with [ACTION]: <ID>."
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                },
            )
            message  = response.choices[0].message
            # Qwen3 and DeepSeek-R1 distills expose chain-of-thought in
            # reasoning_content; older/non-reasoning models may use reasoning.
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            content   = message.content or ""
            full_text = f"{reasoning}\n{content}".strip()

            if response.usage:
                logger.info(
                    f"[sync] tokens: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}"
                )

            action_id = self._parse_action(full_text)
            return full_text, action_id

        except Exception as e:
            logger.error(f"LLMClient sync error: {e}")
            return f"Error: {e}", -1

    # ------------------------------------------------------------------ #
    # Asynchronous (used by the wake cycle for concurrent games)          #
    # ------------------------------------------------------------------ #

    async def a_get_move_with_content(self, prompt: str, system_prompt: str = None) -> tuple:
        """Returns (full_text, parsed_action_id). Async version for game play."""
        system = system_prompt or "You are a Skull King expert. Output strictly in the requested format."
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                },
            )
            message   = response.choices[0].message
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            content   = message.content or ""
            full_text = f"{reasoning}\n{content}".strip()

            if response.usage:
                logger.debug(
                    f"[async] tokens: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}"
                )

            action_id = self._parse_action(full_text)
            logger.debug(f"Parsed action ID: {action_id}")
            return full_text, action_id

        except Exception as e:
            logger.error(f"LLMClient async error: {e}")
            return f"Error: {e}", -1

    # ------------------------------------------------------------------ #
    # General text generation (reflection / pruning — no action parsing) #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Synchronous text generation for non-game tasks (reflection, pruning).
        Returns full_text (reasoning + content). No action-ID parsing."""
        system = system_prompt or "You are a helpful AI assistant."
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
            message   = response.choices[0].message
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            content   = message.content or ""
            full_text = f"{reasoning}\n{content}".strip()
            if response.usage:
                logger.info(
                    f"[generate] tokens: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}"
                )
            return full_text
        except Exception as e:
            logger.error(f"LLMClient generate error: {e}")
            return f"Error: {e}"

    async def a_generate(self, prompt: str, system_prompt: str = None) -> str:
        """Async text generation for sleep cycle (reflection run in parallel)."""
        system = system_prompt or "You are a helpful AI assistant."
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": self.enable_thinking}},
            )
            message   = response.choices[0].message
            reasoning = (
                getattr(message, "reasoning_content", None)
                or getattr(message, "reasoning", None)
                or ""
            )
            content   = message.content or ""
            full_text = f"{reasoning}\n{content}".strip()
            if response.usage:
                logger.info(
                    f"[a_generate] tokens: prompt={response.usage.prompt_tokens}, "
                    f"completion={response.usage.completion_tokens}"
                )
            return full_text
        except Exception as e:
            logger.error(f"LLMClient a_generate error: {e}")
            return f"Error: {e}"

    # ------------------------------------------------------------------ #
    # Action parser                                                        #
    # ------------------------------------------------------------------ #

    def _parse_action(self, content: str) -> int:
        """
        Robust parser for both reasoning and non-reasoning models.
        1. Strips <think>…</think> blocks so numbers in CoT don't interfere.
        2. Finds the LAST [ACTION] or [BID] marker (handles self-corrections).
        3. Falls back to the last number in the cleaned text.
        """
        # Strip Qwen3 / DeepSeek thinking tags
        clean = re.sub(r"◁think▷.*?◁/think▷", "", content, flags=re.DOTALL)
        clean = re.sub(r"<think>.*?</think>",   "", clean,   flags=re.DOTALL)

        # Priority 1: [ACTION]: <number>
        markers = list(re.finditer(r"ACTION", clean, re.IGNORECASE))
        if markers:
            tail    = clean[markers[-1].end() : markers[-1].end() + 50]
            numbers = re.findall(r"\d+", tail)
            if numbers:
                return int(numbers[0])

        # Priority 2: [BID]: <number>  (bidding phase fallback)
        markers = list(re.finditer(r"BID", clean, re.IGNORECASE))
        if markers:
            tail    = clean[markers[-1].end() : markers[-1].end() + 50]
            numbers = re.findall(r"\d+", tail)
            if numbers:
                return int(numbers[0])

        # Priority 3: last number in the entire cleaned text
        all_numbers = re.findall(r"\d+", clean)
        if all_numbers:
            return int(all_numbers[-1])

        logger.warning(f"Parser failed on: {content[:120]}…")
        return -1
