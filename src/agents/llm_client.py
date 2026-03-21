# src/agents/llm_client.py

import re
import logging
from openai import OpenAI, AsyncOpenAI # Add AsyncOpenAI

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, base_url="http://localhost:12345/v1", api_key="lm-studio", model_name="local-model"):
        """
        A generic client that works with LM Studio, vLLM, or Ollama.
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.aclient = AsyncOpenAI(base_url=base_url, api_key=api_key) # NEW: Async client
        self.model_name = model_name

    def get_move_with_content(self, prompt: str, temperature=0.3):
        """Returns (Full String Content, Parsed Action ID)"""
        try:
                        
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Skull King expert. Always end your response with [ACTION]: <ID>."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=4096,
                top_p=0.95
            )
            message = response.choices[0].message
            
            # Reasoning models often put text in 'reasoning' and 'content'
            reasoning = getattr(message, "reasoning", "") or ""
            content = message.content or ""
            
            full_text = f"{reasoning}\n{content}"
            
            # Log Token usage
            if response.usage:
                logger.info(f"Tokens: Prompt={response.usage.prompt_tokens}, "
                            f"Completion={response.usage.completion_tokens}")

            action_id = self._parse_action(full_text)
            return content, action_id
        except Exception as e:
            logger.error(f"LLM Client Error: {e}")
            return f"Error: {e}", -1

    # NEW: Asynchronous method
    async def a_get_move_with_content(self, prompt: str, temperature=0.3):
        try:
            response = await self.aclient.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a Skull King expert. Output strictly in the requested format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=4096, # Increased to ensure reasoning isn't cut off
                extra_body={"chat_template_kwargs": {"enable_thinking": True}}
            )
            
            # print(f"Raw LLM Response: {response}")  # Debug log for raw response
            
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", "") or ""
            content = message.content or ""
            full_text = f"{reasoning}\n{content}"
            
            action_id = self._parse_action(full_text)
            
            # Log Token usage
            if response.usage:
                logger.info(f"Tokens: Prompt={response.usage.prompt_tokens}, "
                            f"Completion={response.usage.completion_tokens}")
            
            # print(f"Full LLM Response Text: {full_text}")  # Debug log for full text
            print(f"Parsed Action ID: {action_id}")  # Debug log for parsed action
            return full_text, action_id
            
        except Exception as e:
            logger.error(f"Async LLM API Error: {e}")
            return f"Error: {e}", -1

    def _parse_action(self, content: str) -> int:
        """
        Ultra-robust parser for Reasoning Models.
        1. Strips thinking blocks to avoid catching numbers in reasoning.
        2. Finds the LAST keyword 'ACTION' or 'BID'.
        3. Extracts the FIRST integer that follows that keyword.
        """
        # 1. Strip reasoning blocks to avoid noise from "thinking"
        clean_text = re.sub(r'◁think▷.*?◁/think▷', '', content, flags=re.DOTALL)
        clean_text = re.sub(r'<think>.*?</think>', '', clean_text, flags=re.DOTALL)

        # 2. Priority 1: Search for 'ACTION'
        # We find all occurrences and take the LAST one (in case of self-correction)
        action_markers = list(re.finditer(r"ACTION", clean_text, re.IGNORECASE))
        if action_markers:
            last_marker = action_markers[-1]
            # Get text following the marker (up to 50 characters is plenty)
            tail = clean_text[last_marker.end() : last_marker.end() + 50]
            # Find the FIRST number in this tail (skips <, >, ID, :, etc.)
            numbers = re.findall(r"\d+", tail)
            if numbers:
                return int(numbers[0])

        # 3. Priority 2: Search for 'BID' (Fallback for bidding rounds)
        bid_markers = list(re.finditer(r"BID", clean_text, re.IGNORECASE))
        if bid_markers:
            last_marker = bid_markers[-1]
            tail = clean_text[last_marker.end() : last_marker.end() + 50]
            numbers = re.findall(r"\d+", tail)
            if numbers:
                return int(numbers[0])

        # 4. Final Fallback: Just get the last number in the entire cleaned text
        # This handles cases where the model forgets the [ACTION] tag entirely
        all_numbers = re.findall(r"\d+", clean_text)
        if all_numbers:
            return int(all_numbers[-1])

        logger.warning(f"PARSER FAILED to find any numeric action in: {content[:100]}...")
        return -1