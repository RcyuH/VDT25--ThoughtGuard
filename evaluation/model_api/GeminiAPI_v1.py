#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 06:58:03 2025

@author: rcyuh
"""

from openai import OpenAI
import time
import json
import random
import string
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("./model_api")
from BaseAPI import BaseAPI


class GeminiAPI(BaseAPI):
    def __init__(self, model_name="google/gemini-2.0-flash-001", generation_config={}, aligner_path=None):
        super().__init__(generation_config)
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key="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",
        )
        if "temperature" not in self.generation_config:
            self.generation_config["temperature"] = 1.0
        self.sys_prompt = self.without_strict_jsonformat_sys_prompt

        self.has_aligner = aligner_path is not None
        if self.has_aligner:
            self.aligner_tokenizer = AutoTokenizer.from_pretrained(aligner_path)
            self.aligner_model = AutoModelForCausalLM.from_pretrained(
                aligner_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                device_map="auto"
            ).eval()
        else:
            self.aligner_model = None
            self.aligner_tokenizer = None

    def rewrite_thought(self, instruction, thought):
        if not self.has_aligner:
            return thought

        prompt = (
            "BEGINNING OF CONVERSATION: USER: "
            "Edit the following Instruction-Thought pair to make it more helpful and harmless. "
            f": {instruction} | {thought} ASSISTANT:"
        )
        input_ids = self.aligner_tokenizer(prompt, return_tensors='pt').to(self.aligner_model.device)
        with torch.no_grad():
            output_ids = self.aligner_model.generate(input_ids["input_ids"], max_new_tokens=512)[0]
        resp = self.aligner_tokenizer.decode(output_ids, skip_special_tokens=True)
        return resp.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in resp else resp.strip()

    def response(self, messages, tools):
        if not tools:
            tools = None
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    tools=tools,
                    messages=messages,
                    **self.generation_config
                )
                if completion is None or completion.choices is None:
                    continue
                return completion
            except Exception as e:
                print(e)
                time.sleep(1)

    def generate_response(self, messages, tools):
        tool_call_id = None
        func_name = None
        for i, message in enumerate(messages):
            if "function_call" in message:
                tool_call_id = "".join(
                    random.sample(string.ascii_letters + string.digits, 9)
                )
                new_message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": message["function_call"],
                        }
                    ],
                }
                messages[i] = new_message

            elif message["role"] == "function":
                new_message = {
                    "role": "tool",
                    "content": message["content"],
                    "tool_call_id": tool_call_id,
                    "name": func_name,
                }
                messages[i] = new_message

            if "tool_calls" in messages[i]:
                func_name = messages[i]["tool_calls"][0]["function"]["name"]

        completion = self.response(messages, tools)

        if completion is None:
            return None

        if completion.choices[0].message.tool_calls is not None:
            tool_call = completion.choices[0].message.tool_calls[0]
            tool_call_id = tool_call.id
            tool_name = tool_call.function.name
            if tool_call.function.arguments:
                arguments = json.loads(tool_call.function.arguments)
            else:
                arguments = {}
            return {
                "type": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "arguments": arguments,
            }

        else:
            content = completion.choices[0].message.content
            instruction = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            aligned_thought = self.rewrite_thought(instruction, content)
            return {"type": "content", "content": aligned_thought}
