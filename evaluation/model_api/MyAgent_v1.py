#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 08:19:53 2025

@author: rcyuh
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import random
import string
import json
import sys
from better_profanity import profanity

sys.path.append("./model_api")
from BaseAPI import BaseAPI


class MyAgent_v1(BaseAPI):
    def __init__(self, model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0", generation_config={}, aligner_paths=None, max_rewrites=3):
        super().__init__(generation_config)
        self.model_path = model_path
        self.generation_config = generation_config
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
            device_map="auto"
        ).eval()
        self.max_rewrites = max_rewrites
        
        self.sys_prompt = self.without_strict_jsonformat_sys_prompt

        # Thought-Aligners
        self.aligners = []
        if aligner_paths:
            for path in aligner_paths:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else "auto",
                    device_map="auto"
                ).eval()
                self.aligners.append((tokenizer, model))

        # Profanity filter
        profanity.load_censor_words()

    def reflect_thought(self, instruction, thought):
        return f"Instruction: {instruction}\nThought: {thought}\nDo you think this thought is safe and helpful?"

    def rewrite_thought(self, instruction, thought):
        current_thought = thought
        for i in range(self.max_rewrites):
            changed = False
            for tokenizer, model in self.aligners:
                prompt = (
                    f"USER: Edit the following Instruction-Thought pair to make it more helpful and harmless."
                    f"\nInstruction: {instruction}\nThought: {current_thought}\nASSISTANT:"
                )
                input_ids = tokenizer(prompt, return_tensors='pt').to(model.device)
                with torch.no_grad():
                    output_ids = model.generate(input_ids["input_ids"], max_new_tokens=256)[0]
                output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
                new_thought = output_text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in output_text else output_text.strip()

                if new_thought != current_thought:
                    current_thought = new_thought
                    changed = True

            if not changed:
                break
        return current_thought

    def rule_based_filter(self, thought):
        return profanity.contains_profanity(thought)

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

        # Get last user message as instruction
        instruction = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        prompt = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages if "content" in m]) + "\nASSISTANT:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512)
        raw_thought = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

        reflection = self.reflect_thought(instruction, raw_thought)
        print("[Reflection Prompt]:", reflection)

        aligned_thought = self.rewrite_thought(instruction, raw_thought)

        if self.rule_based_filter(aligned_thought):
            print("[Blocked] Unsafe content detected.")
            return {"type": "text", "content": "[BLOCKED DUE TO UNSAFE CONTENT]"}

        # Fake tool_call response
        if aligned_thought.startswith("CALL:"):
            try:
                content = aligned_thought[len("CALL:"):].strip()
                parts = json.loads(content)
                return {
                    "type": "tool",
                    "tool_call_id": tool_call_id or "random_id",
                    "tool_name": parts.get("tool_name", "unknown"),
                    "arguments": parts.get("arguments", {}),
                }
            except:
                return {"type": "text", "content": aligned_thought}

        return {"type": "text", "content": aligned_thought}
