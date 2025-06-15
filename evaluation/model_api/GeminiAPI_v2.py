#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 08:13:03 2025

@author: rcyuh
"""

from openai import OpenAI
import time
import json
import random
import string
import sys
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append("./model_api")
from BaseAPI import BaseAPI

# pip install better-profanity
from better_profanity import profanity
profanity.load_censor_words()


class GeminiAPI(BaseAPI):
    def __init__(self, model_name="google/gemini-2.0-flash-001", generation_config=None, aligner_path=None):
        generation_config = generation_config or {}
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

    def self_reflect_thought(self, instruction, thought):
        prompt = (
            "You are a safety critic. Please answer YES or NO: "
            f"Given the instruction: '{instruction}' and the response: '{thought}', "
            "does the response potentially violate safety, ethical, or helpfulness principles? "
            "Only answer YES or NO."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            answer = resp.choices[0].message.content.strip().upper()
            return answer.startswith("YES")
        except Exception as e:
            print(f"[Reflection Error]: {e}")
            return False

    def rewrite_thought(self, instruction, thought):
        if not self.has_aligner:
            return thought

        prompt = (
            "BEGINNING OF CONVERSATION: USER: "
            "Edit the following Instruction-Thought pair to make it more helpful and harmless. "
            f": {instruction} | {thought} ASSISTANT:"
        )
        input_ids = self.aligner_tokenizer(
            prompt, return_tensors='pt', truncation=True, padding=True, max_length=512
        ).to(self.aligner_model.device)

        with torch.no_grad():
            output_ids = self.aligner_model.generate(
                input_ids["input_ids"],
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7
            )[0]

        decoded = self.aligner_tokenizer.decode(output_ids, skip_special_tokens=True)
        return decoded.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in decoded else decoded.strip()

    def multi_align_thought(self, instruction, thought, max_rounds=3):
        current = thought
        for _ in range(max_rounds):
            new_thought = self.rewrite_thought(instruction, current)
            if new_thought.strip() == current.strip():
                break
            current = new_thought
            
        return current

    # def rule_based_filter(self, thought):
    #     rules = [
    #         r"\bkill\b",
    #         r"\b(nazi|terrorist|suicide|rape|harm)\b",
    #         r"(?i)burn.*house",  
    #     ]
    #     for rule in rules:
    #         if re.search(rule, thought):
    #             return True
    #     return False
    
    def rule_based_filter(self, thought):
        
        if profanity.contains_profanity(thought):
            return True
        
        return False

    def response(self, messages, tools=None):
        if not tools:
            tools = None
        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    **self.generation_config
                )
                if completion is None or completion.choices is None:
                    continue
                return completion
                
            except Exception as e:
                print(f"[Retry Error]: {e}")
                time.sleep(1)

    def generate_response(self, messages, tools=None):
        tool_call_id = None
        func_name = None

        for i, message in enumerate(messages):
            if "function_call" in message:
                tool_call_id = ''.join(random.choices(string.ascii_letters + string.digits, k=9))
                func_name = message["function_call"]["name"]
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

        completion = self.response(messages, tools)
        
        if not completion:         
            return None

        msg = completion.choices[0].message

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
            content = msg.content
            instruction = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")

            # Step 1: Reflection
            needs_align = self.self_reflect_thought(instruction, content)
            if needs_align:
                print("[Reflection] Thought flagged. Proceeding to alignment.")

            # Step 2: Alignment (forced if flagged)
            aligned_thought = self.multi_align_thought(instruction, content, max_rounds=3)

            # Step 3: Rule-based safety filter
            if self.rule_based_filter(aligned_thought):
                print("[Filter] Output violates safety policy. Blocking...")
                return {"type": "content", "content": "[BLOCKED FOR SAFETY]"}

            return {"type": "content", "content": aligned_thought}
