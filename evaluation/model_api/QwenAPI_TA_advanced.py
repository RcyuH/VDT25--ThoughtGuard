#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:13:24 2025

@author: rcyuh
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import random
import string
import os
import sys
sys.path.append('./model_api')
from BaseAPI import BaseAPI


class QwenAPI(BaseAPI):
    def __init__(self, model_path, tokenizer_path, generation_config={}):
        super().__init__(generation_config)
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = torch.device('cuda')
        print('loading model...')
        # if 'temperature' not in self.generation_config:
        #     self.generation_config['temperature'] = 1.0
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).to(self.device).eval()
        self.tokenizer.padding_side = 'left'
        print('finish loading')

        self.sys_prompt = self.basic_sys_prompt
        # self.sys_prompt = self.sys_prompt_with_failure_modes
        # self.sys_prompt = self.sys_prompt_with_simple_failure_modes
        
        
    
    
    def multi_align_thought(self, instruction, thought, max_rounds=3):
        current = thought
        for _ in range(max_rounds):
            new_thought = self.rewrite_thought(instruction, current)
            if new_thought.strip() == current.strip():
                break
            current = new_thought
            
        return current

    def rule_based_filter(self, thought):
        
        # pip install better-profanity
        from better_profanity import profanity
        profanity.load_censor_words()
        
        if profanity.contains_profanity(thought):
            return True
        
        return False
        
    def response(self, messages, tools):
        inputs = self.tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, **self.generation_config)
        return self.tokenizer.decode(out[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)

    def generate_response(self, messages, tools):
        completion = self.response(messages, tools)
        
        instruction = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        
        completion = self.multi_align_thought(instruction, completion, max_rounds=3)

        if self.rule_based_filter(completion):
            print("[Filter] Output violates safety policy. Blocking...")
            return {"type": "content", "content": "[BLOCKED FOR SAFETY]"}

        ## tool call part
        print(f'completion: {completion}')
        if '<tool_call>' in completion:
            # completion = completion.replace('<tool_call>', '').replace('</tool_call>', '').strip()
            completion = completion[:completion.find('</tool_call>')].replace('<tool_call>', '').strip()
            start = completion.index('{')
            # print(f'completion: {completion}')
            end = completion.rindex('}')
            completion = completion[start:end + 1]
            completion = completion.replace('\n', '\\n')
            # print(f'completion after find: {completion}')
            if self.is_json(completion):
                # print(f'is json: {completion}')
                res = self.parse_json(completion)
                if 'name' not in res or 'arguments' not in res:
                    print(f"Wrong tool call result: {res}")
                    return {'type': 'error', 'message': f'Wrong tool call result: {res}'}
                tool_call_id = ''.join(random.sample(string.ascii_letters + string.digits, 9))
                tool_name = res['name']
                if isinstance(res['arguments'], dict):
                    arguments = res['arguments']
                    return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments}
                elif self.is_json(res['arguments']):
                    arguments = self.parse_json(res['arguments'])
                    return {'type': 'tool', 'tool_call_id': tool_call_id, 'tool_name': tool_name, 'arguments': arguments}
                else:
                    print(f"Wrong argument format: {res['arguments']}")
                    return {'type': 'error', 'message': f"Wrong argument format: {res['arguments']}"}
            else:
                print(f"Wrong tool call completion result: {completion}")
                return {'type': 'error', 'message': f'Wrong tool call result: {completion}'}

        ## normal content part
        else:
            return {'type': 'content', 'content': completion}

