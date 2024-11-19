import openai
import anthropic
import os
import time
import torch
import gc
from typing import Dict, List
import google.generativeai as palm
import ollama
import asyncio
from ollama import AsyncClient

from openai import OpenAI
from openai.types.chat import ChatCompletion
from openai import OpenAIError

# async def chat():
#   message = {'role': 'user', 'content': 'Why is the sky blue?'}
#   async for part in await AsyncClient().chat(model='smollm', messages=[message], stream=True):
#     print(part['message']['content'], end='', flush=True)

# asyncio.run(chat())

# response = ollama.chat(model='smollm', messages=[
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ])
# print(response['message']['content'])

# stream = ollama.chat(
#     model='smollm',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )

# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)


class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name
    
    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError



# class LanguageModel:
#     def __init__(self, model_name):
#         self.model_name = model_name
    
#     def batched_generate(self, prompts_list: List[str], max_n_tokens: int, temperature: float):
#         """
#         Generates responses for a batch of prompts using a language model.
#         """
#         raise NotImplementedError

class SmolLM(LanguageModel):
    def __init__(self, model_name='smollm'):
        super().__init__(model_name)
        self.client = AsyncClient()

    async def async_generate_single(self, prompt):
        """
        Asynchronously generates a response for a single prompt.
        """
        message = {'role': 'user', 'content': prompt}
        response_text = ""
        
        # Asynchronous streaming chat response
        async for part in await self.client.chat(model=self.model_name, messages=[message], stream=True):
            response_text += part['message']['content']
        
        return response_text

    def sync_generate_single(self, prompt, stream=False):
        """
        Synchronously generates a response for a single prompt.
        """
        message = {'role': 'user', 'content': prompt}
        
        if stream:
            # Streaming synchronous chat response
            response_text = ""
            stream_response = ollama.chat(model=self.model_name, messages=[message], stream=True)
            for chunk in stream_response:
                response_text += chunk['message']['content']
            return response_text
        else:
            # Single response for synchronous chat
            response = ollama.chat(model=self.model_name, messages=[message])
            return response['message']['content']

    def batched_generate(self, prompts_list: List[str], max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using synchronous or asynchronous chat.
        """
        # Collect all responses
        responses = [self.sync_generate_single(prompt) for prompt in prompts_list]
        return responses

    def chat(self, prompt, async_mode=False, stream=False):
        """
        Handles synchronous or asynchronous chat for a single prompt.
        """
        if async_mode:
            return asyncio.run(self.async_generate_single(prompt))
        else:
            return self.sync_generate_single(prompt, stream=stream)


class HuggingFace(LanguageModel):
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        self.eos_token_ids = [self.tokenizer.eos_token_id]

    def batched_generate(self, 
                        full_prompts_list,
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
    
        # Batch generation
        if temperature > 0:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=True,
                temperature=temperature,
                eos_token_id=self.eos_token_ids,
                top_p=top_p,
            )
        else:
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens, 
                do_sample=False,
                eos_token_id=self.eos_token_ids,
                top_p=1,
                temperature=1, # To prevent warning messages
            )
            
        # If the model is not an encoder-decoder type, slice off the input tokens
        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, inputs["input_ids"].shape[1]:]

        # Batch decoding
        outputs_list = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for key in inputs:
            inputs[key].to('cpu')
        output_ids.to('cpu')
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()

        return outputs_list

    def extend_eos_tokens(self):        
        # Add closing braces for Vicuna/Llama eos when using attacker model
        self.eos_token_ids.extend([
            self.tokenizer.encode("}")[1],
            29913, 
            9092,
            16675])

# class GPT(LanguageModel):
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 0.5
#     API_MAX_RETRY = 5
#     API_TIMEOUT = 20
#     openai.api_key = os.getenv("OPENAI_API_KEY")

#     def generate(self, conv: List[Dict], 
#                 max_n_tokens: int, 
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries, OpenAI API format
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#         for _ in range(self.API_MAX_RETRY):
#             try:
#                 response = openai.chat.completions.create(
#                             model = self.model_name,
#                             messages = conv,
#                             max_tokens = max_n_tokens,
#                             temperature = temperature,
#                             top_p = top_p,
#                             #request_timeout = self.API_TIMEOUT,
#                             )
#                 output = response["choices"][0]["message"]["content"]
#                 break
#             except openai.error.OpenAIError as e:
#                 print(type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
        
#             time.sleep(self.API_QUERY_SLEEP)
#         return output 
    
#     def batched_generate(self, 
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int, 
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20


    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def generate(self, conv: List[Dict], 
                 max_n_tokens: int, 
                 temperature: float,
                 top_p: float) -> str:
        '''
        Args:
            conv: List of dictionaries, OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response: ChatCompletion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=self.API_TIMEOUT
                )
                output = response.choices[0].message.content
                break
            except OpenAIError as e:
                print(f"OpenAI API Error: {type(e).__name__} - {str(e)}")
                time.sleep(self.API_RETRY_SLEEP)
            except Exception as e:
                print(f"Unexpected error: {type(e).__name__} - {str(e)}")
                time.sleep(self.API_RETRY_SLEEP)
            
            time.sleep(self.API_QUERY_SLEEP)
        return output 
    
    def batched_generate(self, 
                         convs_list: List[List[Dict]],
                         max_n_tokens: int, 
                         temperature: float,
                         top_p: float = 1.0) -> List[str]:
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]

class Claude():
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_KEY = os.getenv("ANTHROPIC_API_KEY")
   
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.model= anthropic.Anthropic(
            api_key=self.API_KEY,
            )

    def generate(self, conv: List, 
                max_n_tokens: int, 
                temperature: float,
                top_p: float):
        '''
        Args:
            conv: List of conversations 
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        '''
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                completion = self.model.completions.create(
                    model=self.model_name,
                    max_tokens_to_sample=max_n_tokens,
                    prompt=conv,
                    temperature=temperature,
                    top_p=top_p
                )
                output = completion.completion
                break
            except anthropic.APIError as e:
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)
        
            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self, 
                        convs_list: List[List[Dict]],
                        max_n_tokens: int, 
                        temperature: float,
                        top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
        
# class PaLM():
#     API_RETRY_SLEEP = 10
#     API_ERROR_OUTPUT = "$ERROR$"
#     API_QUERY_SLEEP = 1
#     API_MAX_RETRY = 5
#     API_TIMEOUT = 20
#     default_output = "I'm sorry, but I cannot assist with that request."
#     API_KEY = os.getenv("PALM_API_KEY")

#     def __init__(self, model_name) -> None:
#         self.model_name = model_name
#         palm.configure(api_key=self.API_KEY)

#     def generate(self, conv: List, 
#                 max_n_tokens: int, 
#                 temperature: float,
#                 top_p: float):
#         '''
#         Args:
#             conv: List of dictionaries, 
#             max_n_tokens: int, max number of tokens to generate
#             temperature: float, temperature for sampling
#             top_p: float, top p for sampling
#         Returns:
#             str: generated response
#         '''
#         output = self.API_ERROR_OUTPUT
#         for _ in range(self.API_MAX_RETRY):
#             try:
#                 completion = palm.chat(
#                     messages=conv,
#                     temperature=temperature,
#                     top_p=top_p
#                 )
#                 output = completion.last
                
#                 if output is None:
#                     # If PaLM refuses to output and returns None, we replace it with a default output
#                     output = self.default_output
#                 else:
#                     # Use this approximation since PaLM does not allow
#                     # to specify max_tokens. Each token is approximately 4 characters.
#                     output = output[:(max_n_tokens*4)]
#                 break
#             except Exception as e:
#                 print(type(e), e)
#                 time.sleep(self.API_RETRY_SLEEP)
        
#             time.sleep(1)
#         return output
    
#     def batched_generate(self, 
#                         convs_list: List[List[Dict]],
#                         max_n_tokens: int, 
#                         temperature: float,
#                         top_p: float = 1.0,):
#         return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]