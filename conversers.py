import common
from language_models import GPT, Claude, SmolLM, Llama2, Gemma2, Codellama, HuggingFace, Opencoder, Granite3_Guardian, Solar_Pro, Nemotron_Mini, Hermes3, GLM4, Deepseek_V2, Wizardlm2, All_MiniLM, Nomic_Embed_Text, Vicuna, Everythinglm, Tulu3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P   
import asyncio

def load_attack_and_target_models(args):
    # Load attack model and tokenizer
    attackLM = AttackLM(model_name = args.attack_model, 
                        max_n_tokens = args.attack_max_n_tokens, 
                        max_n_attack_attempts = args.max_n_attack_attempts, 
                        temperature = ATTACK_TEMP, # init to 1
                        top_p = ATTACK_TOP_P, # init to 0.9
                        )
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attackLM.model
    targetLM = TargetLM(model_name = args.target_model, 
                        max_n_tokens = args.target_max_n_tokens,
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        preloaded_model = preloaded_model,
                        )
    return attackLM, targetLM

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                temperature: float,
                top_p: float):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        
        # if "vicuna" in model_name or "llama" in model_name:
        #     self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        # Initalize the attack model's generated output to match format
        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \""""
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        # Add prompts and initial seeding messages to conversations (only once)
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            # Get prompts
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                prompt = conv.get_prompt()
                if conv.sep2 is not None:
                    prompt = prompt[:-len(conv.sep2)]
                full_prompts.append(prompt)
            
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]

            # Generate outputs 
            outputs_list = self.model.batched_generate(full_prompts_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)  # Update the conversation with valid generation
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break
        
        if any([output for output in valid_outputs if output is None]):
            print(f"Failed to generate output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs

class TargetLM():
    """
        Base class for target language models.
        
        Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            max_n_tokens: int, 
            temperature: float,
            top_p: float,
            preloaded_model: object = None):
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                # Openai does not have separators
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None) 
                full_prompts.append(conv.get_prompt())
        
        # Check if model is SmolLM and use its specific methods

        if isinstance(self.model, SmolLM) or isinstance(self.model, Llama2) or isinstance(self.model, Gemma2) or isinstance(self.model, Codellama) or isinstance(self.model, Opencoder) or isinstance(self.model, Granite3_Guardian) or isinstance(self.model, Solar_Pro) or isinstance(self.model, Nemotron_Mini) or isinstance(self.model, Hermes3) or isinstance(self.model, GLM4) or isinstance(self.model, Deepseek_V2) or isinstance(self.model, Wizardlm2) or isinstance(self.model, All_MiniLM) or isinstance(self.model, Nomic_Embed_Text) or isinstance(self.model, Tulu3) or isinstance(self.model, Everythinglm) or isinstance(self.model, Vicuna):
            outputs_list = [self.model.sync_generate_single(prompt) for prompt in full_prompts]

        else:
            outputs_list = self.model.batched_generate(full_prompts, 
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p
                                                    )
        return outputs_list



def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    if model_name == "smollm":
        lm = SmolLM(model_name=model_name)
    elif model_name == "llama2":
        lm = Llama2(model_name=model_name)
    elif model_name == "gemma2":
        lm = Gemma2(model_name=model_name)
    elif model_name == "codellama":
        lm = Codellama(model_name=model_name)
    elif model_name == "opencoder":
        lm = Opencoder(model_name=model_name)
    elif model_name == "granite3-guardian":
        lm = Granite3_Guardian(model_name=model_name)
    elif model_name == "solar-pro":
        lm = Solar_Pro(model_name=model_name)
    elif model_name == "nemotron-mini":
        lm = Nemotron_Mini(model_name=model_name)
    elif model_name == "hermes3":
        lm = Hermes3(model_name=model_name)
    elif model_name == "glm4":
        lm = GLM4(model_name=model_name)
    elif model_name == "deepseek-v2":
        lm = Deepseek_V2(model_name=model_name)
    elif model_name == "wizardlm2":
        lm = Wizardlm2(model_name=model_name)
    elif model_name == "all-minilm":
        lm = All_MiniLM(model_name=model_name)
    elif model_name == "nomic-embed-text":
        lm = Nomic_Embed_Text(model_name=model_name)
    elif model_name == "tulu3":
        lm = Wizardlm2(model_name=model_name)
    elif model_name == "everythinglm":
        lm = All_MiniLM(model_name=model_name)
    elif model_name == "vicuna":
        lm = Nomic_Embed_Text(model_name=model_name)
    elif model_name in ["gpt-3.5-turbo", "gpt-4"]:
        lm = GPT(model_name)
    elif model_name in ["claude-2", "claude-instant-1"]:
        lm = Claude(model_name)
    # elif model_name in ["palm-2"]:
    #     lm = PaLM(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,device_map="auto").eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False
        ) 

        # if 'llama-2' in model_path.lower():
        #     tokenizer.pad_token = tokenizer.unk_token
        #     tokenizer.padding_side = 'left'
        # if 'vicuna' in model_path.lower():
        #     tokenizer.pad_token = tokenizer.eos_token
        #     tokenizer.padding_side = 'left'
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict={
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        # "vicuna":{
        #     "path":VICUNA_PATH,
        #     "template":"vicuna_v1.1"
        # },
        # "llama-2":{
        #     "path":LLAMA_PATH,
        #     "template":"llama-2"
        # },
        "claude-3-haiku":{
            "path":"claude-3-haiku",
            "template":"claude-3-haiku"
        },
        "claude-3-sonnet":{
            "path":"claude-3-sonnet",
            "template":"claude-3-sonnet"
        },
        "smollm":{
            "path":"smollm",
            "template":"smollm"
        },
        "llama2":{
            "path":"llama2",
            "template":"llama2"
        },
        "gemma2":{
            "path":"gemma2",
            "template":"gemma2"
        },
        "codellama":{
            "path":"codellama",
            "template":"codellama"
        },
        "opencoder":{
            "path":"opencoder",
            "template":"opencoder"
        },
        "granite3-guardian":{
            "path":"granite3-guardian",
            "template":"granite3-guardian"
        },
        "solar-pro":{
            "path":"solar-pro",
            "template":"solar-pro"
        },
        "nemotron-mini":{
            "path":"nemotron-mini",
            "template":"nemotron-mini"
        },
        "reflection":{
            "path":"reflection",
            "template":"reflection"
        },
        "hermes3":{
            "path":"hermes3",
            "template":"hermes3"
        },
        "glm4":{
            "path":"glm4",
            "template":"glm4"
        },
        "deepseek-v2":{
            "path":"deepseek-v2",
            "template":"deepseek-v2"
        },
        "wizardlm2":{
            "path":"wizardlm2",
            "template":"wizardlm2"
        },
        "all-minilm":{
            "path":"all-minilm",
            "template":"all-minilm"
        },
        "nomic-embed-text":{
            "path":"nomic-embed-text",
            "template":"nomic-embed-text"
        },
        "tulu3":{
            "path":"tulu3",
            "template":"tulu3"
        },
        "everythinglm":{
            "path":"everythinglm",
            "template":"everythinglm"
        },
        "vicuna":{
            "path":"vicuna",
            "template":"vicuna"
        }
        # "palm-2":{
        #     "path":"palm-2",
        #     "template":"palm-2"
        # }
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template   