from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ACTUAL_TOKEN =  os.getenv("HF_AUTH_TOKEN")#User token name

class DialogueManager:
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            use_auth_token=ACTUAL_TOKEN,
            trust_remote_code=True,
            device_map="auto",
            load_in_4bit=True,  # Quantization if needed
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            torch_dtype=torch.bfloat16,)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=ACTUAL_TOKEN,
            trust_remote_code=True,
            use_fast=False,
        )
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model.to(device)
        self.device = device
    
    def build_prompt(self, conversation_history):
        prompt = (
        "You are a conversational assistant that extracts user intent and details from the conversation. "
        "Given the conversation history, output a JSON with the following keys:\n"
        "  • intent\n"
        "  • location\n"
        "  • cuisine  (in singular form, e.g. 'taco' not 'tacos')\n"
        "  • price_range\n"
        "  • recent_review_requested  (True/False)\n"
        "  • other_info\n\n"
         "Conversation:\n" + conversation_history + "\n\nOutput JSON:"
         )
        return prompt
    
    def process_conversation(self, conversation_history, max_new_tokens=150):
        prompt = self.build_prompt(conversation_history)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample=True,
            temperature=0.8,
            pad_token_id = self.tokenizer.eos_token_id
        )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_start = generated_text.find("{")
        json_text = generated_text[json_start:]
        try:
            extracted_data = json.loads(json_text)
        except json.JSONDecodeError:
            extracted_data = {"error": "Could not Parse JSON"}
        return extracted_data
    
# if __name__ == "__main__":
#     dm = DialogueManager()
#     conversation = (
#         "User: I'm looking for a good Chinese restaurant in downtown.\n"
#         "Assistant: Sure, could you tell me if you have any specific price range in mind?\n"
#         "User: Something decent, not too expensive."
#     )
#     result = dm.process_conversation(conversation)
#     print("Extracted structured query: ", result)
        
        
