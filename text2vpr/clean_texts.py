import pandas as pd
import os
import numpy as np
from numpy import nan

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def clean_texts_from_csv(csv_file):    
    
    model_id = "microsoft/Phi-4-mini-instruct"

    # 1. Load Model and Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype="auto", 
        trust_remote_code=True,
        attn_implementation="flash_attention_2" # Optional: speeds up generation on supported GPUs
    )
    
    results = []
    # parse csv file
    df = pd.read_csv(csv_file)
    # go line by line and read columns
    for index, row in df.iterrows():
        image_path = row['image_path']
        description = row['description']    
        to_filter = row['manualy filter']
        if to_filter is nan:
            results.append([image_path, description])
        else:
            # call llm to remove to_filter text from description
            # 3. Format the Prompt
            messages = [
                {"role": "system", "content": "You are a precise editor. Your task is to rewrite paragraphs to remove specific topics while keeping the rest of the text natural and grammatically correct."},
                {"role": "user", "content": f"Rewrite the following paragraph to remove all mention of '{to_filter}':\n\n{description}"}
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 4. Generate the Response
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs, 
                max_new_tokens=2000, 
                do_sample=False                
            )
            
            # Slice the output: [0, len(inputs.input_ids[0]):]
            # This ignores the first 'n' tokens (the prompt)
            generated_tokens = outputs[0][len(inputs.input_ids[0]):]

            # Decode only the new tokens
            new_description = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            results.append([image_path, new_description, description])
    
    # save results to updated 
    csv_file_name = os.path.basename(csv_file)
    df2 = pd.DataFrame(results, columns=['image_path', 'description', 'original_description'])
    df2.to_csv(csv_file_name, index=False)
    
    

if __name__ == "__main__":
    csv_file = 'amstertime_objects.csv'
    clean_texts_from_csv(csv_file)

        