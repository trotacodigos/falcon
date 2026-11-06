"""
system prompt: You are a helpful and precise assistant that selects the necessary skills required to translate sentences.

{prompt1}

[Skill Options]
{skills}

{prompt2}

[{src_lang} Sentence]
{sentence}

{prompt3}

[System]
----------------------------
You are given the following 9 translation skills.

[Skill Options]

What are 3 core skills required to translate the following sentence into a coherent piece of discourse? Especially, select the primary skills uniquely required to translate into any languages within the {domain} domain, rather than skills that could be applied to ordinary sentences.

[{src_lang} Sentence]
{sentence}

Select and write the index of the 3 most primary skills. Also, write a brief description of how the skill should be applied when translating within 1~2 sentences for each selected skill. Finally, after generating two newlines, return a Python list object that includes each index of 3 skills, arranged in descending order of importance, from the most important to the least.

[System]

"""
import json
import random
import copy
import argparse
import pandas as pd
import sys
sys.path.append("..")
sys.path.append("../..")
from ast import literal_eval
import re
from collections import Counter

from call_api import get_api
from module.my_keys import OPENAI_KEYS

import logging
logging.basicConfig(level=logging.INFO)



def get_answer(df):
    """convert index into skill name"""
    df.skillset = df.skillset.apply(literal_eval)
    tmp = {}
    for i, row in df.iterrows():
        indices = parse_response(row.reviews)
        if indices is not None:
            skills = [row.skillset[j] for j in indices]
        else:
            logging.warning('error')
            skills = []
        tmp[i] = skills
        
    df['skill'] = tmp
    # print info:
    all_responses = [vv for v in tmp.values() for vv in v]
    stat = Counter(all_responses)
    logging.info(stat)
    
    return df
    

def parse_response(line: str):
    """get indices from the response"""
    match = re.search(r'```python\n(.*?)\n```', line, re.DOTALL)
    if match:
        code_content = match.group(1)
    else:
        # No code block?
        match = re.search(r'\[\s*([^\[\]]+)\s*\]', line)
        if match:
            code_content = match.group(0)
        else:
            code_content = None
    if code_content is not None:
        return literal_eval(code_content)


def gen_random_skills(skillset: dict) -> tuple:
    """shuffle the skills to avoid possible bias"""
    indices = list(skillset.keys()).copy()
    random.shuffle(indices)
    
    line = ""
    skill_dic = {}
    for i in indices:
        index = int(i) + 1
        criteria = skillset[i]['criteria']
        line += f"\n{index}: {criteria}"
        skill_dic[index] = skillset[i]['skill']

    return line, skill_dic
        
        

def gen_template(src_lang: str, domain: str, tgt_lang: str = None):
    prompt1 = "You are given the following 9 translation skills."
    prompt2 = f"What are 3 core skills required to translate the following sentence into a coherent piece of discourse? Especially, select the primary skills uniquely required to translate into [TGT] within the {domain} domain, rather than skills that could be applied to ordinary sentences.".format(domain=domain)
    
    # TO TEST 
    if tgt_lang is None:
        prompt2 = prompt2.replace("[TGT]", "any languages")
    else:
        prompt2 = prompt2.replace("[TGT]", tgt_lang)
    
    prompt3 = "Select and write the index of the 3 most primary skills. Also, write a brief description of how the skill should be applied when translating within 1~2 sentences for each selected skill. Finally, after generating two newlines, return a Python list object that includes each index of 3 skills, arranged in descending order of importance, from the most important to the least."
    #sentence = f"[{src_lang} Sentence]\n".format(src_lang=src_lang) + "{sentence}"
    template = ("{prompt1}\n\n[Skill Options]\n[SKILL]\n\n{prompt2}\n\n"
                "{sentence}\n\n{prompt3}\n\n[System]\n\n\n"
               )
    return template.format(prompt1=prompt1, prompt2=prompt2, prompt3=prompt3, 
                           sentence="{sentence}", src_lang=src_lang)

    
def gen_messages(src_lang, domain, tgt_lang, src_seg):
    system_prompt = "You are a helpful and precise assistant that selects the necessary skills required to translate sentences as coherent discourse."
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # add prompt:
    template = gen_template(src_lang, domain, tgt_lang)
    messages.append(
        {
            "role": "user",
            "content": template.format(sentence=src_seg)
        }
    )
    return messages


def gen_requests(reviewer: dict, df, skillset: dict):
    requests = []
    skills = []
    indices = []
    for i, row in df.iterrows():
        # context가 sent 아닌 열에 대해서만
        if row.context != "Sentence-level knowledge":
            indices.append(i)
            # generate message with random skills
            skill, skill_dic = gen_random_skills(skillset)
            message = gen_messages(domain=row.domain,
                                   src_seg=row.source, 
                                   src_lang=row.src_lang,
                                   tgt_lang=row.tgt_lang,
                                 )

            message[-1]['content'] = message[-1]['content'].replace('[SKILL]', skill)

            # generate request:
            requests.append({
                    #'review_id': shortuuid.uuid(),
                    'id': i,
                    'request': {
                        'model': reviewer['model'],
                        'messages': message,
                    },
                    'temperature': reviewer.get('temperature', 0.2),
                    'max_tokens': reviewer.get('max_tokens', 1024),
            })
            skills.append(skill_dic)
    
    df_ = df.iloc[indices]
    df_['prompt'] = requests
    df_['skillset'] = skills
    logging.info(f"Reduced the size from {len(df)} to {len(df_)}")
    
    return df_

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-file')
    parser.add_argument('-o', '--out-file')
    parser.add_argument('-s', '--skill-file', default="skillset.json")
    parser.add_argument('-e', '--error-file', help=".jsonl")
    parser.add_argument('-g', '--tgt-lang', action='store_false', help="Disable target language") #when specified
    parser.add_argument('-m', '--model', default='gpt-4.1-mini')
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-x', '--max-tokens', type=int, default=1024)
    args = parser.parse_args()
    
    logging.info(f'{args.tgt_lang=}')
    # Read file:
    with open(args.skill_file, 'r') as f:
        skillset = json.load(f)

    df = gen_requests(reviewer={
                        'model': args.model,
                        'temperature': args.temperature,
                        'max_tokens': args.max_tokens,
                    },
                      df=pd.read_csv(args.data_file), 
                      skillset=skillset)  
    responses = get_api(OPENAI_KEYS, df.prompt.tolist())
    
    if len(df) == len(responses):
        df["reviews"] = [res['content'] for res in responses]
        df["total_token"] = [res['usage'] for res in responses]
        df.to_csv(args.out_file, index=None)

    else:
        filename = '.'.join(args.error_file.split(".")[:-1]) + ".jsonl"
        logging.warning(f"Saving to error file: {filename}")
        
        with open(filename, 'w') as f:
            for line in filename:
                f.write(json.dumps(line) + '\n')
        raise Exception