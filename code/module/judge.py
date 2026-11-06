import sys
sys.path.append("../..")
from call_api import get_api
from module.my_keys import OPENAI_KEYS
import json
import logging
logging.basicConfig(level=logging.INFO)
from ast import literal_eval
import argparse
import pandas as pd
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import math
from ast import literal_eval


def gen_template(src_lang, tgt_lang, domain, **kwargs):
    prompt1 = "We would like to request your feedback on the discourse-level quality of the translation within the {domain} domain. In your feedback, please rate the current translation enclosed with backticks in the following 3 categories, by referring to the preceding segments and following each scoring rubric.".format(domain=domain)
    prompt2 = "Please give feedback on the translation. Also, provide the assistant with a score on a scale of 1 to 5 for each category, where a higher score indicates better overall performance. Make sure to give feedback or comments for each category first, followed by the corresponding score for each category. Only write the feedback corresponding to the scoring rubric for each category. The scores for each category should be orthogonal, indicating that 'register' should not be considered for 'modality' category, for example."
    #prompt3 = "Provide detailed feedback on the translation, and assign a score from 1 to 5 for each category, where a higher score indicates better performance. Make sure to provide comments for each category first, followed by the corresponding score. Only include feedback that aligns with the scoring rubric for each category. Ensure that the scores for each category are independent of one another; for example, aspects of ‘consistency’ should not influence the score for the ‘fluency’ category."
    template = ("{prompt1}\n[Score rubric]\n{skills}\n\n[Preceding segments]\n"
                "{src_lang} source: {prev_src}\n"
                "{tgt_lang} target: {prev_tgt}\n\n"
                "[Current segments]\n{src_lang} source: {src_seg}\n"
                #"{tgt_lang} reference: {reference}\n"
                "{tgt_lang} translation: ```{tgt_seg}```\n"
                "{prompt2}\n\n[System]\n\n\n"
               )
    return template.format(prompt1=prompt1, prompt2=prompt2, src_lang=src_lang, tgt_lang=tgt_lang, **kwargs)


def gen_messages(src_lang, tgt_lang, domain, **kwargs):
    system_prompt = "You are a helpful and precise assistant in assessing the quality of the translation."
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # add prompt:
    messages.append(
        {
            "role": "user",
            "content": gen_template(src_lang, tgt_lang, domain, **kwargs)
        }
    )
    
    return messages


def gen_requests(reviewer, df, skillset: dict):
    assert 'context' in df.columns, "Data is not parsed!"
    
    requests = []
    indices = []
    for i, row in df.iterrows():
        if row.context != "Sentence-level knowledge":
            indices.append(i)
            # get skills rubric
            skills = get_rubric(skillset, row.skill)

            message = gen_messages(src_lang=row.src_lang,
                                  tgt_lang=row.tgt_lang,
                                  domain=row.domain,
                                  src_seg=row.source,
                                  tgt_seg=row.translation,
                                   #reference=row.reference,
                                  prev_src=row.prev_src,
                                  prev_tgt=row.prev_tgt,
                                  skills=skills)
            requests.append(
                {
                    'id': i,
                    'request': {
                        'model': reviewer.get('model', 'gpt-3.5-turbo'),
                        'messages': message,
                    },
                    'temperature': reviewer.get('temperature', 0),
                    'max_tokens': reviewer.get('max_tokens', 1024)
                }
            )
    df_ = df.iloc[indices]
    df_['prompt'] = requests
    logging.info(f"Reduced the size from {len(df)} to {len(df_)}")
    
    return df_
        
        
def get_rubric(skillset: dict, criteria: list) -> str:
    # return a line of definition and score rubric
    if not isinstance(criteria, list):
        #criteria = literal_eval(criteria)
        #criteria = criteria.split("/")
        criteria = literal_eval(criteria)
    assert len(criteria) == 3, f"Wrong number of skillsets: {criteria}"
    
    skilldic = {}
    for k, v in skillset.items():
        tmp = {}
        tmp['criteria'] = v['criteria']
        tmp['scoring'] = v['scoring']
        skilldic[v['skill']] = tmp
    
    line = ""
    for category in criteria:
        dic = skilldic.get(category)
        line += category + dic["criteria"] + "\n" + dic["scoring"] + "\n\n"
    return line
            
    
def batch_inference(df, batch_size=1000):
    for i in tqdm(range(0, len(df), batch_size),
                 desc="Batching",
                 unit="batch",
                 ncols=100,
                 colour="blue",
                 total=math.ceil(len(df) / batch_size)):
        batch = df.iloc[i: i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(df) - 1) // batch_size + 1}")
        yield batch 
        
        
def main(data_file, out_file: str, skillset: dict, error_file: str, model, temperature, max_tokens):   
    # for batch
    if isinstance(data_file, str):
        df = pd.read_csv(data_file)
    else:
        df = data_file.reset_index(drop=True) # 인덱스때문에 오류남
    
    frame = gen_requests(reviewer={
                        'model': model,
                        'temperature': temperature,
                        'max_tokens': max_tokens,
                      },
                      df=df,
                      skillset=skillset)  
    responses = get_api(OPENAI_KEYS, frame.prompt.tolist())

    if len(frame) == len(responses):
        frame["reviews"] = [res['content'] for res in responses]
        frame["total_token"] = [res['usage'] for res in responses]
        frame.to_csv(out_file, index=None)
        
    else:
        print(f"Saving to error file: {error_file}")

        with open(error_file, 'w') as f:
            for line in responses:
                f.write(json.dumps(line) + '\n')
        raise Exception

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-file')
    parser.add_argument('-o', '--out-file')
    parser.add_argument('-s', '--skill-file', default="skillset.json")
    parser.add_argument('-e', '--error-file', help=".jsonl")
    parser.add_argument('-m', '--model', default='gpt-4.1-mini')
    parser.add_argument('-t', '--temperature', type=float, default=0)
    parser.add_argument('-x', '--max-tokens', type=int, default=1024)
    parser.add_argument('-b', '--batch', action='store_true') # no flag == false
    args = parser.parse_args()
    
    with open(args.skill_file, 'r') as f:
        skillset = json.load(f)
    
    if args.batch:
        df = pd.read_csv(args.data_file)
        
        files = []
        for i, batch_df in enumerate(batch_inference(df)):
            out_file = f"{args.out_file[:-4]}_{i}.csv"
            
            main(data_file=batch_df, 
                 out_file=out_file, 
                 skillset=skillset,
                 error_file=args.error_file,
                 model=args.model,
                 temperature=args.temperature,
                 max_tokens=args.max_tokens)
            
            files.append(out_file)
        # after the process, gather all files into one:
        if len(files) > 1:
            final = pd.DataFrame()
            for file in files:
                tmp = pd.read_csv(file)
                final = pd.concat([final, tmp], axis=0)
            final.to_csv(out_file, index=None)
            print("Final file saved to: ", out_file)
        else:
            print("There are only one batch!")
        print('\n==The End==\n')
        
    # normal inference:
    else:
        main(data_file=args.data_file, 
             out_file=args.out_file, 
             skillset=skillset,
             error_file=args.error_file,
             model=args.model,
             temperature=args.temperature,
             max_tokens=args.max_tokens)