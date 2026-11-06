"""
sytem prompt: You are a helpful and precise assistant in labeling the context dependency of the sentence.
--------------------
{prompt}

{definition}

{src_lang} Sentence: {sentence} #sentence-level


--------------------
We would like you to label the context dependency of the following sentence in the {domain} domain. You should classify the external knowledge needed to translate the sentence into sentence-level knowledge, local contextual knowledge, extended contextual knowledge, global contextual knowledge, and universal contextual knowledge. You must write only one class without any explanation.

Sentence-level knowledge: The sentence can be fully understood and translated without any outside information. All necessary meaning is present within the sentence itself — vocabulary, grammar, and semantics are straightforward.
Example: The cat is sleeping on the couch.

Local contextual knowledge: Understanding requires minimal surrounding context — maybe the previous or next sentence — but nothing broader. Without it, pronouns, references, or logical connectors might be confusing.
Example: "She picked it up carefully." (Needs to know who ‘she’ is and what ‘it’ is, but usually just from nearby sentences.)

Extended contextual knowledge: Grasping the meaning requires understanding the broader scene, paragraph, or emotional flow. Cultural nuance, emotional undertones, or evolving character perspectives start to matter.
Example: "He knew it was the only way to save them." (Without knowing the stakes or characters, the meaning could shift a lot.)

Global contextual knowledge: The sentence depends on knowledge of the entire work (novel, article, movie) or even multiple entries (book series, TV seasons). Important world-building, character arcs, fictional history, or long-term motifs influence meaning.
Example: "Winter is coming." (In A Song of Ice and Fire, it’s loaded with symbolic and political meaning; outside that, it just sounds like a weather report.)

Universal contextual knowledge: Understanding draws on extensive external knowledge — history, philosophy, science, mythology, social structures, or famous world events. Without that shared knowledge, translation risks misfiring badly.
Example: "Opening Pandora’s box." (Without knowing Greek mythology, this could be meaningless or totally misinterpreted.)
"""
import json
import sys
sys.path.append("..")
sys.path.append("../..")
import argparse
import pandas as pd

from call_api import get_api
from module.my_keys import OPENAI_KEYS



def gen_template(src_lang: str, domain: str):
    
    prompt = f"We would like you to label the context dependency of the following sentence in the {domain} domain. You should classify the external knowledge needed to translate the sentence into sentence-level knowledge, local contextual knowledge, extended contextual knowledge, global contextual knowledge, and universal contextual knowledge. When multiple criteria are present, select the broader context dependency and write only one class without explanation.".format(domain=domain)
    lv1 = "Sentence-level knowledge: The sentence can be fully understood and translated without any outside information. All necessary meaning is present within the sentence itself — vocabulary, grammar, and semantics are straightforward - leaving no room for lexical ambiguity." + "\n" + "Example: The cat is sleeping on the couch."
    lv2 = "Local contextual knowledge: Understanding requires minimal surrounding context — maybe the previous or next sentence — but nothing broader. Without it, pronouns, references, or logical connectors might be confusing." + "\n" + "Example: \"She picked it up carefully.\" (Needs to know who ‘she’ is and what ‘it’ is, but usually just from nearby sentences.)"
    lv3 = "Extended contextual knowledge: Grasping the meaning requires understanding the broader scene, paragraph, or emotional flow. Cultural nuance, emotional undertones, or evolving character perspectives start to matter." + "\n" + "Example: \"He knew it was the only way to save them.\" (Without knowing the stakes or characters, the meaning could shift a lot.)"
    lv4 = "Global contextual knowledge: The sentence depends on knowledge of the entire work (novel, article, movie) or even multiple entries (book series, TV seasons). Important world-building, character arcs, fictional history, or long-term motifs influence meaning." + "\n" + "Example: \"Winter is coming.\" (In A Song of Ice and Fire, it’s loaded with symbolic and political meaning; outside that, it just sounds like a weather report.)"
    lv5 = "Universal contextual knowledge: Understanding draws on extensive external knowledge — history, philosophy, science, mythology, social structures, or famous world events. Without that shared knowledge, translation risks misfiring badly." + "\n" + "Example: \"Opening Pandora’s box.\" (Without knowing Greek mythology, this could be meaningless or totally misinterpreted.)"
    definition = '\n'.join([lv1, lv2, lv3, lv4, lv5])
    
    template = "{prompt}\n\n{definition}\n\n{src_lang} sentence: {sentence}\n"
    return template.format(prompt=prompt, definition=definition, 
                           src_lang=src_lang, sentence="{sentence}")
    
    
def gen_messages(src_seg, src_lang, domain):
    system_prompt = "You are a helpful and precise assistant in labeling the context dependency of sentences."
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    
    # add prompt:
    template = gen_template(src_lang, domain)
    messages.append(
        {
            "role": "user",
            "content": template.format(sentence=src_seg)
        }
    )
    return messages


def gen_requests(reviewer, df):
    requests = []
    for i, row in df.iterrows():
        message = gen_messages(domain=row.domain,
                              src_seg=row.source, 
                              src_lang=row.src_lang
                             )
        # generate request:
        requests.append({
                #'review_id': shortuuid.uuid(),
                'id': i,
                'request': {
                    'model': reviewer['model'],
                    'messages': message,
                },
                'temperature': reviewer['temperature'],
                'max_tokens': reviewer['max_tokens'],
        })
        
    df['prompt'] = requests
    
    return df

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-file')
    parser.add_argument('-o', '--out-file')
    parser.add_argument('-e', '--error-file', help=".jsonl")
    parser.add_argument('-m', '--model', default='gpt-4.1-mini')
    parser.add_argument('-t', '--temperature', type=float, default=0.2)
    parser.add_argument('-x', '--max-tokens', type=int, default=1024)
    args = parser.parse_args()

    df = gen_requests(reviewer={
        'model': args.model,
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
    },
                      df=pd.read_csv(args.data_file),
                     )  
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