import openai
import random
import time
import copy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential
)

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.WARNING)


def get_api(api_keys: list, requests):
    """main"""
    num_keys = len(api_keys)
    params = calc_default_params(num_keys)
    num_workers = params['num_workers']
    time_btw_request = params['time_btw_request']

    logging.info(f"{num_workers=}, {time_btw_request=}")
    
    responses = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for index, request in tqdm(enumerate(requests),
                                   total=len(requests),
                                   desc="Call_API",
                                   unit="request",
                                   ncols=100,
                                   colour="yellow",
                                  ):
            api_key = api_keys[index % num_keys]
            
            future = executor.submit(call, api_key=api_key, request=request)
            
            # check response:
            response = verify_response(future)
            if response['content'] is None:
                new_request = copy.deepcopy(request)
                new_request['temperature'] = request['temperature'] + 1
                logging.warning(f"Increasing temperature to {new_request['temperature']}>>")
                future = executor.submit(call, api_key=api_key, request=new_request)
                response = verify_response(future)
                
            elif response['content'] == "":
                new_request = copy.deepcopy(request)
                new_request['max_tokens'] = request['max_tokens'] + 200
                logging.warning(f"Increasing `max_tokens` to {new_request['max_tokens']}>>")
                future = executor.submit(call, api_key=api_key, request=new_request)
                response = verify_response(future)
                
            responses.append(response)
            time.sleep(time_btw_request)

    return responses

        
def verify_response(future: object) -> dict:
    """Check response if it is empty
        Return `content` instead of `response`
    """
    response = future.result()
    content = ""
    if 'choices' in vars(response):
        choice = response.choices[0]
        if (choice.message.content is None) | (choice.finish_reason != 'stop'):
            content = ""
        else:
            content = choice.message.content.strip()
    else:
        logging.warning(f'response failed: {response}')
        return {'content': None, 'usage': None} 
    
    return {'content': content, 'usage': response.usage.total_tokens}
        
        
def calc_default_params(num_api_keys: int):
    request_per_minute = 60
    response_second = 5.0
    buffer = 2
    
    requst_per_second = request_per_minute / 60
    sim_num_requests = requst_per_second * response_second
    num_workers = int(sim_num_requests * buffer) 
    total_request_per_second = requst_per_second * num_api_keys
    time_btw_request = 1 / total_request_per_second

    return {
        "num_workers": num_workers, "time_btw_request": time_btw_request,
    }
        
        
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), reraise=True)        
def completion_with_backoff(api_key, **kwargs):
    """create the client"""
    client = openai.OpenAI(api_key=api_key)
    completion = client.chat.completions.create(**kwargs)
    return completion


def call(api_key, request) -> tuple:
    try:
        response = completion_with_backoff(api_key, **request['request'])
        error = None
    except Exception as e:
        response = None
        error = repr(error)
        logging.warning(f"response is None: {error}")
        
    if error is None:
        return response
    else:
        return {}