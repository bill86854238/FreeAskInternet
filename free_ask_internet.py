# -*- coding: utf-8 -*-

import json
import os 
from pprint import pprint
import requests
import trafilatura
from trafilatura import bare_extraction
from concurrent.futures import ThreadPoolExecutor
import concurrent
import requests
import openai
import time 
from datetime import datetime
from urllib.parse import urlparse
import tldextract
import platform
import urllib.parse

# Config
LLM_HISTORY_LIMIT = int(os.environ.get("LLM_HISTORY_LIMIT", 5))

def extract_url_content(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        content =  trafilatura.extract(downloaded)
        return {"url":url, "content":content}
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
        return {"url":url, "content":""}


def search_web_ref(query:str, debug=False):
 
    content_list = []

    try:
        # Using searxng service
        safe_string = urllib.parse.quote_plus(":all !general " + query)

        # Check if running in docker or local to adjust URL if needed, 
        # but sticking to original code's assumption of 'searxng' hostname 
        # or localhost if mapped. The original code used http://searxng:8080.
        # If running locally without docker network, this might fail unless mapped.
        # Assuming environment is set up as per original code.
        searx_url = os.environ.get("SEARXNG_URL", "http://searxng:8080")
        if "localhost" in searx_url or "127.0.0.1" in searx_url:
             pass # URL is likely fine
        
        response = requests.get(f'{searx_url}?q=' + safe_string + '&format=json')
        response.raise_for_status()
        search_results = response.json()
 
        if debug:
            print("JSON Response:")
            pprint(search_results)
        pedding_urls = []

        conv_links = []

        if search_results.get('results'):
            for item in search_results.get('results')[0:9]:
                name = item.get('title')
                snippet = item.get('content')
                url = item.get('url')
                pedding_urls.append(url)

                icon_url = ""
                site_name = ""
                if url:
                    try:
                        url_parsed = urlparse(url)
                        # domain = url_parsed.netloc
                        icon_url =  url_parsed.scheme + '://' + url_parsed.netloc + '/favicon.ico'
                        site_name = tldextract.extract(url).domain
                    except:
                        pass
 
                conv_links.append({
                    'site_name':site_name,
                    'icon_url':icon_url,
                    'title':name,
                    'url':url,
                    'snippet':snippet
                })

            results = []
            futures = []

            executor = ThreadPoolExecutor(max_workers=10) 
            for url in pedding_urls:
                futures.append(executor.submit(extract_url_content,url))
            try:
                for future in futures:
                    res = future.result(timeout=5)
                    results.append(res)
            except concurrent.futures.TimeoutError:
                print("Task timeout")
                executor.shutdown(wait=False,cancel_futures=True)

            for content in results:
                if content and content.get('content'):
                    
                    item_dict = {
                        "url":content.get('url'),
                        "content": content.get('content'),
                        "length":len(content.get('content'))
                    }
                    content_list.append(item_dict)
                if debug:
                    print("URL: {}".format(url))
                    print("=================")
 
        return  conv_links,content_list
    except Exception as ex:
        print(f"Search error: {ex}")
        return [], []


def gen_prompt(question, content_list, history=None, lang="zh-CN", context_length_limit=11000, debug=False):
    """
    Generates the prompt messages including system instructions, history, and current context.
    Returns a list of messages format: [{"role": "...", "content": "..."}]
    """
    if history is None:
        history = []
    
    # 1. Determine language
    answer_language = ' Simplified Chinese '
    if lang == "zh-CN":
        answer_language = ' Simplified Chinese '
    elif lang == "zh-TW":
        answer_language = ' Traditional Chinese '
    elif lang == "en-US":
        answer_language = ' English '

    # 2. Build Context String
    ref_text_list = []
    if content_list:
        ref_index = 1
        for item in content_list:
            content = item.get("content", "")
            if content:
                ref_text_list.append(f"[citation:{ref_index}] {content}")
                ref_index += 1
    
    context_str = "\n\n".join(ref_text_list)
    
    # Truncate context if too long (rough estimation)
    # Reserve tokens for history and system prompt
    reserved_limit = 2000 
    limit_len = max(0, context_length_limit - reserved_limit)
    if len(context_str) > limit_len:
        context_str = context_str[:limit_len] + "... (truncated)"

    # 3. System Prompt
    system_prompt = f"""You are a helpful and knowledgeable AI assistant.
Your goal is to answer the user's question accurately based on the provided reference context and conversation history.
Please answer in {answer_language}.

Rules:
1. Use the provided context to answer. If the context has relevant info, cite it using [citation:x] format at the end of sentences.
2. If a sentence comes from multiple contexts, list all, e.g., [citation:3][citation:5].
3. Do not blindly repeat the context. Summarize and explain.
4. If the context is insufficient, rely on your general knowledge but mention that it's not from the provided context or that information is missing.
5. Maintain a professional and neutral tone.
"""

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # 4. Inject History
    # Filter valid messages and limit history
    valid_history = [h for h in history if h.get("role") in ["user", "assistant"]]
    if LLM_HISTORY_LIMIT > 0:
        valid_history = valid_history[-LLM_HISTORY_LIMIT:]
    
    messages.extend(valid_history)

    # 5. Construct Current User Message with Context
    user_content = f"Question: {question}\n\n"
    
    if context_str:
        user_content += f"Reference Context:\n```\n{context_str}\n```\n\n"
        user_content += "Please answer the question using the context above and our history."
    else:
        user_content += "Please answer the question based on our history and your knowledge."

    messages.append({"role": "user", "content": user_content})

    if debug:
        print("Generated Messages:")
        for m in messages:
            print(f"Role: {m['role']}, Length: {len(m['content'])}")

    return messages


def chat(messages, model:str, llm_auth_token:str, llm_base_url:str, using_custom_llm=False, stream=True, debug=False):
    # Setup OpenAI Client
    # Defaults
    api_key = llm_auth_token if llm_auth_token else "CUSTOM"
    base_url = llm_base_url

    # Predefined models mapping (simulated from original code)
    if not using_custom_llm:
        openai.base_url = "http://127.0.0.1:3040/v1/" # Default
        if model == "gpt3.5":
            base_url = "http://llm-freegpt35:3040/v1/"
        elif model == "kimi":
            base_url = "http://llm-kimi:8000/v1/"
        elif model == "glm4":
            base_url = "http://llm-glm4:8000/v1/"
        elif model == "qwen":
            base_url = "http://llm-qwen:8000/v1/"
    
    if using_custom_llm and llm_base_url:
        base_url = llm_base_url
    
    # Configure global openai (or use client instance if upgrading, but sticking to global for compatibility with older patterns if mixed)
    # However, newer openai versions use client. 
    # The requirement.txt has openai==1.16.2 which uses Client.
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=stream,
            max_tokens=1024,
            temperature=0.2
        )

        if stream:
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            if response.choices:
                yield response.choices[0].message.content
            else:
                yield ""

    except Exception as e:
        print(f"Chat Error: {e}")
        yield f"[Error: {str(e)}]"

    
def ask_internet(query:str,  history=None, model="gpt3.5", llm_auth_token="", llm_base_url="", using_custom_llm=False, debug=False, search_enabled=True):
    
    content_list = []
    search_links = []
    
    if search_enabled:
        if debug:
            print(f"Searching for: {query}")
        search_links, content_list = search_web_ref(query, debug=debug)
    
    # Generate messages
    messages = gen_prompt(query, content_list, history=history, debug=debug)
    
    # Stream answer
    total_token = ""
    for token in chat(messages, model, llm_auth_token, llm_base_url, using_custom_llm, stream=True, debug=debug):
        if token:
            total_token += token
            yield token
            
    # Yield references if search was performed and we have results
    if search_enabled and search_links:
        yield "\n\n---\n**References:**\n"
        count = 1
        for item in search_links:
            url = item.get('url')
            title = item.get('title')
            yield f"{count}. [{title}]({url})\n"
            count += 1