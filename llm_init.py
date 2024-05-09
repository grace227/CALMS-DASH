import argparse
import params
import llms
import requests

def get_parser():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True) #One of mutuallu exclusive args is required
    group.add_argument('-hf', '--huggingface', action='store_true', help='Open-source model')
    group.add_argument('-o', '--openai', action='store_true', help='OpenAI Model')
    return parser

def get_llm_type(option:str='--openai'):
    parser = get_parser()
    llm_type = parser.parse_args(['--openai'])
    return llm_type

def get_llm(option:str='--openai'):
    llm_type = get_llm_type(option)
    if llm_type.openai:
        llm = llms.AnlLLM(params)
    return llm  

def get_embeddings(option:str='--openai'):
    llm_type = get_llm_type(option)
    embed_type = llm_type # Can be different from llm_type
    embeddings = None
    if embed_type.openai:
        if params.init_docs:
            input('WARNING: WILL INIT ALL DOCS WITH OPENAI EMBEDS. Press enter to continue')
        params.embed_path = f"{params.base_path}/anl_openai"
        embeddings = llms.ANLEmbeddingModel(params)
    return embeddings

# def chat_init(llm, embeddings):
#     chat = Chat(llm, embeddings, doc_store=None)
#     return chat

