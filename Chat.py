import os, time, shutil, subprocess, types
import params

if params.set_visible_devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = params.visible_devices

import electron_mic_tools, llms

import torch

from langchain import PromptTemplate, LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
from langchain.agents import Tool, AgentType, initialize_agent
from langchain.document_loaders import OnlinePDFLoader
# from langchain_community.document_loaders import PyPDFLoader
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings 

# import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openparse
from openparse.schemas import Bbox

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeRelationship

# #Setup device
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Device:", device)
# print("Using %d GPUs" %torch.cuda.device_count())

# #Cleanups
# gr.close_all() #Close any existing open ports'

def clean_pdf_paths():
    if os.path.exists(params.pdf_path): #Remove any PDF embeddings
        shutil.rmtree(params.pdf_path)
    if os.path.exists(params.pdf_text_path): #Remove any raw PDF text
        shutil.rmtree(params.pdf_text_path)
    os.mkdir(params.pdf_text_path)


def init_local_llm(params):
    #Create a local tokenizer copy the first time
    if os.path.isdir(params.tokenizer_path):
        tokenizer = AutoTokenizer.from_pretrained(params.tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(params.model_name)
        os.mkdir(params.tokenizer_path)
        tokenizer.save_pretrained(params.tokenizer_path)

    #Setup pipeline
    model = AutoModelForCausalLM.from_pretrained(params.model_name, 
                                                 device_map="auto", 
                                                 torch_dtype=torch.bfloat16)#, load_in_8bit=True)
    pipe = pipeline(
        "text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=params.seq_length,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2
    )

    #Setup LLM chain with memory and context
    return HuggingFacePipeline(pipeline=pipe)

#Setup embedding model
def init_local_embeddings(params):
    return HuggingFaceEmbeddings(model_name=params.embedding_model_name)



"""
===========================
Chat Functionality
===========================
"""

class Chat():
    def __init__(self, llm, embedding, doc_store):
        self.llm = llm 
        self.embedding = embedding
        self.memory, self.conversation = self._init_chain()
        self.doc_store = doc_store
        self.is_PDF = False #Flag to use NER over right set of docs. Changed in update_pdf_docstore


    def _init_chain(self):
        template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Context:
{context}

Current conversation:
{history}
Human: {input}
AI:"""

        PROMPT = PromptTemplate(
            input_variables=["history", "input", "context"], template=template
        )
        memory = ConversationBufferWindowMemory(memory_key="history", 
                                                input_key = "input", 
                                                k=6)

        conversation = LLMChain(
                prompt=PROMPT,
                llm=self.llm, 
                verbose=True, 
                memory=memory
        )

        return memory, conversation

    def _get_context(self, query, doc_store):
        
        # Context retrieval from embeddings
        retriever = VectorIndexRetriever(
            index=doc_store,
            similarity_top_k=3,
        )
        nodes = retriever.retrieve(query)
        op_nodes_dict = {}

        context = ""
        for i in range(min(params.N_hits, len(nodes))):
            bbox = nodes[i].metadata['bbox'][0]
            op_node = types.SimpleNamespace()
            op_node.bbox = Bbox(page=bbox['page'], page_height=bbox['page_height'], page_width=bbox['page_width'], x0=bbox['x0'], y0=bbox['y0'], x1=bbox['x1'], y1=bbox['y1'])
            op_node.coordinate_system = "bottom-left"
            context += nodes[i].text
            filename = nodes[i].node.parent_node.metadata['file_name']
            if filename not in op_nodes_dict:
                op_nodes_dict[filename] = [op_node]
            else:
                op_nodes_dict[filename].append(op_node)
                
        for filename, op_nodes in op_nodes_dict.items():
            pdf = openparse.Pdf('assets/'+filename)
            pdf.export_with_bboxes(op_nodes, output_pdf="bboxed_"+filename)

        #Context retrieval from NER
        ners = llms.ner_hits(query) #Get unique named entities of > some length from query
        ner_hits = []

        #Set path from where to get NER context hits
        if self.is_PDF:
            doc_path = params.pdf_text_path
            print("Getting NER hits from PDF context")
        else: 
            doc_path = params.doc_path_root
            clean_pdf_paths() #Make sure PDF folders are clean to avoid context leak
            print("Getting NER hits from facility context")

        for ner in ners: #Grep NEs from raw text
            try: 
                hit = subprocess.check_output("grep -r -i -h '%s' %s/" %(ner, doc_path), shell=True).decode()
                hits = hit.split("\n") #split all the grep results into indiv strings
                ner_hits.extend(hits)
            except subprocess.CalledProcessError as err:
                if err.returncode > 1:
                    print ("No hits found for: ", ner) 
                    continue
                #Exit values: 0 One or more lines were selected. 1 No lines were selected. >1 An error occurred.
        #print ("NERs", ner_hits)

        ner_hits.sort(key=len, reverse=True) #Sort by length of hits
        #print ("Sorted NERs", ner_hits)

        for i in range(min(params.N_NER_hits, len(ner_hits))):
            print ("Selected NER hit %d : " %i, ner_hits[i])
            context += ner_hits[i]

        return context
    # #Method to find text with highest likely context
    # def _get_context(self, query, doc_store):

    #     # Context retrieval from embeddings
    #     docs = doc_store.similarity_search_with_score(query, k=params.N_hits)
    #     #Get context strings
    #     context=""
    #     print ("Context hits found", len(docs))
    #     for i in range(min(params.N_hits, len(docs))):
    #         if docs[i][1]<params.similarity_cutoff:
    #             context += docs[i][0].page_content +"\n"
    #             print (i+1, len(docs[i][0].page_content), docs[i][1], docs[i][0].page_content)
    #         else:
    #             print ("\n\nIGNORING CONTENT of score %.2f" %docs[i][1],len(docs[i][0].page_content), docs[i][0].page_content)

    #     #Context retrieval from NER
    #     ners = llms.ner_hits(query) #Get unique named entities of > some length from query
    #     ner_hits = []

    #     #Set path from where to get NER context hits
    #     if self.is_PDF:
    #         doc_path = params.pdf_text_path
    #         print("Getting NER hits from PDF context")
    #     else: 
    #         doc_path = params.doc_path_root
    #         clean_pdf_paths() #Make sure PDF folders are clean to avoid context leak
    #         print("Getting NER hits from facility context")

    #     for ner in ners: #Grep NEs from raw text
    #         try: 
    #             hit = subprocess.check_output("grep -r -i -h '%s' %s/" %(ner, doc_path), shell=True).decode()
    #             hits = hit.split("\n") #split all the grep results into indiv strings
    #             ner_hits.extend(hits)
    #         except subprocess.CalledProcessError as err:
    #             if err.returncode > 1:
    #                 print ("No hits found for: ", ner) 
    #                 continue
    #             #Exit values: 0 One or more lines were selected. 1 No lines were selected. >1 An error occurred.
    #     #print ("NERs", ner_hits)

    #     ner_hits.sort(key=len, reverse=True) #Sort by length of hits
    #     #print ("Sorted NERs", ner_hits)

    #     for i in range(min(params.N_NER_hits, len(ner_hits))):
    #         print ("Selected NER hit %d : " %i, ner_hits[i])
    #         context += ner_hits[i]

    #     return context
    
    
    def generate_response(self, history, debug_output):
        user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]
        all_user_messages = [x[0] for x in history]
        print(all_user_messages)

        if self.doc_store is None:
            context = ""
        else:
            context = ""
            for message in all_user_messages:
             context += self._get_context(message, self.doc_store)

        if debug_output:
            inputs = self.conversation.prep_inputs({'input': user_message, 'context':context})
            prompt = self.conversation.prep_prompts([inputs])[0][0].text

        bot_message = self.conversation.predict(input=user_message, context=context)
        #Pass user message and get context and pass to model
        history[-1][1] = "" #Replaces None with empty string -- Gradio code

        if debug_output:
            bot_message = f'---Prompt---\n\n {prompt} \n\n---Response---\n\n {bot_message}'

        for character in bot_message:
            history[-1][1] += character
            #time.sleep(0.02)
            #yield history
        return history

    def add_message(self, user_message, history):
        return "", history + [[user_message, None]]


class PDFChat(Chat):
    def update_pdf_docstore(self, pdf_docs):
        index = None
        for pdf_doc in pdf_docs:
            parser = openparse.DocumentParser()
            parsed_doc = parser.parse(pdf_doc)
            nodes = parsed_doc.to_llama_index_nodes()
            if index is None:
                index = VectorStoreIndex(nodes=nodes)
            else:
                index.insert_nodes(nodes)

        index.storage_context.persist(persist_dir='./assets/')
        self.doc_store = index
        self.is_PDF = True

        return "PDF Ready"
    

class ToolChat(Chat):
    """
    Implements an agentexector in a chat context. The agentexecutor is called in a fundimentally
    differnet way than the other chains, so custom implementaiton for much of the class.
    """
    def _init_chain(self):
        """
        tools = [
            electron_mic_tools.VirtualImagingSim(params.spec_init)   
        ]
        """

        tools = [electron_mic_tools.probe_sim_tool, electron_mic_tools.virtual_imaging_tool]

        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=6)
        conversation = initialize_agent(tools, 
                                       self.llm, 
                                       agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                                       verbose=True, 
                                       handle_parsing_errors='Check your output and make sure it conforms!',
                                       max_iterations=5,
                                       memory=memory)
        return memory, conversation
    
    
    def generate_response(self, history, debug_output):
        user_message = history[-1][0] #History is list of tuple list. E.g. : [['Hi', 'Test'], ['Hello again', '']]

        # TODO: Implement debug output for langchain agents. Might have to use a callback?
        print(f'User input: {user_message}')
        bot_message = self.conversation.run(user_message)
        #Pass user message and get context and pass to model
        history[-1][1] = "" #Replaces None with empty string -- Gradio code

        for character in bot_message:
            history[-1][1] += character
            time.sleep(0.02)
            yield history


