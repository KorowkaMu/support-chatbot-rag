from langchain.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.llms import OpenAI
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os


my_activeloop_id="your_deeplake_id" #replace with your deeplake id
my_activeloop_dataset_name = "your_dataset_name" #replace with your dataset name
dataset_path = f"hub://{my_activeloop_id}/{my_activeloop_dataset_name}"

def data_lake():
    embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

    dbs = DeepLake(
        dataset_path=dataset_path, 
        read_only=True, 
        embedding=embeddings
        )
    retriever = dbs.as_retriever()
    retriever.search_kwargs["distance_metric"] = "cos"
    retriever.search_kwargs["fetch_k"] = 20
    retriever.search_kwargs["maximal_marginal_relevance"] = True
    retriever.search_kwargs["k"] = 20

    compressor_llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
    compressor = LLMChainExtractor.from_llm(compressor_llm)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
        )
    return dbs, compression_retriever, retriever

dbs, compression_retriever, retriever = data_lake()

def memory():
    memory=ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True, 
        output_key='answer'
        )
    return memory

memory=memory()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", streaming=True, verbose=True, temperature=0, max_tokens=1500)

# Define the system message template
system_template = """You are an exceptional customer support chatbot. Use the context below to answer user's question.
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        ----------------
        {context}"""

# Create the chat prompt templates
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")]


qa_prompt = ChatPromptTemplate.from_messages(messages)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=compression_retriever,
    memory=memory,
    verbose=False,
    chain_type="stuff",
    return_source_documents=False,
    combine_docs_chain_kwargs={"prompt": qa_prompt}
)

# Load the memory variables, which include the chat history
memory_variables = memory.load_memory_variables({})

# Variable to store the result
result = {}


#1st question
query = "Question 1 contents?"
result = chain({"question": query, "chat_history": memory_variables})
print(result)

#2nd question
query = "Follow up question contents?"
result = chain({"question": query, "chat_history": memory_variables})
print(result)
