from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import SeleniumURLLoader

#list of URLs to extract help center knowledgebase
urls = ['https://intercom.help/....'
        ]

#load conetents of pages
loader = SeleniumURLLoader(urls=urls)
docs_not_splitted = loader.load()

#use CharacterTextSplitter to split the docst into smaller chunks to be stored in the vecor store
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
docs = text_splitter.split_documents(docs_not_splitted)

embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002")

# create Deep Lake dataset
#use your organization id here. (by default, org id is your username)
my_activeloop_id="your_deeplake_id"
my_activeloop_dataset_name = "your_dataset_name"
dataset_path = f"hub://{my_activeloop_id}/{my_activeloop_dataset_name}"

#uncomment the following line if you need to create a new dataset
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)

#reload the existing dataset
#db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings, read_only=False)

#add embedded doucments to the dataset
db.add_documents(docs)