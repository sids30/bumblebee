from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import nltk
import os
import pandas as pd

# Download required NLTK data
nltk.download('punkt')

# Set Hugging Face API token as environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "Your HuggingFaceHub API Token Key" # I had my own key for testing, please create your own for free off the site if you do not have one

# Load and process unsorted data
loader = TextLoader("durham_college_data.txt")
documents = loader.load()
text_splitter = NLTKTextSplitter(chunk_size=8000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Load CSV file using pandas (sorted scraped data)
csv_data = pd.read_csv("qa_pairs.csv")

# Convert CSV data to list of document objects
csv_documents = []
for _, row in csv_data.iterrows():
    question = row['question']
    answer = row['answer']
    document = Document(
        page_content=f"Question: {question}\nAnswer: {answer}",
        metadata={"source": "CSV"}
    )
    csv_documents.append(document)

# Combine unsorted data text and CSV file
texts.extend(csv_documents)

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Use large Google Flan language model
llm = HuggingFaceHub(repo_id="google/flan-t5-large", task='text2text-generation')

# Create RetrievalQA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="map_reduce", retriever=retriever, return_source_documents=False, verbose=True
)

# Chatbot prompt loop
while True:
    print("Ask me a question about Durham College (To Quit: Type 'exit'):")
    query = input()
    if query.lower() == 'exit':
        break
    result = qa({"query": query})
    print(f"Result: {result['result']}")
