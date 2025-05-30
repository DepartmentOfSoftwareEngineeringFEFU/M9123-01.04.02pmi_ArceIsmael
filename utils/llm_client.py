from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests
from utils.prompts import system_prompt, rag_prompt
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tiktoken


class LangChainWrapper:
    def __init__(self, engine: str = "gpt-4o", temperature: float = 0, streaming: bool = True):
        try:
            load_dotenv()

            # Check for required environment variables
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")

            self.llm = ChatOpenAI(model_name=engine, temperature=temperature, streaming=streaming)
            self.rag_chain = None
        except Exception as e:
            print(f"Error initializing LangChainWrapper: {str(e)}")
            raise

    def create_rag_chain(
        self, splits: list[Document], embedding_model: str = "text-embedding-3-large"
    ):
        try:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            # Initialize empty vector store
            embeddings = OpenAIEmbeddings(model=embedding_model)
            vectorstore = InMemoryVectorStore(embedding=embeddings)

            # Process documents in batches to avoid token limits
            batch_size = 100  # Process 100 documents at a time

            for i in range(0, len(splits), batch_size):
                batch = splits[i : i + batch_size]
                # Add documents to vector store in batches
                vectorstore.add_documents(batch)

            retriever = vectorstore.as_retriever()

            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            self.rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        except Exception as e:
            print(f"Error creating RAG chain: {str(e)}")
            raise

    def query_rag_chain(self, question: str = rag_prompt):
        try:
            if self.rag_chain is None:
                raise ValueError("RAG chain not initialized. Call create_rag_chain first.")
            results = self.rag_chain.invoke({"input": question})
            return results
        except Exception as e:
            print(f"Error querying RAG chain: {str(e)}")
            raise

    def query_llm(self, question: str):
        try:
            results = self.llm.invoke(question)
            return results
        except Exception as e:
            print(f"Error querying LLM: {str(e)}")
            raise


class LLMClient:
    def __init__(self, base_url: str = "https://llm.iacpass.dvo.ru"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 3000,
    ):
        payload = {
            "conversation": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "generation_config": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "max_time": 360,
                "repetition_penalty": 1,
                "top_p": 1,
            },
        }

        api_url = "https://llm.iacpaas.dvo.ru/inference"

        response = requests.get(
            api_url,
            json=payload,
        )

        response.raise_for_status()

        return response.json()["output"]
