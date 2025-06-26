from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import requests
import json
from utils.prompts import system_prompt, rag_prompt
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import tiktoken
import openai


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
    def __init__(self, base_url: str = "https://llm.iacpaas.dvo.ru/api/inference"):

        try:
            load_dotenv()
            self.api_token = os.getenv("IACPAAS_TOKEN")
            self.model_name = os.getenv("IACPAAS_MODEL")
        except Exception as e:
            print(f"Error loading IACPAAS API key: {str(e)}")
            raise

        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
        }

    def query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 3000,
    ):
        payload = {
            "auth_token": self.api_token,
            "model_name": self.model_name,
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

        try:
            # Debug: Log payload size and first part of content
            print(f"🔍 IACPAAS DEBUG: Payload size: {len(str(payload))} chars")
            print(f"🔍 IACPAAS DEBUG: User prompt size: {len(user_prompt)} chars")
            print(f"🔍 IACPAAS DEBUG: User prompt preview: {user_prompt[:200]}...")

            response = requests.get(
                self.base_url,
                json=payload,
            )

            print(f"🔍 IACPAAS DEBUG: Response status: {response.status_code}")

            if response.status_code != 200:
                print(f"🔍 IACPAAS DEBUG: Response text: {response.text}")

            response.raise_for_status()

            try:
                response_data = response.json()
                return response_data["output"]
            except json.JSONDecodeError as e:
                print(f"❌ IACPAAS JSON parsing error: {str(e)}")
                print(f"❌ Raw response content: {response.text[:500]}...")
                raise Exception(f"Failed to parse IACPAAS response as JSON: {str(e)}")

        except requests.exceptions.RequestException as e:
            print(f"❌ IACPAAS Request Error: {str(e)}")
            if hasattr(e, "response") and e.response is not None:
                print(f"❌ Response content: {e.response.text}")
            raise
        except Exception as e:
            print(f"❌ IACPAAS General Error: {str(e)}")
            raise


class OpenAIClient:
    """OpenAI API client for comparison testing"""

    def __init__(self, model: str = "gpt-4o-mini"):
        load_dotenv()

        # Check for required environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def query_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.01,
        max_tokens: int = 3000,
    ):
        """Query OpenAI with the same interface as LLMClient"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=(
                    {"type": "json_object"} if "json" in user_prompt.lower() else {"type": "text"}
                ),
            )

            return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
