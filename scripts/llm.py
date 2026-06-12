import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


load_dotenv()


class OpenAILLM:
    def __init__(self, model="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0.1,
        )

    def generate(self, query: str, context: str):

        prompt = f"""
Context:
{context}

Question:
{query}

Answer clearly.
"""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content