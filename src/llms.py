import os

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from enum import Enum

class LLMLabel(Enum):
    GPT_4 = "gpt_4"
    GPT_4O = "gpt_4o"
    GPT_4O_MINI = "gpt_4o_mini"
    GPT_O1_MINI = "gpt_o1_mini" # Not working yet
    SONNET = "sonnet_3_5"
    GEMINI = "gemini_1_5_pro"
    LLAMA = "llama_3_8b"


# Type definitions
LLMs = dict[LLMLabel, Runnable]


# Initialize LangChain
def init_langchain():
    # Use the LangChain API key as needed (e.g., for tracing)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Get the LLMs
def get_llms(llm_names: list[LLMLabel]) -> LLMs:
    llms: LLMs = {}

    # OpenAI models: https://platform.openai.com/docs/models
    if LLMLabel.GPT_4 in llm_names:
        # https://platform.openai.com/docs/models
        llms[LLMLabel.GPT_4] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-0613",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.GPT_4O in llm_names:
        llms[LLMLabel.GPT_4O] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-2024-08-06",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.GPT_4O_MINI in llm_names:
        llms[LLMLabel.GPT_4O_MINI] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.GPT_O1_MINI in llm_names:
        llms[LLMLabel.GPT_O1_MINI] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-o1-mini-2024-09-12",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.SONNET in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/anthropic/
        llms[LLMLabel.SONNET] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.GEMINI in llm_names:
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        llms[LLMLabel.GEMINI] = ChatVertexAI(
            api_key=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            model="gemini-1.5-pro-002",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.LLAMA in llm_names:
        hugging_face_endpoint = HuggingFaceEndpoint(
            # repo_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
            # repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=1, # Suppress explanation of response.
            temperature=0.1 # 0 doesn't work
        )
        prompt_template = PromptTemplate(
            input_variables=["question"],
            template="Question: {question}\n\nAnswer:"
        )
        llms[LLMLabel.LLAMA] = LLMChain(llm=hugging_face_endpoint, prompt=prompt_template)

    return llms
