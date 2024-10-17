import os

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from enum import Enum

class LLMLabel(Enum):
    GPT_4 = "gpt_4"
    GPT_4O = "gpt_4o"
    GPT_4O_MINI = "gpt_4o_mini"
    GPT_O1_MINI = "gpt_o1_mini" # Not working yet
    SONNET = "sonnet_3_5"
    GEMINI_1_5_PRO = "gemini_1_5_pro"
    GEMINI_1_5_FLASH = "gemini_1_5_flash"
    LLAMA_3_1_70B = "llama_3_1_70b"


# Type definitions
LLMs = dict[LLMLabel, Runnable]


# Initialize LangChain
def init_langchain():
    # Use the LangChain API key as needed (e.g., for tracing)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")


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

    if LLMLabel.GEMINI_1_5_PRO in llm_names:
        llms[LLMLabel.GEMINI_1_5_PRO] = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-pro-002",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.GEMINI_1_5_FLASH in llm_names:
        llms[LLMLabel.GEMINI_1_5_FLASH] = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-flash-002",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLMLabel.LLAMA_3_1_70B in llm_names:
        llms[LLMLabel.LLAMA_3_1_70B] = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            # repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            # max_new_tokens=3, # Suppress explanation of response.
            temperature=0.1 # 0 doesn't work
        )

    return llms
