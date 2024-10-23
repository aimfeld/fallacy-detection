"""
LLMs (Large Language Models) for Fallacy Detection.
"""
import os

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint
from enum import Enum

# LLM keys
class LLM(Enum):
    GPT_4 = "gpt_4"
    GPT_4O = "gpt_4o"
    GPT_4O_MINI = "gpt_4o_mini"
    O1_MINI = "o1_mini" # Not working yet
    O1_PREVIEW = "o1_preview"
    CLAUDE_3_5_SONNET = "claude_3_5_sonnet"
    CLAUDE_3_OPUS = "claude_3_opus"
    CLAUDE_3_HAIKU = "claude_3_haiku"
    GEMINI_1_5_PRO = "gemini_1_5_pro"
    GEMINI_1_5_FLASH = "gemini_1_5_flash"
    GEMINI_1_5_FLASH_8B = "gemini_1_5_flash_8b"
    LLAMA_3_1_70B = "llama_3_1_70b"

    @property
    def label(self):
        return LLMLabel[self.name].value

    @property
    def group(self):
        return LLMGroup[self.name].value

    @property
    def provider(self):
        return LLMProvider[self.name].value


# LLM labels for display
class LLMLabel(Enum):
    GPT_4 = "GPT-4"
    GPT_4O = "GPT-4o"
    GPT_4O_MINI = "GPT-4o Mini"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview" # Lowercase is intential, see https://platform.openai.com/docs/models/o1
    CLAUDE_3_5_SONNET = "Claude 3.5 Sonnet"
    CLAUDE_3_OPUS = "Claude 3 Opus"
    CLAUDE_3_HAIKU = "Claude 3 Haiku"
    GEMINI_1_5_PRO = "Gemini 1.5 Pro"
    GEMINI_1_5_FLASH = "Gemini 1.5 Flash"
    GEMINI_1_5_FLASH_8B = "Gemini 1.5 Flash 8B"
    LLAMA_3_1_70B = "Llama 3.1 70B"


class LLMGroup(Enum):
    GPT_4 = "flagship"
    GPT_4O = "flagship"
    GPT_4O_MINI = "lightweight"
    O1_MINI = "lightweight"
    O1_PREVIEW = "flagship"
    CLAUDE_3_5_SONNET = "flagship"
    CLAUDE_3_OPUS = "flagship"
    CLAUDE_3_HAIKU = "lightweight"
    GEMINI_1_5_PRO = "flagship"
    GEMINI_1_5_FLASH = "lightweight"
    GEMINI_1_5_FLASH_8B = "lightweight"
    LLAMA_3_1_70B = "open-source"


class LLMProvider(Enum):
    GPT_4 = "OpenAI"
    GPT_4O = "OpenAI"
    GPT_4O_MINI = "OpenAI"
    O1_MINI = "OpenAI"
    O1_PREVIEW = "OpenAI"
    CLAUDE_3_5_SONNET = "Anthropic"
    CLAUDE_3_OPUS = "Anthropic"
    CLAUDE_3_HAIKU = "Anthropic"
    GEMINI_1_5_PRO = "Google"
    GEMINI_1_5_FLASH = "Google"
    GEMINI_1_5_FLASH_8B = "Google"
    LLAMA_3_1_70B = "Meta"


# Type definitions
LLMs = dict[LLM, Runnable]


# Initialize LangChain
def init_langchain():
    # Use the LangChain API key as needed (e.g., for tracing)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")


# Get the LLMs
def get_llms(llm_names: list[LLM]) -> LLMs:
    llms: LLMs = {}

    # OpenAI models: https://platform.openai.com/docs/models
    if LLM.GPT_4 in llm_names:
        # https://platform.openai.com/docs/models
        llms[LLM.GPT_4] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4-0613",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.GPT_4O in llm_names:
        llms[LLM.GPT_4O] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-2024-08-06",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.GPT_4O_MINI in llm_names:
        llms[LLM.GPT_4O_MINI] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.O1_MINI in llm_names:
        llms[LLM.O1_MINI] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="o1-mini-2024-09-12",
            temperature=1, # Only temperature=1 is allowed
            timeout=10.0, # Needs longer to respond
            max_retries=2,
        )

    if LLM.O1_PREVIEW in llm_names:
        llms[LLM.O1_PREVIEW] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="o1-preview-2024-09-12",
            temperature=1, # Only temperature=1 is allowed
            timeout=30.0, # Needs longer to respond
            max_retries=2,
        )

    if LLM.CLAUDE_3_5_SONNET in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/anthropic/
        llms[LLM.CLAUDE_3_5_SONNET] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.CLAUDE_3_OPUS in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/anthropic/
        llms[LLM.CLAUDE_3_OPUS] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-opus-20240229",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.CLAUDE_3_HAIKU in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/anthropic/
        llms[LLM.CLAUDE_3_HAIKU] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-haiku-20240307",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.GEMINI_1_5_PRO in llm_names:
        llms[LLM.GEMINI_1_5_PRO] = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-pro-002",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.GEMINI_1_5_FLASH in llm_names:
        llms[LLM.GEMINI_1_5_FLASH] = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-flash-002",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.GEMINI_1_5_FLASH_8B in llm_names:
        llms[LLM.GEMINI_1_5_FLASH_8B] = ChatGoogleGenerativeAI(
            api_key=os.getenv("GOOGLE_API_KEY"),
            model="gemini-1.5-flash-8b-001",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.LLAMA_3_1_70B in llm_names:
        llms[LLM.LLAMA_3_1_70B] = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            # repo_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
            repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            # repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=3, # Suppress explanation of response.
            temperature=0.1 # 0 doesn't work
        )

    return llms
