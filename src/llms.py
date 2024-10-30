"""
LLMs (Large Language Models) for Fallacy Detection.
"""
import os

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"
    META = "Meta"


class LLMGroup(Enum):
    FLAGSHIP = "flagship"
    MEDIUM = "medium"
    LIGHTWEIGHT = "lightweight"
    FINE_TUNED = "fine-tuned"


class LLM(Enum):
    GPT_4 = ("gpt_4", "GPT-4", LLMGroup.FLAGSHIP, LLMProvider.OPENAI)
    GPT_4O = ("gpt_4o", "GPT-4o", LLMGroup.FLAGSHIP, LLMProvider.OPENAI)
    GPT_4O_MINI = ("gpt_4o_mini", "GPT-4o Mini", LLMGroup.LIGHTWEIGHT, LLMProvider.OPENAI)
    GPT_4O_MINI_TUNED = ("gpt_4o_mini_tuned_v1", "GPT-4o Mini Tuned", LLMGroup.FINE_TUNED, LLMProvider.OPENAI)
    O1_MINI = ("o1_mini", "o1-mini", LLMGroup.LIGHTWEIGHT, LLMProvider.OPENAI)
    O1_PREVIEW = ("o1_preview", "o1-preview", LLMGroup.FLAGSHIP, LLMProvider.OPENAI)
    CLAUDE_3_5_SONNET = ("claude_3_5_sonnet", "Claude 3.5 Sonnet", LLMGroup.FLAGSHIP, LLMProvider.ANTHROPIC)
    CLAUDE_3_OPUS = ("claude_3_opus", "Claude 3 Opus", LLMGroup.FLAGSHIP, LLMProvider.ANTHROPIC)
    CLAUDE_3_HAIKU = ("claude_3_haiku", "Claude 3 Haiku", LLMGroup.LIGHTWEIGHT, LLMProvider.ANTHROPIC)
    GEMINI_1_5_PRO = ("gemini_1_5_pro", "Gemini 1.5 Pro", LLMGroup.FLAGSHIP, LLMProvider.GOOGLE)
    GEMINI_1_5_FLASH = ("gemini_1_5_flash", "Gemini 1.5 Flash", LLMGroup.LIGHTWEIGHT, LLMProvider.GOOGLE)
    GEMINI_1_5_FLASH_8B = ("gemini_1_5_flash_8b", "Gemini 1.5 Flash 8B", LLMGroup.LIGHTWEIGHT, LLMProvider.GOOGLE)
    LLAMA_3_1_405B = ("llama_3_1_405b", "Llama 3.1 405B", LLMGroup.FLAGSHIP, LLMProvider.META)
    LLAMA_3_1_70B = ("llama_3_1_70b", "Llama 3.1 70B", LLMGroup.MEDIUM, LLMProvider.META)
    LLAMA_3_1_8B = ("llama_3_1_8b", "Llama 3.1 8B", LLMGroup.LIGHTWEIGHT, LLMProvider.META)


    def __init__(self, key: str, label: str, group: LLMGroup, provider: LLMProvider):
        self._key = key
        self._label = label
        self._group = group
        self._provider = provider

    @property
    def key(self) -> str:
        return self._key

    @property
    def label(self) -> str:
        return self._label

    @property
    def group(self) -> LLMGroup:
        return self._group

    @property
    def provider(self) -> LLMProvider:
        return self._provider


# Type definitions
LLMs = dict[LLM, Runnable]


def init_langchain():
    """
    Initialize LangChain
    """
    # Use the LangChain API key as needed (e.g., for tracing)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")


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

    if LLM.GPT_4O_MINI_TUNED in llm_names:
        llms[LLM.GPT_4O_MINI_TUNED] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="ft:gpt-4o-mini-2024-07-18:personal:fallacy-detection-v1:ANJNVY26",
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

    # Meta models: https://huggingface.co/meta-llama
    if LLM.LLAMA_3_1_405B in llm_names:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            repo_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
            task="text-generation",
            timeout=30.0,
        )
        # The ChatHuggingFace wrapper adds model specific special tokens, see https://huggingface.co/blog/langchain
        # Use bind() to work around bug: https://github.com/langchain-ai/langchain/issues/23586
        llms[LLM.LLAMA_3_1_405B] = ChatHuggingFace(llm=llm).bind(
            max_tokens=8192, # Prevent cutoff for CoT prompt answers
            temperature=0.0
        ).with_retry(stop_after_attempt=3)

    if LLM.LLAMA_3_1_70B in llm_names:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            task="text-generation",
            timeout=20.0,
        )
        llms[LLM.LLAMA_3_1_70B] = ChatHuggingFace(llm=llm).bind(
            max_tokens=8192, # Prevent cutoff for CoT prompt answers
            temperature=0.0
        ).with_retry(stop_after_attempt=3)

    if LLM.LLAMA_3_1_8B in llm_names:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation",
            timeout=10.0,
        )
        llms[LLM.LLAMA_3_1_8B] = ChatHuggingFace(llm=llm).bind(
            max_tokens=8192, # Prevent cutoff for CoT prompt answers
            temperature=0.0
        ).with_retry(stop_after_attempt=3)


    return llms
