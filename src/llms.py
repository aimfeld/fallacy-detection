"""
LLMs (Large Language Models) for Fallacy Detection.
"""
import os

from .mafalda import FallacyResponse
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_mistralai import ChatMistralAI
from langchain_ollama import ChatOllama # For running DeepSeek model locally
from openai import OpenAI
from .deepseek import CustomDeepSeekChatModel
from enum import Enum


class LLMProvider(Enum):
    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"
    GOOGLE = "Google"
    META = "Meta"
    MISTRAL_AI = "Mistral AI"
    DEEPSEEK = "DeepSeek"
    NONE = "None"


class LLMGroup(Enum):
    LARGE = "large" # LLMs: >20B parameters
    SMALL = "small" # SLMs: 500M - 20B parameters
    REASONING = "reasoning" # Models with internal reasoning capabilities
    FINE_TUNED = "fine-tuned"
    HUMAN = "human" # Human reasoning

class LLM(Enum):
    GPT_4 = ("gpt_4", "GPT-4", LLMGroup.LARGE, LLMProvider.OPENAI)
    GPT_4O = ("gpt_4o", "GPT-4o", LLMGroup.LARGE, LLMProvider.OPENAI)
    GPT_4O_MINI = ("gpt_4o_mini", "GPT-4o Mini", LLMGroup.SMALL, LLMProvider.OPENAI)
    GPT_4O_MINI_IDENTIFICATION = ("gpt_4o_mini_identification", "GPT-4o Mini Identification", LLMGroup.FINE_TUNED, LLMProvider.OPENAI)
    GPT_4O_MINI_CLASSIFICATION = ("gpt_4o_mini_classification", "GPT-4o Mini Classification", LLMGroup.FINE_TUNED, LLMProvider.OPENAI)
    O1_MINI = ("o1_mini", "o1-mini", LLMGroup.REASONING, LLMProvider.OPENAI)
    O1_PREVIEW = ("o1_preview", "o1-preview", LLMGroup.REASONING, LLMProvider.OPENAI)
    CLAUDE_3_5_SONNET = ("claude_3_5_sonnet", "Claude 3.5 Sonnet", LLMGroup.LARGE, LLMProvider.ANTHROPIC)
    CLAUDE_3_5_SONNET_20241022 = ("claude_3_5_sonnet_20241022", "Claude 3.5 Sonnet 20241022", LLMGroup.LARGE, LLMProvider.ANTHROPIC)
    CLAUDE_3_OPUS = ("claude_3_opus", "Claude 3 Opus", LLMGroup.LARGE, LLMProvider.ANTHROPIC)
    CLAUDE_3_HAIKU = ("claude_3_haiku", "Claude 3 Haiku", LLMGroup.SMALL, LLMProvider.ANTHROPIC)
    GEMINI_1_5_PRO = ("gemini_1_5_pro", "Gemini 1.5 Pro", LLMGroup.LARGE, LLMProvider.GOOGLE)
    GEMINI_1_5_FLASH = ("gemini_1_5_flash", "Gemini 1.5 Flash", LLMGroup.SMALL, LLMProvider.GOOGLE)
    GEMINI_1_5_FLASH_8B = ("gemini_1_5_flash_8b", "Gemini 1.5 Flash 8B", LLMGroup.SMALL, LLMProvider.GOOGLE)
    LLAMA_3_1_405B = ("llama_3_1_405b", "Llama 3.1 405B", LLMGroup.LARGE, LLMProvider.META)
    LLAMA_3_1_70B = ("llama_3_1_70b", "Llama 3.1 70B", LLMGroup.LARGE, LLMProvider.META)
    LLAMA_3_1_8B = ("llama_3_1_8b", "Llama 3.1 8B", LLMGroup.SMALL, LLMProvider.META)
    MISTRAL_LARGE_2 = ("mistral_large_2", "Mistral Large", LLMGroup.LARGE, LLMProvider.MISTRAL_AI) # 123B
    MISTRAL_SMALL_2 = ("mistral_small_2", "Mistral Small", LLMGroup.SMALL, LLMProvider.MISTRAL_AI) # 22B
    DEEPSEEK_R1_14B = ("deepseek_r1_14b", "DeepSeek R1 14B", LLMGroup.REASONING, LLMProvider.DEEPSEEK)
    DEEPSEEK_R1_671B = ("deepseek_r1_671b", "DeepSeek R1 671B", LLMGroup.REASONING, LLMProvider.DEEPSEEK)

    # Human
    ADRIAN = ("adrian", "Adrian", LLMGroup.HUMAN, LLMProvider.NONE)

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


# noinspection PyArgumentList
def get_llms(llm_names: list[LLM]) -> LLMs:
    llms: LLMs = {}

    # -------------------------------------------------------------------------
    # OpenAI
    # https://platform.openai.com/usage
    # https://platform.openai.com/docs/models
    # https://platform.openai.com/settings/organization/billing/overview
    # https://openai.com/api/pricing/
    # -------------------------------------------------------------------------
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

    if LLM.GPT_4O_MINI_IDENTIFICATION in llm_names:
        llms[LLM.GPT_4O_MINI_IDENTIFICATION] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="ft:gpt-4o-mini-2024-07-18:personal:fallacy-identification-v2:AQH3aRxC",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.GPT_4O_MINI_CLASSIFICATION in llm_names:
        llms[LLM.GPT_4O_MINI_CLASSIFICATION] = ChatOpenAI(
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

    # -------------------------------------------------------------------------
    # Anthropic
    # https://console.anthropic.com/dashboard
    # https://docs.anthropic.com/en/docs/about-claude/models
    # https://python.langchain.com/docs/integrations/platforms/anthropic/
    # -------------------------------------------------------------------------
    if LLM.CLAUDE_3_5_SONNET in llm_names:
        llms[LLM.CLAUDE_3_5_SONNET] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    # The latest version of Claude 3.5 Sonnet, but the main model under test here is version 20240620
    if LLM.CLAUDE_3_5_SONNET_20241022 in llm_names:
        llms[LLM.CLAUDE_3_5_SONNET_20241022] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.CLAUDE_3_OPUS in llm_names:
        llms[LLM.CLAUDE_3_OPUS] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-opus-20240229",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    if LLM.CLAUDE_3_HAIKU in llm_names:
        llms[LLM.CLAUDE_3_HAIKU] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-haiku-20240307",
            temperature=0,
            timeout=3.0,
            max_retries=2,
        )

    # -------------------------------------------------------------------------
    # Google
    # https://python.langchain.com/docs/integrations/chat/google_generative_ai/
    # https://ai.google.dev/gemini-api/docs/models/gemini
    # https://aistudio.google.com/app/apikey
    # https://console.cloud.google.com/billing
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Meta models: https://huggingface.co/meta-llama
    # https://huggingface.co/
    # -------------------------------------------------------------------------
    if LLM.LLAMA_3_1_405B in llm_names:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            repo_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
            task="text-generation",
            timeout=30.0,
        )
        llms[LLM.LLAMA_3_1_405B] = _get_chat_hugging_face(llm)

    if LLM.LLAMA_3_1_70B in llm_names:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            repo_id="meta-llama/Meta-Llama-3.1-70B-Instruct",
            task="text-generation",
            timeout=20.0,
        )
        llms[LLM.LLAMA_3_1_70B] = _get_chat_hugging_face(llm)

    if LLM.LLAMA_3_1_8B in llm_names:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
            task="text-generation",
            timeout=10.0,
        )
        llms[LLM.LLAMA_3_1_8B] = _get_chat_hugging_face(llm)


    # -------------------------------------------------------------------------
    # Mistral AI
    # https://console.mistral.ai/
    # https://docs.mistral.ai/getting-started/models/models_overview/
    # -------------------------------------------------------------------------
    if LLM.MISTRAL_LARGE_2 in llm_names:
        llms[LLM.MISTRAL_LARGE_2] = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-large-2407",
            temperature=0,
            timeout=30.0, # This API times out a lot
            max_retries=5,
        )

    if LLM.MISTRAL_SMALL_2 in llm_names:
        llms[LLM.MISTRAL_SMALL_2] = ChatMistralAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            model="mistral-small-2409",
            temperature=0,
            timeout=30.0, # This API times out a lot
            max_retries=5,
        )

    # -------------------------------------------------------------------------
    # DeepSeek
    # -------------------------------------------------------------------------
    # DeepSeek R1 14B runs locally with a NVIDIA 4060/8GB GPU
    if LLM.DEEPSEEK_R1_14B in llm_names:
        llms[LLM.DEEPSEEK_R1_14B] = ChatOllama(
        model="deepseek-r1:14b",
        temperature=0,
    )

    if LLM.DEEPSEEK_R1_671B in llm_names:
        openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("OPENROUTER_API_KEY"))
        llms[LLM.DEEPSEEK_R1_671B] = CustomDeepSeekChatModel(openai_client)


    # if LLM.DEEPSEEK_R1_671B in llm_names:
    #     llms[LLM.DEEPSEEK_R1_671B] = ChatOpenAI(
    #         base_url="https://openrouter.ai/api/v1",
    #         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    #         model="deepseek/deepseek-r1",
    #         extra_body={"include_reasoning": True}, # See https://github.com/huggingface/chat-ui/issues/1664
    #         max_tokens=1024,
    #         temperature=0,
    #         timeout=120.0,
    #         max_retries=2,
    #     )


    return llms

# noinspection PyArgumentList
def get_fallacy_search_llms(llm_names: list[LLM]) -> LLMs:
    """
    Get the LLM for searching for fallacies in text. Only OpenAI models with structured outputs are supported, see
    https://platform.openai.com/docs/guides/structured-outputs
    """
    llms: LLMs = {}

    if LLM.GPT_4O in llm_names:
        llms[LLM.GPT_4O] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-2024-08-06",
            temperature=0, # Higher temperature might generate more identified fallacies
            timeout=30.0,
            max_retries=2,
        )

    if LLM.GPT_4O_MINI in llm_names:
        llms[LLM.GPT_4O_MINI] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            timeout=30.0,
            max_retries=2,
        )

    if LLM.GPT_4O_MINI_IDENTIFICATION in llm_names:
        llms[LLM.GPT_4O_MINI_IDENTIFICATION] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="ft:gpt-4o-mini-2024-07-18:personal:fallacy-identification-v2:AQH3aRxC",
            temperature=0,
            timeout=30.0,
            max_retries=2,
        )

    if LLM.GPT_4O_MINI_CLASSIFICATION in llm_names:
        llms[LLM.GPT_4O_MINI_CLASSIFICATION] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            # The model should have been named fallacy-classification, for better constistency in terminology
            model="ft:gpt-4o-mini-2024-07-18:personal:fallacy-detection-v1:ANJNVY26",
            temperature=0,
            timeout=30.0,
            max_retries=2,
        )

    # Note that o1-mini does not support the system role, nor structured output

    prompt = ChatPromptTemplate.from_messages(
        [('system', '{system_prompt}'), ('user', '{input}')]
    )

    # Models will generate validated structured outputs.
    for llm in llms:
        # noinspection PyUnresolvedReferences
        llms[llm] = prompt | llms[llm].with_structured_output(FallacyResponse, method='json_schema')

    return llms


def _get_chat_hugging_face(llm: HuggingFaceEndpoint) -> Runnable:
    """
    The ChatHuggingFace wrapper adds model specific special tokens, see https://huggingface.co/blog/langchain
    Use bind() to work around bug: https://github.com/langchain-ai/langchain/issues/23586
    """
    return ChatHuggingFace(llm=llm).bind(
        max_tokens=4096, # Prevent cutoff for CoT prompt answers
        temperature=0.0
    ).with_retry(stop_after_attempt=3)


