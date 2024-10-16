import os

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

LLM_GPT_4 = "gpt_4"
LLM_GPT_4O = "gpt_4o"
LLM_SONNET = "sonnet"
LLM_GEMINI = "gemini"
LLM_LLAMA = "llama"

# Initialize LangChain
def init_langchain():
    # Use the LangChain API key as needed (e.g., for tracing)
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Type definitions
LLMs = dict[str, Runnable]


# Get the LLMs
def get_llms(llm_names: list[str]) -> LLMs:
    llms: LLMs = {}

    if LLM_GPT_4 in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/openai/
        llms[LLM_GPT_4] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLM_GPT_4O in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/openai/
        llms[LLM_GPT_4O] = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLM_SONNET in llm_names:
        # https://python.langchain.com/docs/integrations/platforms/anthropic/
        llms[LLM_SONNET] = ChatAnthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            model="claude-3-5-sonnet-20240620",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLM_GEMINI in llm_names:
        # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
        llms[LLM_GEMINI] = ChatVertexAI(
            api_key=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
            model="gemini-1.5-pro",
            temperature=0,
            timeout=3000,
            max_retries=3,
        )

    if LLM_LLAMA in llm_names:
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
        llms[LLM_LLAMA] = LLMChain(llm=hugging_face_endpoint, prompt=prompt_template)

    return llms
