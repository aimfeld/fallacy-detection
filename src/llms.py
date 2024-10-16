import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import ChatVertexAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

LLM_4O = "4o"
LLM_SONNET = "sonnet"
LLM_GEMINI = "gemini"
LLM_LLAMA = "llama"

def get_llms() -> dict:
    # https://python.langchain.com/docs/integrations/platforms/openai/
    llm_4o = ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0,
        timeout=3000,
        max_retries=2,
    )

    # https://python.langchain.com/docs/integrations/platforms/anthropic/
    llm_sonnet = ChatAnthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        timeout=3000,
        max_retries=2,
    )

    # https://python.langchain.com/docs/integrations/chat/google_vertex_ai_palm/
    llm_gemini = ChatVertexAI(
        api_key=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        model="gemini-1.5-pro",
        temperature=0,
        timeout=3000,
        max_retries=2,
    )

    hugging_face_endpoint = HuggingFaceEndpoint(
        #repo_id="meta-llama/Meta-Llama-3.1-405B-Instruct",
        #repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=1, # Suppress explanation of response.
        temperature=0.1 # 0 doesn't work
    )
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Question: {question}\n\nAnswer:"
    )
    llm_llama = LLMChain(llm=hugging_face_endpoint, prompt=prompt_template)

    return {
        LLM_4O: llm_4o,
        LLM_SONNET: llm_sonnet,
        LLM_GEMINI: llm_gemini,
        LLM_LLAMA: llm_llama
    }