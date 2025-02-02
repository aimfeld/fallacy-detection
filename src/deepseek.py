from typing import Any, Dict, List, Optional
from langchain.schema import AIMessage, HumanMessage, BaseMessage
from langchain.chat_models.base import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr
from openai import OpenAI


class CustomDeepSeekChatModel(BaseChatModel):
    """
    A custom LangChain ChatModel for DeepSeek, which combines reasoning with the final answer.
    Reasoning is currently not included when using ChatOpenAI, even with the `include_reasoning` parameter.
    """
    model_name: str = "deepseek/deepseek-r1"
    _client: OpenAI = PrivateAttr()

    def __init__(self, client: OpenAI, **kwargs):
        super().__init__(**kwargs)  # Initialize Pydantic BaseModel fields
        self._client = client

    @property
    def _llm_type(self) -> str:
        """Return the type of the language model."""
        return "custom_deepseek"

    def _call(self, messages: List[HumanMessage], **kwargs: Any) -> ChatResult:
        """
        Process input messages and return an AIMessage with concatenated reasoning and final answer.
        """
        # Prepare the input for DeepSeek
        chat_messages = [{"role": "user", "content": msg.content} for msg in messages]
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=chat_messages,
            extra_body={"include_reasoning": True},
            max_tokens=4096
        )

        # Extract reasoning and final content
        reasoning = response.choices[0].message.model_extra.get("reasoning", "")
        final_answer = response.choices[0].message.content
        combined_message = f"<think>{reasoning}</think>\n\n{final_answer}"

        # Create a ChatGeneration object
        generation = ChatGeneration(
            message=AIMessage(content=combined_message),
            generation_info={"model_name": self.model_name}
        )

        # Return ChatResult with the generation
        return ChatResult(
            generations=[generation],
            llm_output={"model_name": self.model_name}
        )

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        """
        Implements the abstract `_generate` method required by BaseChatModel.
        This method wraps around `_call` to provide compatibility with LangChain's internal APIs.
        """
        human_messages = [msg for msg in messages if isinstance(msg, HumanMessage)]

        # Call the custom logic from `_call`
        return self._call(human_messages, **kwargs)

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for debugging."""
        return {"model_name": self.model_name}