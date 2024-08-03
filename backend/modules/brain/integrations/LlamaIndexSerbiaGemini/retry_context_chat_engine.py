
from typing import List, Optional

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import trace_method
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)
from llama_index.core.chat_engine.context import ContextChatEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation.base import BaseEvaluator
from llama_index.core.indices.query.query_transform.feedback_transform import (
    FeedbackQueryTransformation,
)
from llama_index.core.schema import QueryBundle

DEFAULT_CONTEXT_TEMPLATE = (
    "Context information is below."
    "\n--------------------\n"
    "{context_str}"
    "\n--------------------\n"
)


class CustomChatEngine(ContextChatEngine):
    def __init__(
        self,
        context_chat_engine: ContextChatEngine,
        evaluator: BaseEvaluator,
        max_retries: int = 3
    ) -> None:
        self._max_retries = max_retries
        self._evaluator = evaluator
        self._context_chat_engine = context_chat_engine
        super().__init__(context_chat_engine._retriever, context_chat_engine._llm, context_chat_engine._memory, context_chat_engine._prefix_messages, context_chat_engine._node_postprocessors, context_chat_engine._context_template, context_chat_engine.callback_manager)

    @trace_method("chat")
    def chat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        
        response:AgentChatResponse = self._context_chat_engine.chat(message=message, chat_history=chat_history)
        if self._max_retries <= 0:
            return response
        
        typed_response = Response(
            response=response.response,
            source_nodes=response.source_nodes
        )

        query_str = message
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            print("Evaluation returned True.")
            return response
        else:
            print("Evaluation returned False.")
            new_query_engine = CustomChatEngine(
                context_chat_engine=self._context_chat_engine,
                evaluator=self._evaluator, 
                max_retries=self._max_retries - 1
            )
            query_transformer = FeedbackQueryTransformation()
            query_bundle = QueryBundle(query_str=message)
            new_query = query_transformer.run(query_bundle, {"evaluation": eval})
            return new_query_engine.chat(new_query, chat_history)
        
    @trace_method("chat")
    async def achat(
        self, message: str, chat_history: Optional[List[ChatMessage]] = None
    ) -> AgentChatResponse:
        
        response:AgentChatResponse = await self._context_chat_engine.achat(message=message, chat_history=chat_history)

        if self._max_retries <= 0:
            return response
        
        typed_response = Response(
            response=response.response,
            source_nodes=response.source_nodes
        )

        query_str = message
        eval = self._evaluator.evaluate_response(query_str, typed_response)
        if eval.passing:
            print("Evaluation returned True.")
            return response
        else:
            print("Evaluation returned False.")
            print(response)
            new_query_engine = CustomChatEngine(
                context_chat_engine=self._context_chat_engine,
                evaluator=self._evaluator, 
                max_retries=self._max_retries - 1
            )
            query_transformer = FeedbackQueryTransformation()
            query_bundle = QueryBundle(query_str=message)
            new_query = query_transformer.run(query_bundle, {"evaluation": eval})
            return await new_query_engine.achat(new_query.query_str, chat_history)
