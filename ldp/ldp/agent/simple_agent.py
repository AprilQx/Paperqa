from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Self, cast

import tiktoken
from aviary.core import Message, Tool, ToolRequestMessage, ToolResponseMessage
from aviary.message import EnvStateMessage
from llmclient import CommonLLMNames
from pydantic import BaseModel, ConfigDict, Field

from ldp.graph import ConfigOp, FxnOp, LLMCallOp, OpResult, compute_graph
from ldp.llms import prepend_sys

from . import DEFAULT_LLM_COMPLETION_TIMEOUT
from .agent import Agent


class HiddenEnvStateMessage(EnvStateMessage):
    content: str = "[Previous environment state - hidden]"

class CostCalculator:
    """Class to calculate the cost of a message."""

    MODEL_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # Cost per 1K tokens
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-4o-mini": {"input": 0.0005, "output": 0.0015}}
    
    @staticmethod
    def num_tokens_from_messages(messages: list[Message], model: str) -> int:
        """Calculate number of tokens in the messages."""
        encoding = tiktoken.encoding_for_model(model)
        num_tokens = 0
        for message in messages:
            # Add tokens for message content
            if hasattr(message, 'content') and message.content:
                num_tokens += len(encoding.encode(str(message.content)))
            # Add tokens for message role (3 tokens for role)
            if hasattr(message, 'role'):
                num_tokens += 3
        return num_tokens
    
    @staticmethod
    def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
        """Calculate cost based on token usage and model."""
        if model not in CostCalculator.MODEL_COSTS:
            raise ValueError(f"Unknown model: {model}")
            
        costs = CostCalculator.MODEL_COSTS[model]
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return input_cost + output_cost


def hide_action_content(msg: ToolRequestMessage) -> ToolRequestMessage:
    return msg.model_copy(update={"content": None})


class SimpleAgentState(BaseModel):
    """Simple bucket for an Agent to access tools and store messages."""

    model_config = ConfigDict(extra="forbid")

    tools: list[Tool] = Field(default_factory=list)
    messages: list[ToolRequestMessage | ToolResponseMessage | Message] = Field(
        default_factory=list
    )
    hide_old_env_states: bool = Field(
        default=False,
        description="Whether to hide old EnvStateMessages.",
    )
    hide_old_action_content: bool = Field(
        default=False,
        description="If True, will hide the content of old ToolRequestMessages.",
    )

    total_cost: float = Field(
        default=0.0,
        description="Total cost of API calls made by this agent.",
    )

    total_tokens: dict = Field(
        default_factory=lambda: {"input": 0, "output": 0},
        description="Total tokens used by this agent.",
    )

    def get_next_state(
        self,
        obs: list[Message] | None = None,
        tools: list[Tool] | None = None,
        hide_old_env_states: bool | None = None,
        **kwargs,
    ) -> Self:
        """
        Get the next agent state without mutating the optional prior state.

        Do not mutate self here, just read from it.

        Args:
            obs: Optional observation messages to use in creating the next state.
            tools: Optional list of tools available to the agent. If unspecified, these
                should be pulled from the prior_state.
            hide_old_env_states: Optional override of self.hide_old_env_states.
                TODO: do we still need this override?
            kwargs: Additional keyword arguments to pass to this class's constructor.

        Returns:
            The next agent state (which is not an in-place change to self).
        """
        old_messages = self.messages

        hide_old_env_states = (
            hide_old_env_states
            if hide_old_env_states is not None
            else self.hide_old_env_states
        )
        if hide_old_env_states:
            old_messages = [
                HiddenEnvStateMessage() if isinstance(m, EnvStateMessage) else m
                for m in old_messages
            ]
        if self.hide_old_action_content:
            old_messages = [
                hide_action_content(m) if isinstance(m, ToolRequestMessage) else m
                for m in old_messages
            ]

        return type(self)(
            tools=tools if tools is not None else self.tools,
            messages=old_messages + (obs or []),
            hide_old_env_states=hide_old_env_states,
            hide_old_action_content=self.hide_old_action_content,
            **kwargs,
        )


class SimpleAgent(BaseModel, Agent[SimpleAgentState]):
    """Simple agent that can pick and invoke tools with a language model."""

    # Freeze to ensure the only mutation happens in either the agent state (which is
    # passed around) or in the internal Ops
    model_config = ConfigDict(frozen=True)

    llm_model: dict[str, Any] = Field(
        default={
            "model": CommonLLMNames.GPT_4O.value,
            "temperature": 0.1,
            "timeout": DEFAULT_LLM_COMPLETION_TIMEOUT,
        },
        description="Starting configuration for the LLM model. Trainable.",
    )
    sys_prompt: str | None = Field(
        default=None,
        description=(
            "Opt-in system prompt. If one is passed, the system prompt is not set up to"
            " be trainable, because this class is meant to be quite simple as far as"
            " possible hyperparameters."
        ),
    )

    hide_old_env_states: bool = Field(
        default=False,
        description="See SimpleAgentState.hide_old_env_states.",
    )

    hide_old_action_content: bool = Field(
        default=False,
        description="See SimpleAgentState.hide_old_action_content.",
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._config_op = ConfigOp[dict](config=self.llm_model)
        self._llm_call_op = LLMCallOp()
        self._cost_calculator = CostCalculator()

    async def init_state(self, tools: list[Tool]) -> SimpleAgentState:
        return SimpleAgentState(
            tools=tools,
            hide_old_env_states=self.hide_old_env_states,
            hide_old_action_content=self.hide_old_action_content,
        )

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(obs)

        messages = (
            prepend_sys(next_state.messages, sys_content=self.sys_prompt)
            if self.sys_prompt is not None
            else next_state.messages
        )

        # Calculate input tokens
        input_tokens = self._cost_calculator.num_tokens_from_messages(
            messages, 
            self.llm_model["model"]
        )
        next_state.total_tokens["input"] += input_tokens

        result = cast(
            OpResult[ToolRequestMessage],
            await self._llm_call_op(
                await self._config_op(), msgs=messages, tools=next_state.tools
            ),
        )

        # Calculate output tokens
        output_tokens = self._cost_calculator.num_tokens_from_messages(
            [result.value], 
            self.llm_model["model"]
        )
        next_state.total_tokens["output"] += output_tokens

         #Calculate cost
        call_cost = self._cost_calculator.calculate_cost(
            input_tokens,
            output_tokens,
            self.llm_model["model"]
        )
        next_state.total_cost += call_cost
        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, call_cost


class NoToolsSimpleAgent(SimpleAgent):
    """Agent whose action is parsed out of a LLM prompt without tool schemae."""

    def __init__(
        self,
        cast_tool_request: (
            Callable[[Message], ToolRequestMessage]
            | Callable[[Message], Awaitable[ToolRequestMessage]]
        ),
        **kwargs,
    ):
        """Initialize.

        Args:
            cast_tool_request: Function that is given a plain text message and produces
                a tool request message.
            **kwargs: Keyword arguments to pass to SimpleAgent's constructor.
        """
        super().__init__(**kwargs)
        self._cast_tool_request_op = FxnOp[ToolRequestMessage](cast_tool_request)
        self._cost_calculator = CostCalculator()

    @compute_graph()
    async def get_asv(
        self, agent_state: SimpleAgentState, obs: list[Message]
    ) -> tuple[OpResult[ToolRequestMessage], SimpleAgentState, float]:
        next_state = agent_state.get_next_state(obs)

        messages = (
            prepend_sys(next_state.messages, sys_content=self.sys_prompt)
            if self.sys_prompt is not None
            else next_state.messages
        )
         # Calculate input tokens
        input_tokens = self._cost_calculator.num_tokens_from_messages(
            messages, 
            self.llm_model["model"]
        )
        next_state.total_tokens["input"] += input_tokens

        llm_result = await self._llm_call_op(await self._config_op(), msgs=messages)
        result = await self._cast_tool_request_op(llm_result)

        # Calculate output tokens
        output_tokens = self._cost_calculator.num_tokens_from_messages(
            [result.value], 
            self.llm_model["model"]
        )
        next_state.total_tokens["output"] += output_tokens
        
        # Calculate cost
        call_cost = self._cost_calculator.calculate_cost(
            input_tokens,
            output_tokens,
            self.llm_model["model"]
        )
        next_state.total_cost += call_cost

        next_state.messages = [*next_state.messages, result.value]
        return result, next_state, call_cost
