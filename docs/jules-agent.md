# Google Jules Integration Design Spec

## 1. Overview and Goal
This document outlines the architecture and implementation plan for integrating Google Jules—an autonomous, asynchronous coding agent—into the Omnigent ecosystem. The integration allows Omnigent to dispatch Jules as a background worker seamlessly, leveraging Omnigent's existing asynchronous lifecycle and harness architecture.

## 2. Omnigent Architectural Context
Omnigent enforces a strict separation of concerns that makes integrating autonomous agents straightforward if the boundaries are respected:

*   **`AgentSpec` (Logical):** Defines the identity of an agent (its system prompt, tools, metadata) via configuration (e.g., `default_policies.yaml`).
*   **`Executor` (Runtime):** The component that actually communicates with the underlying LLM or Agent SDK (e.g., `ClaudeNativeExecutor`, `CursorExecutor`).
*   **Harness Process:** Executors run in isolated background processes (harnesses) communicating with the main Omnigent server over Unix domain sockets using HTTP/SSE.
*   **Sub-agent Lifecycle & Asynchrony:** Omnigent handles asynchronous execution at the workflow layer via `sys_session_send`. When a parent agent delegates a task to Jules via `sys_session_send`, the system spawns a background conversation and invokes the Jules harness. The parent does *not* block. When Jules finishes, Omnigent routes the `TurnComplete` event to the parent's `async_inbox`. 

Because of this design, the **`JulesExecutor` does not need to implement custom background-threading for asynchrony**. It only needs to implement a standard, blocking `async def run_turn(...)` generator. Omnigent's outer orchestration handles the asynchronous lifecycle automatically.

## 3. Step-by-Step Implementation Guide

### Step 1: Create the `JulesExecutor` (`omnigent/inner/jules_executor.py`)
Create a new executor class that bridges Jules' native SDK events into Omnigent's `ExecutorEvent` streaming protocol. 

If Jules runs its own agent loop, it must route tool execution requests (e.g., running bash commands, reading files) back to Omnigent using the injected `_tool_executor` callback to ensure it runs inside Omnigent's secure sandboxes.

```python
import asyncio
from typing import Any, AsyncIterator
from omnigent.inner.executor import (
    Executor, ExecutorEvent, TextChunk, ReasoningChunk, 
    ToolCallRequest, TurnComplete, ExecutorError, ExecutorConfig
)

class JulesExecutor(Executor):
    def __init__(self, model: str | None, os_env: Any, **kwargs):
        self.model = model
        self.os_env = os_env
        # _tool_executor is injected by ExecutorAdapter after initialization.
        # It provides the callback to run Omnigent tools.
        self._tool_executor = None 

    def handles_tools_internally(self) -> bool:
        """
        Return True if Jules runs its own loop and calls tools in-process,
        rather than yielding control back to Omnigent to manage the loop.
        """
        return True

    async def run_turn(
        self, messages: list[dict], tools: list[dict], system_prompt: str, config: ExecutorConfig | None = None
    ) -> AsyncIterator[ExecutorEvent]:
        
        try:
            # 1. Start the Jules agent using the accumulated conversation history
            # (Theoretical Jules SDK call)
            jules_stream = await jules_sdk.run_agent_stream(
                messages=messages, 
                system_instruction=system_prompt,
                tools=self._map_tools_to_jules(tools)
            )

            # 2. Bridge Jules' internal events to Omnigent's UI streams
            async for event in jules_stream:
                if event.type == "text":
                    yield TextChunk(text=event.delta)
                elif event.type == "thought":
                    yield ReasoningChunk(text=event.delta)
                elif event.type == "tool_call":
                    # If Jules decides to run a tool, execute it via Omnigent's context
                    if self._tool_executor:
                        yield ToolCallRequest(tool_name=event.name, arguments=event.args)
                        
                        # In-process bridge to Omnigent's environment/sandbox
                        tool_result = await self._tool_executor(event.name, event.args)
                        
                        # Feed the result back into Jules' active stream
                        await jules_stream.provide_tool_result(event.call_id, tool_result)

            # 3. Complete the turn
            final_result = await jules_stream.get_final_response()
            yield TurnComplete(result=final_result)

        except asyncio.CancelledError:
            yield TurnCancelled()
            raise
        except Exception as e:
            yield ExecutorError(message=str(e))
```

### Step 2: Build the Harness Entrypoint (`omnigent/inner/jules_harness.py`)
This module provides the factory function required by Omnigent's `HarnessProcessManager`. The `ExecutorAdapter` automatically wraps the `JulesExecutor` into an HTTP service.

```python
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter
from omnigent.inner.jules_executor import JulesExecutor
import os

def _build_jules_executor() -> JulesExecutor:
    """
    Construct a JulesExecutor from env-var config.
    Called lazily by the ExecutorAdapter on the first turn.
    """
    return JulesExecutor(
        model=os.environ.get("HARNESS_JULES_MODEL"),
        # Extract and parse HARNESS_JULES_OS_ENV if needed
        os_env=None 
    )

def create_app():
    """Build the Jules harness's FastAPI app (required entry point)."""
    adapter = ExecutorAdapter(executor_factory=_build_jules_executor)
    return adapter.build()
```

### Step 3: Register the Harness
Omnigent needs to know where to find the `jules` harness module. Register it in the `_HARNESS_MODULES` dictionary.

**File:** `omnigent/runtime/harnesses/__init__.py`
```python
_HARNESS_MODULES: dict[str, str] = {
    # ... existing harnesses ...
    "cursor": "omnigent.inner.cursor_harness",
    
    # Register Jules
    "jules": "omnigent.inner.jules_harness",
}
```

### Step 4: Define the Jules Agent in Policies
With the backend plumbing established, expose Jules to the workspace via an `AgentSpec` declaration in `default_policies.yaml` (or a project-level configuration).

**File:** `omnigent/resources/default_policies.yaml` (or equivalent)
```yaml
agents:
  jules:
    description: "Autonomous Google Jules coding agent."
    # Enabling async grants Jules the `sys_session_send`, `sys_read_inbox`, 
    # and `sys_cancel_task` builtins, allowing Jules to dispatch sub-agents of its own.
    async: true 
    executor:
      # Use the base omnigent executor protocol but route to the custom harness
      type: "omnigent"
      harness: "jules"
      model: "jules-v1-flash" # Default model to pass to the executor
```

## 4. Critical Architectural Considerations

### First-Party Tool Parity
By routing tool executions through `self._tool_executor` in `JulesExecutor`, Jules gains full first-party parity within Omnigent. This means Jules can execute standard tools (Bash, Read, Write) inside Omnigent's sandboxes and can even invoke `sys_session_send` to orchestrate further sub-agents, adhering to workspace policies seamlessly.

### Lazy Dependencies
All `import jules_sdk` or heavy dependency initializations must occur inside the `_build_jules_executor` factory or within the `JulesExecutor` itself. This prevents Omnigent from crashing during boot or process launching if a user does not have the optional Jules SDK installed in their environment.

### Approvals and Interruptions
If Jules reaches a state requiring user approval (e.g., executing a destructive command or pushing a PR), it should yield an `ApprovalEvent` (if supported by the updated executor protocol) or gracefully pause execution. Omnigent's core CLI will handle halting, prompting the user, and resuming the executor based on the user's input. 

### Cancellation
The `JulesExecutor.run_turn` must respect `asyncio.CancelledError`. If a user types `Ctrl+C` or a parent agent calls `sys_cancel_task` on Jules, Omnigent's runner will cancel the task driving `run_turn`. The executor must catch this, clean up the underlying Jules SDK process, yield `TurnCancelled()`, and re-raise.