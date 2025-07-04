import sys, json
from datetime import datetime
from functools import wraps
from agents.state_types import NodeStatus, AgentState

def record_node(node_key: str):
    """
    Wraps a node func(agent_state), emits start/end events, captures timing,
    errors, token & cost metrics (piped in via LangGraphLLMExecutor), and
    pushes a unified record into agent_state.memory['pipeline_data'].
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(agent_state: AgentState, *args, **kwargs):
            # Initialize node‐record
            record = {
                "node": node_key,
                "status": NodeStatus.RUNNING,
                "start_time": None,
                "end_time": None,
                "duration": None,
                "cost_llm": 0.0,
                "tokens_sent": 0,
                "tokens_generated": 0,
                "tools_used": [],
                "errors": [],
                "files_read": [],
                "files_created": [],
                "custom_metrics": {}
            }

            # Mark start
            start = datetime.utcnow()
            record["start_time"] = start.isoformat()
            sys.stdout.write(json.dumps({
                "event_type":"node_start",
                "data":{"node":node_key,"timestamp":record["start_time"]}
            }) + "\n")
            sys.stdout.flush()

            try:
                # Give the LLM–executor hooks a way to record into this record
                setattr(agent_state, "_current_node_record", record)

                result = fn(agent_state, *args, **kwargs)
                record["status"] = NodeStatus.SUCCESS

            except Exception as e:
                record["status"] = NodeStatus.ERROR
                record["errors"].append(str(e))
                agent_state.error_log.append(f"{node_key}: {e}")
                raise

            finally:
                # Finalize timing
                end = datetime.utcnow()
                record["end_time"]  = end.isoformat()
                record["duration"]  = (end-start).total_seconds()

                # Clean up hook
                if hasattr(agent_state, "_current_node_record"):
                    delattr(agent_state, "_current_node_record")

                # Append to pipeline history
                pipeline = agent_state.memory.setdefault("pipeline_data", [])
                pipeline.append(record)

                # Emit end event
                sys.stdout.write(json.dumps({
                    "event_type":"node_end",
                    "data":record
                }) + "\n")
                sys.stdout.flush()

            return result
        return wrapper
    return decorator


class LangGraphLLMExecutor:
    def __init__(self, node_name: str):
        provider, model_name = _get_node_config(node_name)
        self.llm = _init_llm(provider, model_name)
        self.provider = provider
        self.model_name = model_name

    def generate(self, messages, agent_state: AgentState):
        # 1) Append to global messages
        for msg in messages:
            agent_state.messages.append({"role": msg.type, "content": msg.content})

        # 2) Invoke
        if self.provider == "openai":
            with get_openai_callback() as cb:
                response = self.llm(messages)
            prompt_tokens    = cb.prompt_tokens
            completion_tokens= cb.completion_tokens
            cost             = cb.total_cost
        else:
            response = self.llm(messages)
            usage = getattr(response, "usage", {}) or {}
            prompt_tokens     = usage.get("prompt_tokens",0)
            completion_tokens = usage.get("completion_tokens",0)
            cost              = 0.0  # or extract if available

        # 3) Push into current node record if present
        rec = getattr(agent_state, "_current_node_record", None)
        if rec is not None:
            rec["tokens_sent"]      += prompt_tokens
            rec["tokens_generated"] += completion_tokens
            rec["cost_llm"]         += cost
            rec["tools_used"].append(f"{self.provider}:{self.model_name}")

        # 4) Append LLM reply
        agent_state.messages.append({"role":"assistant","content":response.content})
        return response


@record_node("summarization")
def summarization_node(agent_state: AgentState) -> AgentState:
    executor = LangGraphLLMExecutor("summarizer")
    resp = executor.generate([HumanMessage(content=...)], agent_state)
    agent_state.memory["summaries"] = resp.content
    # If you want custom metrics:
    rec = agent_state.memory["pipeline_data"][-1]
    rec["custom_metrics"]["sections_processed"] = len(...)
    return agent_state
