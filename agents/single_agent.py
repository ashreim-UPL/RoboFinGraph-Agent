# single_agent.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from collections import defaultdict
from autogen import ConversableAgent, UserProxyAgent
from agents.finrobot_base import FinRobot
from .utils import Cache
from tools.rag_tools import get_rag_function, register_function
from .triggers import trigger_on_file_save_confirmation
from prompts.prompt_templates import instruction_message

class SingleAssistantBase(ABC):
    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] = {},
        work_dir: str = "coding",
    ):
        self.assistant = FinRobot(
            agent_config=agent_config,
            llm_config=llm_config,
            proxy=None,
            work_dir=work_dir,
        )

    @abstractmethod
    def chat(self, message: str, use_cache=False, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass


class SingleAssistant(SingleAssistantBase):
    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] = {},
        is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        **kwargs,
    ):
        work_dir_for_run = code_execution_config.get("work_dir", "coding")
        super().__init__(agent_config, llm_config=llm_config, work_dir=work_dir_for_run)
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        self.assistant.register_proxy(self.user_proxy)

    def summarize(self, prompt, **kwargs):
        """
        Run a single-shot summarization using the agent’s preferred LLM backend.
        Returns the assistant's reply as a string.
        """

        # Initiate chat with the assistant
        chat_result = self.user_proxy.initiate_chat(self.assistant, message=prompt, **kwargs)
    
        if chat_result:
            return chat_result
        return None
        
    def chat(self, message: str, use_cache=False, **kwargs):
        with Cache.disk() as cache:
            self.user_proxy.initiate_chat(
                self.assistant,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )
        print("Current chat finished. Resetting agents ...")
        self.reset()

    def reset(self):
        self.user_proxy.reset()
        self.assistant.reset()


class SingleAssistantRAG(SingleAssistant):
    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] = {},
        is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        retrieve_config={},
        rag_description="",
        **kwargs,
    ):
        super().__init__(
            agent_config,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        assert retrieve_config, "retrieve config cannot be empty for RAG Agent."
        rag_func, rag_assistant = get_rag_function(retrieve_config, rag_description)
        self.rag_assistant = rag_assistant
        register_function(
            rag_func,
            caller=self.assistant,
            executor=self.user_proxy,
            description=rag_description or rag_func.__doc__,
        )

    def reset(self):
        super().reset()
        self.rag_assistant.reset()


class SingleAssistantShadow(SingleAssistant):
    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        llm_config: Dict[str, Any] = {},
        is_termination_msg=lambda x: x.get("content", "").endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=10,
        code_execution_config={"work_dir": "coding", "use_docker": False},
        **kwargs,
    ):
        super().__init__(
            agent_config,
            llm_config=llm_config,
            is_termination_msg=is_termination_msg,
            human_input_mode=human_input_mode,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            code_execution_config=code_execution_config,
            **kwargs,
        )
        if isinstance(agent_config, dict):
            agent_config_shadow = agent_config.copy()
            agent_config_shadow["name"] = agent_config["name"] + "_Shadow"
            agent_config_shadow["toolkits"] = []
        else:
            agent_config_shadow = agent_config + "_Shadow"

        self.assistant_shadow = FinRobot(
            agent_config_shadow,
            toolkits=[],
            llm_config=llm_config,
            proxy=None,
            work_dir=code_execution_config.get("work_dir", "coding"),
        )
        self.assistant.register_nested_chats(
            [
                {
                    "sender": self.assistant,
                    "recipient": self.assistant_shadow,
                    "message": instruction_message,
                    "summary_method": "last_msg",
                    "max_turns": 2,
                    "silent": True,
                }
            ],
            trigger=trigger_on_file_save_confirmation,
        )