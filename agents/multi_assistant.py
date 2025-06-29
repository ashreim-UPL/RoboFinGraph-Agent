from abc import ABC, abstractmethod
from typing import Dict, List, Any
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager
from .finrobot_base import FinRobot
from .utils import Cache

class MultiAssistantBase(ABC):
    def __init__(
        self,
        agent_configs: List[str | Dict[str, Any]],
        group_config: Dict[str, Any] = {},
        llm_config: Dict[str, Any] = {},
        work_dir: str = "coding",
        **kwargs,
    ):
        self.agent_configs = agent_configs
        self.group_config = group_config
        self.llm_config = llm_config
        self.work_dir = work_dir
        self.agents = [
            FinRobot(
                agent_config=cfg,
                llm_config=llm_config,
                proxy=None,
                work_dir=work_dir,
            )
            for cfg in agent_configs
        ]
        self.user_proxy = UserProxyAgent(
            name="User_Proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={"work_dir": work_dir, "use_docker": False},
            **kwargs,
        )
        self.representative = self._get_representative()

    @abstractmethod
    def _get_representative(self):
        pass

    def chat(self, message: str, use_cache=False, **kwargs):
        with Cache.disk() as cache:
            self.user_proxy.initiate_chat(
                self.representative,
                message=message,
                cache=cache if use_cache else None,
                **kwargs,
            )
        print("Multi-agent chat finished. Resetting ...")
        self.reset()

    def reset(self):
        self.user_proxy.reset()
        for agent in self.agents:
            agent.reset()
        self.representative.reset()


class MultiAssistant(MultiAssistantBase):
    """
    Group Chat Workflow with multiple agents.
    """

    def _get_representative(self):

        def custom_speaker_selection_func(
            last_speaker: ConversableAgent, groupchat: GroupChat
        ):
            messages = groupchat.messages
            if len(messages) <= 1:
                return groupchat.agents[0]
            if last_speaker is self.user_proxy:
                return groupchat.agent_by_name(messages[-2]["name"])
            elif "tool_calls" in messages[-1] or messages[-1]["content"].endswith("TERMINATE"):
                return self.user_proxy
            else:
                return groupchat.next_agent(last_speaker, groupchat.agents[:-1])

        self.group_chat = GroupChat(
            self.agents + [self.user_proxy],
            messages=[],
            speaker_selection_method=custom_speaker_selection_func,
            send_introductions=True,
        )
        manager_name = (self.group_config.get("name", "") + "_chat_manager").strip("_")
        manager = GroupChatManager(
            self.group_chat, name=manager_name, llm_config=self.llm_config
        )
        return manager


class MultiAssistantWithLeader(MultiAssistantBase):
    """
    Delegation workflow with a leader agent coordinating others.
    """

    def __init__(
        self,
        agent_configs: List[str | Dict[str, Any]],
        leader_config: str | Dict[str, Any],
        group_config: Dict[str, Any] = {},
        llm_config: Dict[str, Any] = {},
        work_dir: str = "coding",
        **kwargs,
    ):
        self.leader_config = leader_config
        super().__init__(agent_configs, group_config, llm_config, work_dir, **kwargs)

    def _get_representative(self):
        self.leader = FinRobot(
            self.leader_config,
            llm_config=self.llm_config,
            proxy=self.user_proxy,
            work_dir=self.work_dir,
        )
        for agent in self.agents:
            self.leader.register_nested_chats(
                [
                    {
                        "sender": self.leader,
                        "recipient": agent,
                        "message": "Please assist with the following task...",
                        "summary_method": "last_msg",
                        "silent": False,
                    }
                ]
            )
        return self.leader
