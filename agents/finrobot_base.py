# agents/finrobot_base.py
from autogen import AssistantAgent, UserProxyAgent
from typing import List, Callable, Dict, Any
from tools.toolkit_loader import register_toolkits
from prompts.prompt_templates import role_system_message, leader_system_message

# Assuming 'library' is defined in another file, e.g., 'agent_library.py'
# If not, you'll need to define it or import it.
# For now, let's add a placeholder.
library = {} 

class FinRobot(AssistantAgent):

    def __init__(
        self,
        agent_config: str | Dict[str, Any],
        work_dir: str ='coding',
        system_message: str | None = None,
        toolkits: List[Callable | dict | type] = [],
        proxy: UserProxyAgent | None = None,
        **kwargs,
    ):
        orig_name = ""
        if isinstance(agent_config, str):
            orig_name = agent_config
            name = orig_name.replace("_Shadow", "")
            # This line requires 'library' to be populated from your agent configurations
            if name in library:
                agent_config = library[name]
            else:
                # Fallback if the agent is not in the library
                agent_config = {"name": name, "description": "A financial robot.", "toolkits": []}


        agent_config = self._preprocess_config(agent_config)

        assert agent_config, f"agent_config is required."
        assert agent_config.get("name", ""), f"name needs to be in config."

        name = orig_name if orig_name else agent_config["name"]
        default_system_message = agent_config.get("profile", None)
        default_toolkits = agent_config.get("toolkits", [])

        system_message = system_message or default_system_message
        self.toolkits = toolkits or default_toolkits

        name = name.replace(" ", "_").strip()

        super().__init__(
            name, system_message, description=agent_config["description"], **kwargs
        )

        if proxy is not None:
            self.register_proxy(proxy)

    def _preprocess_config(self, config):
        role_prompt, leader_prompt, responsibilities = "", "", ""

        if "responsibilities" in config:
            title = config.get("title", config.get("name", ""))
            if "name" not in config:
                config["name"] = title
            responsibilities = config["responsibilities"]
            responsibilities = (
                "\n".join([f" - {r}" for r in responsibilities])
                if isinstance(responsibilities, list)
                else responsibilities
            )
            role_prompt = role_system_message.format(
                title=title,
                responsibilities=responsibilities,
            )

        name = config.get("name", "")
        description = (
            f"Name: {name}\nResponsibility:\n{responsibilities}"
            if responsibilities
            else f"Name: {name}"
        )
        config["description"] = description.strip()

        if "group_desc" in config:
            group_desc = config["group_desc"]
            leader_prompt = leader_system_message.format(group_desc=group_desc)

        config["profile"] = (
            (role_prompt + "\n\n").strip()
            + (leader_prompt + "\n\n").strip()
            + config.get("profile", "")
        ).strip()

        return config


    def register_proxy(self, proxy):
        if not hasattr(proxy, "code_execution_config"):
            proxy.code_execution_config = {"use_docker": False}
        register_toolkits(self.toolkits, self, proxy)
