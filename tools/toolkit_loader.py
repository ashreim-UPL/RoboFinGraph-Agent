# toolkit_loader.py

from langchain.agents import Tool

from typing import Callable, List, Dict, Any
from autogen import Agent, UserProxyAgent

# Audit Tools
from tools.audit_utils import calculate_token_cost, validate_pipeline_accuracy

# Financial Tools
from tools.financial_tools import (
    resolve_company_name,
    get_stock_data,
    get_filing_date,
    fetch_peers
)
# IO Tools
from tools.file_utils import (
    read_json_file,
    save_to_file,
    get_file_metadata,
    list_directory_files,
    read_text_file,
    get_file_size,
    read_image_metadata
)

# Optional: RAG Tool
# from tools.rag_loader import query_rag_api

# === IO Tools ===
def load_io_tools() -> List[Tool]:
    return [
        Tool.from_function(
            name="Read_JSON",
            func=read_json_file,
            description="Read and parse a JSON file by giving its full path."
        ),
        Tool.from_function(
            name="Read_Text",
            func=read_text_file,
            description="Read the contents of a text file by providing the file path."
        ),
        Tool.from_function(
            name="Save_File",
            func=save_to_file,
            description="Save data to a file. Provide filename and content."
        ),
        Tool.from_function(
            name="List_Directory",
            func=list_directory_files,
            description="List all files in a given directory."
        ),
        Tool.from_function(
            name="Get_File_Metadata",
            func=get_file_metadata,
            description="Get metadata (size, type, last modified) for a given file path."
        ),
        Tool.from_function(
            name="Get_File_Size",
            func=get_file_size,
            description="Return file size in bytes."
        ),
        Tool.from_function(
            name="Read_Image_Metadata",
            func=read_image_metadata,
            description="Returns width, height, and format of image files (e.g., PNGs)."
        ),
    ]

# === Financial Tools ===
def load_financial_tools() -> List[Tool]:
    return [
        Tool.from_function(name="Resolve_Company_Name", func=resolve_company_name, description="Fix typos and get company metadata."),
        Tool.from_function(name="Get_Stock_Data", func=get_stock_data, description="Fetch latest stock data for a company."),
        Tool.from_function(name="Get_Filing_Date", func=get_filing_date, description="Get the latest SEC filing date."),
        Tool.from_function(name="Fetch_Peers", func=fetch_peers, description="Get a list of similar companies or competitors.")
    ]

# === Audit Tools ===
def load_audit_tools() -> List[Tool]:
    return [
        Tool.from_function(name="Token_Cost_Estimator", func=calculate_token_cost, description="Estimate LLM token usage."),
        Tool.from_function(name="Pipeline_Validator", func=validate_pipeline_accuracy, description="Audit pipeline correctness and quality.")
    ]

# === Optional RAG Tool (uncomment if used) ===
# def load_rag_tools() -> List[Tool]:
#     return [
#         Tool.from_function(name="Query_RAG_Embedding", func=query_rag_api, description="Retrieve documents using RAG embedding search")
#     ]

# === Full Toolkit Loader ===
def load_all_tools() -> List[Tool]:
    return (
        load_io_tools() +
        load_financial_tools() +
        load_audit_tools()
        # + load_rag_tools()  # Uncomment if needed
    )

def register_toolkits(
    toolkits: List[Callable | Dict | type],
    caller: Agent,
    executor: UserProxyAgent
):
    """
    Registers a list of functions (tools) with a caller and an executor agent.

    Args:
        toolkits (List[Callable | Dict | type]): A list of functions or tools to register.
        caller (Agent): The agent that will be calling the tools.
        executor (UserProxyAgent): The agent that will be executing the tools.
    """
    if not toolkits:
        return

    for tool in toolkits:
        if isinstance(tool, Callable):
            caller.register_function(
                function_map={tool.__name__: tool},
                caller=caller,
                executor=executor,
            )
        elif isinstance(tool, dict):
            # Assuming the dict is a function map
            caller.register_function(
                function_map=tool,
                caller=caller,
                executor=executor,
            )