# Finrobot

An advanced agentic workflow system.
# FinRobot Multi-Agent Orchestration (LangGraph + LLMs)

## Overview
This project orchestrates a multi-agent pipeline using LangGraph, OpenAI, and financial APIs. It mimics how financial analysts collect, summarize, and audit investment reports.

## Agents
- **ExpertInvestorUS/India**: Oversees orchestration.
- **DataCOT (US/India)**: Collects financial data.
- **ConceptCOT**: Summarizes collected data.
- **ThesisCOT**: Builds the final report.
- **Auditor (LLM/Non-LLM)**: Evaluates logic, cost, hallucinations.
- **IOAgent**: Manages file I/O.

## Run the Graph
```bash
python main.py

