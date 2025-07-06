export const SETUP_STEPS_LIST = [
  { key: "api", label: "Setting up APIs" },
  { key: "llm", label: "Setting up LLM Model" },
  { key: "agents", label: "Setting up Agents" },
  { key: "tools", label: "Setting up Tools" },
  { key: "region", label: "Resolving Company Region & Peers" },
];

export const STATUS_COLOR = {
  pending: "bg-yellow-400",
  success: "bg-green-500",
  error: "bg-red-500",
};

export const AGENT_ROLES_FOR_MODEL_ASSIGNMENT = [
  "resolve_company",
  "llm_decision",
  "data_collection_us",
  "data_collection_indian",
  "validate_collected_data",
  "synchronize_data",
  "summarizer",
  "validate_summarized_data",
  "concept",
  "validate_analyzed_data",
  "thesis",
  "audit"
];
