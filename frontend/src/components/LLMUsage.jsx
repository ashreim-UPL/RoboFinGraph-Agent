import React from "react";
import IconLLM from "./icons/IconLLM";

const LLMUsage = ({ llmCalls }) => (
  <div className="bg-white rounded-lg shadow p-4 mb-4">
    <div className="flex items-center font-semibold mb-2">
      <IconLLM />
      LLM Usage
    </div>
    <div className="text-xs text-gray-500">
      {llmCalls === 0
        ? "No LLM calls yet."
        : `${llmCalls} LLM call${llmCalls > 1 ? "s" : ""} made.`}
    </div>
  </div>
);

export default LLMUsage;
