import React from "react";
import IconPipeline from "./icons/IconPipeline";

const PipelineKPIs = ({ kpis, reportUrl }) => {
  const toolSuccess =
    kpis.tool_calls > 0
      ? `${kpis.successful_tool_calls}/${kpis.tool_calls}`
      : "N/A";

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <div className="flex items-center font-semibold mb-2">
        <IconPipeline />
        Pipeline KPIs
      </div>
      <div className="text-xs">
        <div className="flex justify-between">
          <span>End-to-End Latency:</span>
          <span>{kpis.latency.toFixed(2)}s</span>
        </div>
        <div className="flex justify-between">
          <span>Total LLM Tokens:</span>
          <span>{kpis.tokens}</span>
        </div>
        <div className="flex justify-between">
          <span>Estimated LLM Cost:</span>
          <span>${kpis.cost.toFixed(5)}</span>
        </div>
        <div className="flex justify-between">
          <span>Tool Call Success:</span>
          <span>{toolSuccess}</span>
        </div>
        <div className="flex justify-between">
          <span>Pipeline Errors:</span>
          <span>{kpis.errors}</span>
        </div>
        <div className="flex justify-between mt-2 text-gray-400">
          <span>LLM-as-Judge Metrics (Post-run)</span>
        </div>
        <div className="flex justify-between">
          <span>Answer Relevance:</span>
          <span>{kpis.relevance ?? "N/A"}</span>
        </div>
        <div className="flex justify-between">
          <span>Faithfulness:</span>
          <span>{kpis.faithfulness ?? "N/A"}</span>
        </div>
      </div>
      {reportUrl && (
        <a
          href={reportUrl}
          className="mt-3 block py-2 bg-green-600 text-white text-center rounded font-semibold hover:bg-green-700"
          download
        >
          Download Report
        </a>
      )}
    </div>
  );
};

export default PipelineKPIs;
