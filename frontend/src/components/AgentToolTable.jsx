import React from "react";

const AgentToolTable = ({ agentSteps }) => (
  <div className="bg-white rounded-lg shadow p-4 mb-4">
    <div className="font-semibold mb-2">Agent & Tool Execution</div>
    <table className="min-w-full text-xs">
      <thead>
        <tr className="border-b">
          <th className="text-left py-1 px-2">AGENT</th>
          <th className="text-left py-1 px-2">TOOL</th>
          <th className="text-left py-1 px-2">STATUS</th>
          <th className="text-left py-1 px-2">DURATION</th>
        </tr>
      </thead>
      <tbody>
        {agentSteps.length === 0 ? (
          <tr>
            <td colSpan={4} className="py-2 text-gray-400 text-center">
              No agent steps yet.
            </td>
          </tr>
        ) : (
          agentSteps.map((step, i) => (
            <tr key={i} className="border-b last:border-0">
              <td className="py-1 px-2">{step.agentName}</td>
              <td className="py-1 px-2">{step.toolName}</td>
              <td className="py-1 px-2">
                {step.status === "Running" && (
                  <span className="text-blue-600">Running</span>
                )}
                {step.status === "Completed" && (
                  <span className="text-green-600">Completed</span>
                )}
                {step.status === "Error" && (
                  <span className="text-red-600">Error</span>
                )}
              </td>
              <td className="py-1 px-2">
                {step.duration ? `${step.duration.toFixed(0)} ms` : "..."}
              </td>
            </tr>
          ))
        )}
      </tbody>
    </table>
  </div>
);

export default AgentToolTable;
