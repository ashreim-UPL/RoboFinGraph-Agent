import React from "react";
import IconAgent from "./icons/IconAgent";

function AgentSetupViewer({ region, agentSetup }) {
  if (!agentSetup || !agentSetup.leader || !agentSetup.agents) {
    return (
      <div className="bg-white rounded-lg shadow p-4 mb-4">
        <div className="flex items-center font-semibold mb-2">
          <IconAgent />
          Agent Setup: {region === "IN" ? "India" : "US"}
        </div>
        <div className="text-gray-400 text-xs">Loading agent details...</div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4 max-h-[400px] overflow-y-auto">
      <div className="flex items-center font-semibold mb-3">
        <IconAgent />
        <h2 className="text-lg font-semibold">Agent Setup - {region === "IN" ? "India" : "US"}</h2>
      </div>

      <section className="mb-4">
        <h3 className="font-semibold text-md mb-2 flex items-center">
          <svg className="h-4 w-4 text-yellow-500 mr-1" fill="none" viewBox="0 0 24 24">
            <circle cx="12" cy="12" r="8" stroke="currentColor" strokeWidth="1.5" />
            <path d="M12 8V12M12 12L14 14" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          Leader
        </h3>
        <div className="ml-5">
          <p className="font-semibold">{agentSetup.leader.title}</p>
          <ul className="list-disc list-inside text-sm text-gray-700">
            {agentSetup.leader.responsibilities.map((resp, idx) => (
              <li key={idx}>{resp}</li>
            ))}
          </ul>
        </div>
      </section>

      <section>
        <h3 className="font-semibold text-md mb-2 flex items-center">
          <svg className="h-4 w-4 text-gray-500 mr-1" fill="none" viewBox="0 0 24 24">
            <rect x="4" y="4" width="16" height="16" rx="2" stroke="currentColor" strokeWidth="1.5" />
            <path d="M8 12H16M12 8V16" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
          Agents
        </h3>
        <div className="ml-5 space-y-3">
          {agentSetup.agents.map((agent, idx) => (
            <div key={idx}>
              <p className="font-semibold text-sm">{agent.title}</p>
              <ul className="list-disc list-inside text-sm text-gray-600">
                {agent.responsibilities.map((resp, i) => (
                  <li key={i}>{resp}</li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default AgentSetupViewer;
