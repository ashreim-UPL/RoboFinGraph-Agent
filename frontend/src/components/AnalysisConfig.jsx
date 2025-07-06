import React from "react";
import IconKey from "./icons/IconKey";
import { formatAgentRoleKey } from "../utils/formatters";

/**
 * Configuration panel for company, year, report type, verbose mode,
 * and global LLM provider selection.
 */
const AnalysisConfig = ({
  company, setCompany,
  year, setYear,
  reportType, setReportType,
  selectedGlobalProvider, setSelectedGlobalProvider,
  availableProviders,
  isRunning, onStart,
  verbose, setVerbose
}) => {
  const providerOptions = Object.keys(availableProviders).map(key => ({
    key: key,
    name: formatAgentRoleKey(key.replace(/_llm_models$/, ''))
  }));

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <div className="flex items-center font-semibold mb-2">
        <IconKey />
        Analysis Configuration
      </div>
      <div className="mb-2">
        <label className="block text-xs text-gray-500">Company Name or Ticker</label>
        <input
          className="border px-2 py-1 rounded w-full"
          value={company}
          onChange={e => setCompany(e.target.value)}
          disabled={isRunning}
        />
      </div>
      <div className="flex gap-2 mb-2">
        <div className="flex-1">
          <label className="block text-xs text-gray-500">Year</label>
          <select
            className="border px-2 py-1 rounded w-full"
            value={year}
            onChange={e => setYear(e.target.value)}
            disabled={isRunning}
          >
            {[2025,2024,2023,2022,2021].map(y => (
              <option key={y} value={y}>{y}</option>
            ))}
          </select>
        </div>
        <div className="flex-1">
          <label className="block text-xs text-gray-500">Report Type</label>
          <select
            className="border px-2 py-1 rounded w-full"
            value={reportType}
            onChange={e => setReportType(e.target.value)}
            disabled={isRunning}
          >
            <option value="kpi_bullet_insights">KPI Insights</option>
          </select>
        </div>
      </div>
      <div className="mb-3">
        <label className="inline-flex items-center">
          <input
            type="checkbox"
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
            checked={verbose}
            onChange={e => setVerbose(e.target.checked)}
            disabled={isRunning}
          />
          <span className="ml-2 text-sm">Verbose (debug logging)</span>
        </label>
      </div>

      {/* Global LLM Provider Selection (Single Dropdown) */}
      <div className="mb-3">
        <label htmlFor="global-provider-select" className="block text-xs text-gray-500 mb-2">
          Select Main LLM Provider for all Agents
        </label>
        <select
          id="global-provider-select"
          className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
          value={selectedGlobalProvider || ''}
          onChange={(e) => setSelectedGlobalProvider(e.target.value)}
          disabled={isRunning || !providerOptions.length}
        >
          {!providerOptions.length && <option value="">No providers available</option>}
          {providerOptions.map(provider => (
            <option key={provider.key} value={provider.key}>
              {provider.name}
            </option>
          ))}
        </select>
      </div>

      {/* Displaying selected LLM Models per agent (read-only) */}
      {selectedGlobalProvider && availableProviders[selectedGlobalProvider] && (
        <div className="mb-4 p-3 bg-gray-50 rounded-md border border-gray-200">
          <h4 className="font-semibold text-sm mb-2 text-gray-700">
            Selected Models from {providerOptions.find(p => p.key === selectedGlobalProvider)?.name || 'Provider'}:
          </h4>
          <div className="overflow-x-auto">
            <table className="table-auto w-full text-xs border border-gray-200">
              <tbody>
                {Object.entries(availableProviders[selectedGlobalProvider]).map(([agentRoleKey, modelName]) => (
                  <tr key={agentRoleKey} className="hover:bg-gray-100">
                    <td className="px-2 py-1 font-medium text-gray-800 capitalize whitespace-nowrap border-b border-gray-100">
                      {formatAgentRoleKey(agentRoleKey)}
                    </td>
                    <td className="px-2 py-1 text-gray-600 border-b border-gray-100 break-all">
                      {modelName}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Disable if no provider selected */}
      <button
        className="w-full py-2 bg-indigo-600 text-white rounded font-semibold flex items-center justify-center"
        onClick={onStart}
        disabled={isRunning || !selectedGlobalProvider} // Disable if no provider selected
      >
        {isRunning ? <><IconSpinner /> Processing...</> : "Start Analysis"}
      </button>
    </div>
  );
};
export default AnalysisConfig;
