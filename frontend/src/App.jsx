import React, { useState, useEffect, useRef } from "react";
import AnalysisConfig from "./components/AnalysisConfig";
import LLMUsage from "./components/LLMUsage";
import PipelineKPIs from "./components/PipelineKPIs";
import AgentSetupViewer from "./components/AgentSetupViewer";
import AgentToolTable from "./components/AgentToolTable";
import RawEventLog from "./components/RawEventLog";
import GraphViewer from "./components/GraphViewer";
import LoggingSetupStatus from "./components/LoggingSetupStatus";
import IconSpinner from "./components/icons/IconSpinner";
import { AGENT_ROLES_FOR_MODEL_ASSIGNMENT } from "./utils/constants";

function App() {
  const closedByComplete = useRef(false);
  const [company, setCompany] = useState("Microsoft");
  const [year, setYear] = useState("2024");
  const [reportType, setReportType] = useState("kpi_bullet_insights");
  const [selectedGlobalProvider, setSelectedGlobalProvider] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const [reportUrl, setReportUrl] = useState(null);
  const [availableProviders, setAvailableProviders] = useState({});
  const [isLoadingModels, setIsLoadingModels] = useState(true);
  const [verbose, setVerbose] = useState(false);
  const [agentSetup, setAgentSetup] = useState(null);
  const [region, setRegion] = useState("US");
  const [setupStatus, setSetupStatus] = useState({
    api: "pending", llm: "pending", agents: "pending", tools: "pending", region: "pending",
  });
  const [agentSteps, setAgentSteps] = useState([]);
  const [log, setLog] = useState([]);
  const [llmCalls, setLlmCalls] = useState(0);
  const [kpis, setKpis] = useState({
    latency: 0, tokens: 0, cost: 0, tool_calls: 0, successful_tool_calls: 0,
    errors: 0, relevance: "N/A", faithfulness: "N/A"
  });
  const [mermaidGraphSyntax, setMermaidGraphSyntax] = useState(null);

  const eventSourceRef = useRef(null);

  // Fetch available LLM providers on mount
  useEffect(() => {
    let isMounted = true;
    setIsLoadingModels(true);
    setError(null);

    fetch('/available_models')
      .then(response => {
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        return response.json();
      })
      .then(data => {
        if (!isMounted) return;
        const receivedProviders = data.available_providers_llm || {};
        setAvailableProviders(receivedProviders);

        const firstAvailableProviderKey = Object.keys(receivedProviders)[0] || '';
        setSelectedGlobalProvider(firstAvailableProviderKey);
      })
      .catch(err => {
        if (!isMounted) return;
        setError("Failed to load available LLM providers from server. Check console for details.");
        setAvailableProviders({});
      })
      .finally(() => {
        if (isMounted) setIsLoadingModels(false);
      });

    return () => {
      isMounted = false;
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  // Start analysis and handle SSE events
  const startAnalysis = () => {
    if (!selectedGlobalProvider) {
      setError("Please select a main LLM provider to start the analysis.");
      return;
    }

    closedByComplete.current = false;
    setIsRunning(true);
    setError(null);
    setAgentSteps([]);
    setLog([]);
    setLlmCalls(0);
    setReportUrl(null);
    setAgentSetup(null);
    setMermaidGraphSyntax(null);
    setKpis({
      latency: 0, tokens: 0, cost: 0, tool_calls: 0, successful_tool_calls: 0,
      errors: 0, relevance: "N/A", faithfulness: "N/A"
    });
    setSetupStatus({
      api: "pending", llm: "pending", agents: "pending", tools: "pending", region: "pending",
    });

    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }

    const llmModelsToSend = {};
    const selectedProviderConfig = availableProviders[selectedGlobalProvider];

    if (selectedProviderConfig) {
      AGENT_ROLES_FOR_MODEL_ASSIGNMENT.forEach(role => {
        if (selectedProviderConfig[role]) {
          llmModelsToSend[role] = selectedProviderConfig[role];
        } else {
          llmModelsToSend[role] = "default-fallback-model";
        }
      });
    } else {
      setError(`Configuration for selected provider '${selectedGlobalProvider}' not found. Cannot start analysis.`);
      setIsRunning(false);
      return;
    }

    const llmModelsParam = encodeURIComponent(JSON.stringify(llmModelsToSend));

    const eventSource = new window.EventSource(
      `/stream?company=${encodeURIComponent(company)}&year=${year}&report_type=${reportType}&llm_models=${llmModelsParam}&verbose=${verbose}`
    );
    eventSourceRef.current = eventSource;

    eventSource.onmessage = (event) => {
      try {
        const eventData = JSON.parse(event.data);
        setLog(prev => [...prev, event.data]);

        switch(eventData.event_type) {
          case 'setup_progress':
            if (eventData.data && eventData.data.step) {
              let statusKey = '';
              if (eventData.data.step.includes("Setting up APIs")) statusKey = "api";
              else if (eventData.data.step.includes("Setting up LLM Model")) statusKey = "llm";
              else if (eventData.data.step.includes("Setting up Agents")) statusKey = "agents";
              else if (eventData.data.step.includes("Setting up Tools")) statusKey = "tools";
              if (statusKey) {
                setSetupStatus(prev => ({ ...prev, [statusKey]: "success" }));
              }
            }
            break;
          case 'agent_setup':
            setAgentSetup(eventData.data.roles);
            setRegion(eventData.data.region);
            setSetupStatus(prev => ({ ...prev, agents: "success" }));
            break;
          case 'agent_end':
            if (eventData.data.agent_name === 'CompanyResolver' && eventData.data.output?.status === 'company_found') {
              setSetupStatus(prev => ({ ...prev, region: "success" }));
            }
            break;
          case 'agent_start':
            setAgentSteps(prev => [
              ...prev,
              {
                id: `${eventData.data.agent_name}-${Date.now()}`,
                agentName: eventData.data.agent_name,
                toolName: 'Processing...',
                status: 'Running',
                duration: null
              }
            ]);
            break;
          case 'node_end':
            if (eventData.payload && eventData.payload.node) {
              setAgentSteps(prev => prev.map(step => {
                if (step.agentName === eventData.payload.node && step.status === 'Running') {
                  const toolsUsed = eventData.payload.tools_used?.map(t => t.name).join(', ') || 'N/A';
                  return {
                    ...step,
                    toolName: toolsUsed,
                    status: eventData.payload.errors?.length > 0 ? 'Error' : 'Completed',
                    duration: eventData.payload.duration ? eventData.payload.duration * 1000 : null
                  };
                }
                return step;
              }));
            }
            break;
          case 'tool_result':
            setAgentSteps(prev => prev.map(step =>
              step.id.startsWith(eventData.data.agent_name) && step.status === 'Running'
                ? {
                    ...step,
                    toolName: eventData.data.tool_name,
                    status: eventData.data.success ? 'Running' : 'Error'
                  }
                : step
            ));
            setKpis(prev => ({
              ...prev,
              tool_calls: (prev.tool_calls || 0) + 1,
              successful_tool_calls: (prev.successful_tool_calls || 0) + (eventData.data.success ? 1 : 0)
            }));
            break;
          case 'llm_metrics':
            setLlmCalls(prev => prev + 1);
            setKpis(prev => ({
              ...prev,
              tokens: prev.tokens + (eventData.data.tokens || 0),
              cost: prev.cost + (eventData.data.cost || 0)
            }));
            break;
          case 'pipeline_end':
            setKpis(prev => ({
              ...prev,
              latency: eventData.data.end_to_end_latency_ms / 1000
            }));
            if (eventData.data.final_output) {
              const match = eventData.data.final_output.match(/report available at: (.*)/i);
              if (match && match[1]) setReportUrl(match[1]);
            }
            break;
          case 'evaluation_metric':
            if (eventData.data.metric_name === 'Answer Relevance') {
              setKpis(prev => ({ ...prev, relevance: eventData.data.score }));
            }
            if (eventData.data.metric_name === 'Faithfulness') {
              setKpis(prev => ({ ...prev, faithfulness: eventData.data.score }));
            }
            break;
          case 'pipeline_error':
            setError(eventData.data.error || "Pipeline error occurred");
            setKpis(prev => ({
              ...prev,
              errors: prev.errors + 1
            }));
            break;
          case 'graph_definition':
            if (eventData.data && eventData.data.mermaid_syntax) {
              setMermaidGraphSyntax(eventData.data.mermaid_syntax);
            }
            break;
          case 'log':
            if (eventData.data && eventData.data.message) {
              const message = eventData.data.message;
              const toolCallMatch = message.match(/INFO:tools\.graph_tools:Calling tool: (\w+) with kwargs: /);
              if (toolCallMatch) {
                const toolName = toolCallMatch[1];
                setAgentSteps(prev => {
                  const newSteps = [...prev];
                  if (newSteps.length > 0) {
                    const lastRunningAgentIndex = newSteps.findIndex(step => step.status === 'Running');
                    if (lastRunningAgentIndex !== -1) {
                      newSteps[lastRunningAgentIndex] = {
                        ...newSteps[lastRunningAgentIndex],
                        toolName: toolName,
                      };
                    }
                  }
                  return newSteps;
                });
              }
            }
            break;
          default:
            // Unhandled event type
            break;
        }
      } catch (e) {
        // Error parsing SSE event data
      }
    };

    eventSource.onerror = (e) => {
      if (!closedByComplete.current) {
        setError("Connection to server lost or failed to establish. Please check backend logs.");
      }
      setIsRunning(false);
      eventSource.close();
      closedByComplete.current = false;
    };

    eventSource.addEventListener('pipeline_complete', () => {
      closedByComplete.current = true;
      setIsRunning(false);
      eventSource.close();
    });
  };

  if (isLoadingModels) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <IconSpinner />
        <span className="mt-2 text-gray-500">Loading available LLM providers...</span>
      </div>
    );
  }

  if (error || Object.keys(availableProviders).length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-screen">
        <span className="text-red-600">{error || "No LLM providers available from backend."}</span>
        <button
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded font-semibold hover:bg-blue-700"
          onClick={() => window.location.reload()}
        >
          Try Reloading
        </button>
      </div>
    );
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6">Financial Analyzer</h1>
      <div className="flex flex-col md:flex-row gap-6">
        {/* Left Sidebar */}
        <div className="md:w-1/3 flex flex-col gap-4">
          <AnalysisConfig
            company={company} setCompany={setCompany}
            year={year} setYear={setYear}
            reportType={reportType} setReportType={setReportType}
            selectedGlobalProvider={selectedGlobalProvider} setSelectedGlobalProvider={setSelectedGlobalProvider}
            availableProviders={availableProviders}
            isRunning={isRunning}
            onStart={startAnalysis}
            verbose={verbose} setVerbose={setVerbose}
          />
          <LLMUsage llmCalls={llmCalls} />
        </div>
        {/* Middle Section */}
        <div className="md:w-2/3 flex flex-col gap-4">
          <AgentToolTable agentSteps={agentSteps} />
          <RawEventLog log={log} />
          <GraphViewer mermaidSyntax={mermaidGraphSyntax} />
        </div>
        {/* Right Sidebar */}
        <div className="md:w-1/3 flex flex-col gap-4">
          <LoggingSetupStatus setupStatus={setupStatus} />
          <PipelineKPIs kpis={kpis} reportUrl={reportUrl} />
          <AgentSetupViewer region={region} agentSetup={agentSetup} />
        </div>
      </div>
      {error && <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-md">{error}</div>}
    </div>
  );
}

export default App;
