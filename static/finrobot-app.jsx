import React from 'react';

const { useState, useEffect, useRef, useMemo } = React;

// --- Helper Icon Components (using inline SVG) ---
const IconSpinner = () => (
    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
    </svg>
);

const IconAgent = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5 mr-2 text-gray-500"><path d="M12 8V4H8"/><rect x="4" y="12" width="16" height="8" rx="2"/><path d="M10 12v-2a2 2 0 0 1 2-2v0a2 2 0 0 1 2 2v2"/><path d="M16 12v-2a2 2 0 0 0-2-2v0a2 2 0 0 0-2 2v2"/></svg>;

const IconTool = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5 mr-2 text-gray-500"><path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/></svg>;

const IconChart = () => <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5 mr-2 text-gray-500"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>

// --- Reusable UI Components ---
const Card = ({ title, children, icon }) => (
    <div className="bg-white p-4 rounded-lg shadow-md">
        <h2 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">{icon}{title}</h2>
        {children}
    </div>
);

const KPI = ({ label, value }) => (
    <div className="flex justify-between items-center py-1 text-sm">
        <span className="text-gray-600">{label}:</span>
        <span className="font-semibold text-gray-800">{value}</span>
    </div>
);

// --- Main Application Components ---

const InputForm = ({ isRunning, onStart }) => {
    const [company, setCompany] = useState("Microsoft");
    const [year, setYear] = useState("2023");
    const [reportType, setReportType] = useState("kpi_bullet_insights");
    
    const currentYear = new Date().getFullYear();
    const years = Array.from({ length: 10 }, (_, i) => currentYear - i);

    const handleSubmit = (e) => {
        e.preventDefault();
        onStart({ company, year, reportType });
    };

    return (
        <Card title="Analysis Configuration" icon={<IconTool />}>
            <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                    <label htmlFor="company" className="block text-sm font-medium text-gray-700">Company Name or Ticker</label>
                    <input id="company" type="text" value={company} onChange={(e) => setCompany(e.target.value)} className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" disabled={isRunning} />
                </div>
                <div className="grid grid-cols-2 gap-4">
                    <div>
                        <label htmlFor="year" className="block text-sm font-medium text-gray-700">Year</label>
                        <select id="year" value={year} onChange={(e) => setYear(e.target.value)} className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" disabled={isRunning}>
                            {years.map(y => <option key={y} value={y}>{y}</option>)}
                        </select>
                    </div>
                    <div>
                        <label htmlFor="reportType" className="block text-sm font-medium text-gray-700">Report Type</label>
                        <select id="reportType" value={reportType} onChange={(e) => setReportType(e.target.value)} className="mt-1 block w-full px-3 py-2 bg-white border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-indigo-500 focus:border-indigo-500" disabled={isRunning}>
                            <option value="kpi_bullet_insights">KPI Insights</option>
                            <option value="full_analysis">Full Analysis</option>
                            <option value="swot_analysis">SWOT Analysis</option>
                        </select>
                    </div>
                </div>
                <button type="submit" disabled={isRunning} className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
                    {isRunning ? <><IconSpinner /> Running...</> : "Start Analysis"}
                </button>
            </form>
        </Card>
    );
};

const AgentStepsTable = ({ agentSteps }) => (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
        <h2 className="text-lg font-semibold text-gray-800 p-4 border-b border-gray-200">Agent & Tool Execution</h2>
        <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                    <tr>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Agent</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Tool</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                        <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Duration</th>
                    </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                    {agentSteps.map((step, index) => (
                        <tr key={index} className="animate-fade-in">
                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{step.agentName}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{step.toolName}</td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm">
                                {step.status === 'Running' && <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">Running</span>}
                                {step.status === 'Completed' && <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">Completed</span>}
                                {step.status === 'Error' && <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">Error</span>}
                            </td>
                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{step.duration ? `${step.duration.toFixed(0)} ms` : '...'}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </div>
);


const App = () => {
    // UI and View State
    const [isRunning, setIsRunning] = useState(false);
    const [error, setError] = useState(null);
    const [view, setView] = useState('analysis'); // 'analysis' or 'report'
    
    // Data State
    const [companyInfo, setCompanyInfo] = useState(null);
    const [agentSteps, setAgentSteps] = useState([]);
    const [llmModels, setLlmModels] = useState({});
    const [kpis, setKpis] = useState({ 
        total_time: 0, total_tokens: 0, total_cost: 0, 
        tool_calls: 0, successful_tool_calls: 0, pipeline_errors: 0,
        relevance_score: null, faithfulness_score: null 
    });
    const [finalReport, setFinalReport] = useState(null);
    const [runId, setRunId] = useState(null);
    
    const eventSourceRef = useRef(null);

    useEffect(() => {
        return () => eventSourceRef.current?.close();
    }, []);

    const processEvent = (eventData) => {
        switch (eventData.event_type) {
            case 'pipeline_start':
                setRunId(eventData.data.run_id);
                break;
            case 'agent_start':
                setAgentSteps(prev => [...prev, {
                    agentName: eventData.data.agent_name,
                    toolName: eventData.data.inputs?.tool_name || 'Processing...',
                    status: 'Running',
                    id: `${eventData.data.agent_name}-${prev.length}`
                }]);
                break;
            case 'agent_end':
                 setAgentSteps(prev => prev.map(step => 
                    (step.agentName === eventData.data.agent_name && step.status === 'Running')
                        ? { ...step, status: 'Completed', duration: eventData.data.agent_latency_ms }
                        : step
                ));
                 // *** FIX STARTS HERE ***
                 // This specifically checks for the output of the CompanyResolver agent
                 // and updates the companyInfo state when it's available.
                 if (eventData.data.agent_name === 'CompanyResolver' && eventData.data.output) {
                    setCompanyInfo(eventData.data.output);
                }
                 // *** FIX ENDS HERE ***
                break;
             case 'tool_result':
                 setAgentSteps(prev => prev.map(step =>
                    (step.agentName === eventData.data.agent_name && step.status === 'Running')
                        ? { ...step, toolName: eventData.data.tool_name, duration: eventData.data.tool_latency_ms, status: eventData.data.success ? 'Completed' : 'Error' }
                        : step
                ));
                setKpis(prev => ({
                    ...prev,
                    tool_calls: prev.tool_calls + 1,
                    successful_tool_calls: prev.successful_tool_calls + (eventData.data.success ? 1 : 0)
                }));
                 break;
            case 'llm_metrics':
                setLlmModels(prev => ({...prev, [eventData.data.agent_name]: eventData.data.model_name }));
                setKpis(prev => ({
                    ...prev,
                    total_tokens: prev.total_tokens + (eventData.data.tokens || 0),
                    total_cost: prev.total_cost + (eventData.data.cost || 0)
                }));
                break;
            case 'pipeline_end':
                setKpis(prev => ({ ...prev, total_time: eventData.data.end_to_end_latency_ms }));
                if (typeof eventData.data.final_output === 'string') {
                    const match = eventData.data.final_output.match(/report available at: (.*)/i);
                    if (match && match[1]) setFinalReport({ type: 'link', content: match[1] });
                    else setFinalReport({type: 'text', content: eventData.data.final_output});
                }
                break;
            case 'pipeline_error':
                setError(eventData.data.error || "An unknown pipeline error occurred.");
                setKpis(prev => ({ ...prev, pipeline_errors: prev.pipeline_errors + 1 }));
                break;
            case 'evaluation_metric':
                if (eventData.data.metric_name === 'Answer Relevance') setKpis(prev => ({ ...prev, relevance_score: eventData.data.score }));
                if (eventData.data.metric_name === 'Faithfulness') setKpis(prev => ({ ...prev, faithfulness_score: eventData.data.score }));
                break;
        }
    };

    const handleStartAnalysis = (config) => {
        setIsRunning(true);
        setError(null);
        setCompanyInfo(null);
        setAgentSteps([]);
        setLlmModels({});
        setKpis({ 
            total_time: 0, total_tokens: 0, total_cost: 0, 
            tool_calls: 0, successful_tool_calls: 0, pipeline_errors: 0,
            relevance_score: null, faithfulness_score: null
        });
        setFinalReport(null);
        setRunId(null);
        setView('analysis');
        
        const { company, year, reportType } = config;
        const es = new EventSource(`/stream?company=${encodeURIComponent(company)}&year=${year}&report_type=${reportType}`);
        eventSourceRef.current = es;

        es.onmessage = (event) => {
            try {
                const eventData = JSON.parse(event.data);
                if (eventData.event_type === 'pipeline_complete') {
                    es.close();
                    setIsRunning(false);
                } else {
                    processEvent(eventData);
                }
            } catch (e) {
                console.error("Failed to parse event data:", event.data, e);
            }
        };

        es.onerror = (err) => {
            setError("Connection to the server was lost or an error occurred.");
            setIsRunning(false);
            es.close();
        };
    };

    const analysisView = (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-1 space-y-6">
                <InputForm isRunning={isRunning} onStart={handleStartAnalysis} />
                {companyInfo && (
                     <Card title="Company Information" icon={<IconAgent />}>
                        <KPI label="Name" value={companyInfo.official_name} />
                        <KPI label="Region" value={companyInfo.region} />
                     </Card>
                )}
                <Card title="LLM Usage" icon={<IconTool />}>
                    {Object.keys(llmModels).length > 0 ? Object.entries(llmModels).map(([agent, model]) => (
                        <KPI key={agent} label={agent} value={model} />
                    )) : <p className="text-sm text-gray-500">No LLM calls yet.</p>}
                </Card>
                <Card title="Pipeline KPIs" icon={<IconChart />}>
                   <KPI label="End-to-End Latency" value={`${(kpis.total_time / 1000).toFixed(2)}s`} />
                   <KPI label="Total LLM Tokens" value={kpis.total_tokens.toLocaleString()} />
                   <KPI label="Estimated LLM Cost" value={`$${kpis.total_cost.toFixed(5)}`} />
                   <hr className="my-2 border-t border-gray-200"/>
                   <KPI label="Tool Call Success" value={kpis.tool_calls > 0 ? `${((kpis.successful_tool_calls / kpis.tool_calls) * 100).toFixed(1)}%` : 'N/A'} />
                   <KPI label="Pipeline Errors" value={kpis.pipeline_errors} />
                   <hr className="my-2 border-t border-gray-200"/>
                   <div className="text-xs text-gray-400 pt-1">LLM-as-Judge Metrics (Post-run)</div>
                   <KPI label="Answer Relevance" value={kpis.relevance_score ? `${kpis.relevance_score}/5.0` : 'N/A'} />
                   <KPI label="Faithfulness" value={kpis.faithfulness_score ? `${kpis.faithfulness_score}/5.0` : 'N/A'} />
                </Card>
            </div>
            <div className="lg:col-span-2">
                <AgentStepsTable agentSteps={agentSteps} />
            </div>
        </div>
    );
    
    const reportView = (
        <div>
             <button onClick={() => setView('analysis')} className="mb-4 px-4 py-2 bg-gray-200 text-gray-800 font-semibold rounded-md shadow-sm hover:bg-gray-300">
                &larr; Back to Analysis
             </button>
             <Card title="Final Report">
                {finalReport?.type === 'link' ? 
                    <a href={`/reports/${finalReport.content}`} target="_blank" className="text-indigo-600 hover:underline">Download/View Report: {finalReport.content}</a> :
                    <pre className="whitespace-pre-wrap font-mono text-sm">{finalReport?.content}</pre>
                }
             </Card>
        </div>
    );

    return (
        <div className="max-w-7xl mx-auto font-sans p-4 sm:p-6">
            <header className="mb-6 flex justify-between items-center">
                <h1 className="text-3xl font-bold text-gray-900">Financial Analyzer</h1>
                 {finalReport && view === 'analysis' && (
                     <button onClick={() => setView('report')} className="px-4 py-2 bg-green-600 text-white font-semibold rounded-md shadow-sm hover:bg-green-700">
                        View Report &rarr;
                     </button>
                 )}
            </header>
            
             {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-6" role="alert">
                    <strong className="font-bold">Error: </strong>
                    <span className="block sm:inline">{error}</span>
                </div>
            )}
            
            {view === 'analysis' ? analysisView : reportView}
        </div>
    );
};

// Use the new React 18 createRoot API
const container = document.getElementById('root');
const root = ReactDOM.createRoot(container);
root.render(<App />);
