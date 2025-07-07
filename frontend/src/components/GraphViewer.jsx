import React, { useEffect, useRef, useState } from "react";
import IconGraph from "./icons/IconGraph";
import cleanMermaid from "../utils/cleanMermaid";

// Assumes Mermaid is loaded globally via CDN and available as window.mermaid

const GraphViewer = ({ mermaidSyntax }) => {
  const graphRef = useRef(null);
  const [cleaned, setCleaned] = useState("");

  useEffect(() => {
    if (mermaidSyntax && graphRef.current && window.mermaid) {
      // Clean the Mermaid string
      const cleanedMermaid = cleanMermaid(mermaidSyntax);
      setCleaned(cleanedMermaid);

      // Generate a unique graph ID for Mermaid to render into
      const graphId = `graphDiv-${Date.now()}`;
      graphRef.current.innerHTML = `<div id="${graphId}" class="mermaid-graph"></div>`;

      // Render the Mermaid diagram
      window.mermaid
        .render(graphId, cleanedMermaid)
        .then(({ svg }) => {
          if (graphRef.current) {
            graphRef.current.innerHTML = svg;
          }
        })
        .catch(error => {
          if (graphRef.current) {
            graphRef.current.innerHTML = `<div class="text-red-500">Error rendering graph: ${error.message}</div>`;
          }
        });
    } else {
      setCleaned("");
      if (graphRef.current) graphRef.current.innerHTML = "";
    }
  }, [mermaidSyntax]);

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-8 mt-8">
      <div className="font-semibold mb-4 flex items-center">
        <IconGraph />
        Workflow Graph
      </div>
      {mermaidSyntax ? (
        <>
          <div className="mb-3">
            <div className="text-xs text-gray-500 mb-1">Cleaned Mermaid String:</div>
            <pre className="bg-gray-100 rounded p-2 overflow-x-auto max-h-40 text-xs border border-gray-200 mb-4">
              {cleaned}
            </pre>
          </div>
          <div
            ref={graphRef}
            className="overflow-auto"
            style={{
              minHeight: "400px",
              maxHeight: "900px",
              border: "1px solid #ececec",
              borderRadius: "10px",
              background: "#fcfcfc",
            }}
          >
            {/* Mermaid will inject SVG here */}
          </div>
        </>
      ) : (
        <div
          style={{
            minHeight: "400px",
            border: "1px solid #ececec",
            borderRadius: "10px",
            background: "#fcfcfc",
          }}
          className="flex items-center justify-center text-gray-400 text-xs"
        >
          Waiting for graph definition...
        </div>
      )}
    </div>
  );
};

export default GraphViewer;
