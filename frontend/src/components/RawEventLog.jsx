import React from "react";

const RawEventLog = ({ log }) => (
  <div className="bg-gray-900 rounded-lg shadow p-4 text-xs text-white font-mono h-40 overflow-y-auto">
    <div className="flex gap-4 mb-1">
      <span className="font-semibold text-gray-300">Raw Event Log</span>
    </div>
    <div>
      {log.length === 0 ? (
        <div className="text-gray-500">No events yet.</div>
      ) : (
        log.map((line, i) => {
          try {
            const eventData = JSON.parse(line);
            // Prioritize displaying messages from 'log' events
            if (
              eventData.event_type === "log" &&
              eventData.data &&
              eventData.data.message
            ) {
              return (
                <div key={i} className="text-gray-400">
                  {eventData.data.message}
                </div>
              );
            }
            // For other structured events, display their type and stringified payload
            return (
              <div key={i} className="text-blue-300">
                {eventData.event_type}:{" "}
                {JSON.stringify(eventData.payload || eventData.data, null, 2)}
              </div>
            );
          } catch (e) {
            // Fallback for non-JSON lines (should ideally not happen if backend is consistent)
            return (
              <div key={i} className="text-red-400">
                Error parsing log line: {line}
              </div>
            );
          }
        })
      )}
    </div>
  </div>
);

export default RawEventLog;
