import React from "react";
import { SETUP_STEPS_LIST, STATUS_COLOR } from "../utils/constants";

function LoggingSetupStatus({ setupStatus }) {
  return (
    <div className="bg-white rounded-lg shadow p-4 mb-4">
      <div className="font-semibold mb-2">Setup Progress</div>
      <table className="w-full text-xs">
        <tbody>
          {SETUP_STEPS_LIST.map((step) => (
            <tr key={step.key}>
              <td className="pr-2 py-1">
                <span
                  className={`inline-block w-4 h-4 rounded-full border border-gray-300 ${STATUS_COLOR[setupStatus[step.key] || "pending"]}`}
                  title={setupStatus[step.key]}
                ></span>
              </td>
              <td className="py-1">{step.label}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
export default LoggingSetupStatus;
