import { useEffect, useState } from "react";
import { fetchTools } from "../api";
import type { ToolInfo } from "../api";

interface ToolSelectorProps {
  value: string | null;
  onChange: (tool: string | null) => void;
}

const AUTO_VALUE = "";

/**
 * Lets the person single out one specialist for the NEXT message,
 * bypassing the supervisor's own routing for that turn -- see
 * agents/api.py's ChatRequest.tool docstring and graph.py's module
 * docstring for exactly what this does and doesn't change server-side.
 *
 * "Auto (supervisor picks)" is the first option and the default
 * (`value === null`) -- normal, unmodified behavior, exactly what every
 * conversation gets unless a person deliberately overrides it here. This
 * is an isolation/debugging control, not a way to get a "better" answer
 * -- forcing a specialist skips the supervisor's own multi-specialist
 * routing and re-route loop entirely, so the wrong forced choice can
 * answer worse than letting the supervisor pick would have.
 */
export function ToolSelector({ value, onChange }: ToolSelectorProps) {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchTools()
      .then((res) => setTools(res.tools))
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  const selected = tools.find((t) => t.name === value);

  return (
    <div className="aui-tool-selector">
      <select
        className="aui-tool-selector-select"
        value={value ?? AUTO_VALUE}
        onChange={(e) => onChange(e.target.value === AUTO_VALUE ? null : e.target.value)}
        title={selected?.description ?? "Let the supervisor pick among every specialist"}
      >
        <option value={AUTO_VALUE}>Auto (supervisor picks)</option>
        {tools.map((t) => (
          <option key={t.name} value={t.name} title={t.description}>
            {t.name}
          </option>
        ))}
      </select>
      {error && <span className="aui-tool-selector-error">tools unavailable</span>}
    </div>
  );
}
