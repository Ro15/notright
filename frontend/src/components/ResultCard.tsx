
import { AnalysisResponse } from "../lib/api";

type Props = {
  result: AnalysisResponse;
};

const labelColors: Record<string, string> = {
  "Likely AI": "bg-rose-100 text-rose-700",
  "Likely Human": "bg-emerald-100 text-emerald-700",
  Unclear: "bg-amber-100 text-amber-700",
};

export default function ResultCard({ result }: Props) {
  const color = labelColors[result.label] || "bg-slate-100 text-slate-700";

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="p-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-slate-500 uppercase tracking-wide">
              AI-likelihood score
            </p>
            <p className="text-4xl font-semibold text-slate-900">
              {result.ai_likelihood_score}
            </p>
          </div>
          <span className={`px-3 py-1 rounded-full text-sm font-semibold ${color}`}>
            {result.label}
          </span>
        </div>

        <div>
          <p className="text-sm font-semibold text-slate-800 mb-2">Reasoning</p>
          <ul className="list-disc list-inside text-slate-700 space-y-1">
            {result.reasoning.map((r, idx) => (
              <li key={idx}>{r}</li>
            ))}
          </ul>
        </div>

        <div>
          <p className="text-sm font-semibold text-slate-800 mb-2">Extracted preview</p>
          <div className="rounded-lg bg-slate-900 text-slate-100 p-4 text-sm font-mono whitespace-pre-wrap max-h-64 overflow-auto">
            {result.extracted_preview || "No preview available."}
          </div>
        </div>
      </div>
    </div>
  );
}
