
import { useState } from "react";
import UploadCard from "./components/UploadCard";
import ResultCard from "./components/ResultCard";
import { AnalysisResponse } from "./lib/api";

function App() {
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-4xl mx-auto px-4 py-12">
        <header className="text-center mb-10">
          <h1 className="text-4xl font-semibold text-slate-900">TrustCheck</h1>
          <p className="text-slate-600 mt-3">
            Upload a file to get an AI-likelihood score (not a guarantee).
          </p>
        </header>

        {error && (
          <div className="mb-6 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-rose-700">
            {error}
          </div>
        )}

        <UploadCard
          setResult={setResult}
          setError={setError}
          loading={loading}
          setLoading={setLoading}
        />

        {result && (
          <div className="mt-8">
            <ResultCard result={result} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
