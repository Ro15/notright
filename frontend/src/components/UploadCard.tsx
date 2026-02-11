
import { ChangeEvent, useRef, useState } from "react";
import { AnalysisResponse, analyzeFile } from "../lib/api";

type Props = {
  setResult: (r: AnalysisResponse | null) => void;
  setError: (e: string | null) => void;
  loading: boolean;
  setLoading: (v: boolean) => void;
};

const MAX_BYTES = 10 * 1024 * 1024;

const allowed =
  ".txt,.md,.csv,.json,.pdf,.docx,.png,.jpg,.jpeg";

export default function UploadCard({
  setResult,
  setError,
  loading,
  setLoading,
}: Props) {
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const onFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;

    if (f.size > MAX_BYTES) {
      setError("File too large. Max 10MB.");
      setFile(null);
      return;
    }

    setError(null);
    setResult(null);
    setFile(f);
  };

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const res = await analyzeFile(file);
      setResult(res);
    } catch (err: any) {
      setResult(null);
      setError(err?.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200 bg-white shadow-sm gradient-card">
      <div className="p-6 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-lg font-semibold text-slate-900">Upload file</p>
            <p className="text-sm text-slate-600">
              Accepted: txt, md, csv, json, pdf, docx, png, jpg, jpeg (max 10MB)
            </p>
          </div>
          <button
            onClick={() => inputRef.current?.click()}
            className="rounded-lg border border-primary/30 bg-white px-4 py-2 text-primary font-medium hover:bg-primary/5 transition disabled:opacity-50"
            disabled={loading}
          >
            Choose File
          </button>
        </div>

        <input
          type="file"
          accept={allowed}
          className="hidden"
          ref={inputRef}
          onChange={onFileChange}
        />

        <div className="rounded-lg border border-dashed border-slate-200 bg-white px-4 py-5">
          {file ? (
            <div className="flex items-center justify-between text-sm text-slate-700">
              <div>
                <p className="font-medium">{file.name}</p>
                <p className="text-xs text-slate-500">
                  {(file.size / 1024).toFixed(1)} KB
                </p>
              </div>
              <button
                className="text-primary hover:underline"
                onClick={() => {
                  setFile(null);
                  setResult(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
              >
                Clear
              </button>
            </div>
          ) : (
            <p className="text-slate-500 text-sm">
              No file selected yet.
            </p>
          )}
        </div>

        <div className="flex justify-end">
          <button
            onClick={handleAnalyze}
            disabled={!file || loading}
            className="inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-white font-semibold shadow hover:bg-primary/90 transition disabled:opacity-50"
          >
            {loading && (
              <span className="h-4 w-4 border-2 border-white/70 border-t-transparent rounded-full animate-spin" />
            )}
            Analyze
          </button>
        </div>
      </div>
    </div>
  );
}
