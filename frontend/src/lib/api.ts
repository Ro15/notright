
export type AnalysisResponse = {
  ai_likelihood_score: number;
  label: "Likely AI" | "Unclear" | "Likely Human";
  reasoning: string[];
  extracted_preview: string;
};

const API_URL = "http://localhost:8000/analyze";

export async function analyzeFile(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(API_URL, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data?.error || "Failed to analyze file.");
  }
  return data as AnalysisResponse;
}
