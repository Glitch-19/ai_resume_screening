import { useState, useRef, useCallback } from "react";
import axios from "axios";
import {
  Brain,
  Upload,
  FileText,
  Trash2,
  Search,
  Trophy,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  Sparkles,
  BarChart3,
} from "lucide-react";

// Axios instance – uses Vite proxy (/api -> http://localhost:8000)
const api = axios.create({
  baseURL: "",
  timeout: 30000,
});

function Header({ health }) {
  return (
    <header className="border-b bg-white/80 backdrop-blur sticky top-0 z-30">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-10 w-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow">
            <Brain className="h-6 w-6" />
          </div>
          <div>
            <h1 className="text-lg sm:text-xl font-bold tracking-tight text-gray-900">
              AI Resume Screening <span className="text-indigo-600">& Job Matching</span>
            </h1>
            <p className="text-xs sm:text-sm text-gray-500">
              Hybrid TF-IDF (0.4) + Semantic MiniLM-L6-v2 (0.6) • Skill Gap Analysis
            </p>
          </div>
        </div>
        <div className="hidden sm:flex items-center gap-3">
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="text-sm text-gray-600 hover:text-indigo-600 transition"
          >
            API Docs
          </a>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border ${
              health?.status === "ok"
                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                : "bg-amber-50 text-amber-700 border-amber-200"
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${health?.status === "ok" ? "bg-emerald-500" : "bg-amber-500"}`} />
            {health ? (health.model_loaded ? "Model Ready" : "API Ready (TF-IDF)") : "Checking..."}
          </span>
        </div>
      </div>
    </header>
  );
}

function ProgressBar({ value }) {
  const color =
    value >= 80 ? "bg-emerald-500" : value >= 60 ? "bg-amber-500" : value >= 40 ? "bg-orange-500" : "bg-red-500";
  const bg =
    value >= 80
      ? "bg-emerald-100"
      : value >= 60
        ? "bg-amber-100"
        : value >= 40
          ? "bg-orange-100"
          : "bg-red-100";
  return (
    <div className={`h-2 w-full rounded-full ${bg} overflow-hidden`}>
      <div className={`h-full ${color} rounded-full transition-all duration-700`} style={{ width: `${Math.min(100, value)}%` }} />
    </div>
  );
}

function Badge({ children, variant = "green" }) {
  const styles =
    variant === "green"
      ? "bg-emerald-50 text-emerald-700 border-emerald-200"
      : variant === "red"
        ? "bg-red-50 text-red-700 border-red-200"
        : "bg-gray-50 text-gray-700 border-gray-200";
  return (
    <span className={`inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-medium ${styles}`}>
      {children}
    </span>
  );
}

export default function App() {
  const [jd, setJd] = useState(
    "We need a Python developer with AWS, Docker, Kubernetes and machine learning experience. Must know FastAPI and PostgreSQL. Familiar with CI/CD, Git and microservices."
  );
  const [files, setFiles] = useState([]); // { id, file, status, candidate_name, resume_text, char_count, error }
  const [isDragging, setIsDragging] = useState(false);
  const [isParsing, setIsParsing] = useState(false);
  const [isMatching, setIsMatching] = useState(false);
  const [results, setResults] = useState(null); // MatchResponse
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const fileInputRef = useRef(null);

  // Fetch health on mount
  useState(() => {
    api
      .get("/health")
      .then((r) => setHealth(r.data))
      .catch(() => setHealth({ status: "error" }));
  });

  const totalParsed = files.filter((f) => f.status === "parsed").length;

  const handleFiles = useCallback(
    async (fileList) => {
      const incoming = Array.from(fileList).filter((f) => f.size > 0);
      if (incoming.length === 0) return;

      // Validate quickly in UI
      const newEntries = incoming.map((file) => ({
        id: `${file.name}-${file.size}-${Date.now()}-${Math.random()}`,
        file,
        filename: file.name,
        status: "queued",
        candidate_name: null,
        resume_text: null,
        char_count: 0,
        error: null,
      }));
      setFiles((prev) => [...prev, ...newEntries]);
      setError(null);
      setIsParsing(true);

      // Parse each via /api/upload-resume sequentially (could be parallel)
      for (const entry of newEntries) {
        try {
          // optimistic update to parsing
          setFiles((prev) => prev.map((p) => (p.id === entry.id ? { ...p, status: "parsing" } : p)));
          const form = new FormData();
          form.append("file", entry.file);
          const { data } = await api.post("/api/upload-resume", form, {
            headers: { "Content-Type": "multipart/form-data" },
          });
          setFiles((prev) =>
            prev.map((p) =>
              p.id === entry.id
                ? {
                    ...p,
                    status: "parsed",
                    candidate_name: data.candidate_name,
                    resume_text: data.raw_text,
                    char_count: data.char_count,
                    error: null,
                  }
                : p
            )
          );
        } catch (err) {
          const msg = err?.response?.data?.detail || err.message || "Failed to parse";
          setFiles((prev) => prev.map((p) => (p.id === entry.id ? { ...p, status: "error", error: msg } : p)));
        }
      }
      setIsParsing(false);
    },
    []
  );

  const onDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files) handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const onFileChange = (e) => {
    if (e.target.files) handleFiles(e.target.files);
    // reset input to allow re-select same file
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeFile = (id) => {
    setFiles((prev) => prev.filter((p) => p.id !== id));
  };

  const handleMatch = async () => {
    setError(null);
    if (!jd.trim() || jd.trim().length < 10) {
      setError("Please provide a job description (at least 10 characters).");
      return;
    }
    const parsed = files.filter((f) => f.status === "parsed" && f.resume_text);
    if (parsed.length === 0) {
      setError("Please upload at least one PDF/DOCX resume (and wait for parsing to finish).");
      return;
    }
    setIsMatching(true);
    try {
      const payload = {
        job_description: jd,
        candidates: parsed.map((p) => ({
          candidate_name: p.candidate_name || p.filename.replace(/\.[^/.]+$/, "").replace(/[_-]+/g, " "),
          resume_text: p.resume_text,
          filename: p.filename,
        })),
      };
      const { data } = await api.post("/api/match", payload);
      setResults(data);
      // Scroll to results
      setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 100);
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message || "Matching failed";
      setError(Array.isArray(msg) ? msg.map((m) => m.msg).join(", ") : typeof msg === "string" ? msg : JSON.stringify(msg));
    } finally {
      setIsMatching(false);
    }
  };

  const handleClear = () => {
    setResults(null);
    setError(null);
  };

  const demoLoad = async () => {
    // Load demo candidates without uploading files (fast path)
    const demo = [
      {
        candidate_name: "Alice Chen",
        resume_text:
          "Alice Chen\nSenior Python Developer with 5 years building ML pipelines on AWS. Expert in Python, FastAPI, PostgreSQL, Docker, Kubernetes. Implemented ETL with Airflow and model serving with Docker. Strong in microservices and CI/CD.",
      },
      {
        candidate_name: "Bob Kumar",
        resume_text:
          "Bob Kumar\nGraphic Designer specializing in Photoshop, Illustrator, Figma typography branding. 8 years creative direction.",
      },
      {
        candidate_name: "Carol Singh",
        resume_text:
          "Carol Singh\nPython developer with AWS and Docker experience. Familiar with Kubernetes, built data pipelines with Spark. Learning machine learning fundamentals. Has Git and REST API knowledge.",
      },
    ];
    setFiles(
      demo.map((d, i) => ({
        id: `demo-${i}`,
        file: null,
        filename: `${d.candidate_name.replace(/\s+/g, "_")}.txt`,
        status: "parsed",
        candidate_name: d.candidate_name,
        resume_text: d.resume_text,
        char_count: d.resume_text.length,
        error: null,
      }))
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-indigo-50/40 via-white to-white">
      <Header health={health} />

      <main className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 py-6 sm:py-8 space-y-6">
        {/* Top grid: JD + Upload */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Job Description */}
          <section className="rounded-2xl border bg-white shadow-sm flex flex-col">
            <div className="p-5 border-b flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-indigo-100 text-indigo-600 flex items-center justify-center">
                  <FileText className="h-4 w-4" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-gray-900">Job Description</h2>
                  <p className="text-xs text-gray-500">Paste the JD to match against</p>
                </div>
              </div>
              <span className="text-xs text-gray-500">{jd.length} chars</span>
            </div>
            <div className="p-5 flex-1 flex flex-col gap-4">
              <textarea
                value={jd}
                onChange={(e) => setJd(e.target.value)}
                rows={10}
                placeholder="We need a Python developer with AWS, Docker..."
                className="w-full flex-1 min-h-[180px] rounded-xl border border-gray-200 bg-gray-50/50 p-4 text-sm leading-relaxed placeholder:text-gray-400 focus:bg-white focus:border-indigo-300 focus:ring-4 focus:ring-indigo-100 outline-none transition"
              />
              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-500">Tip: Include required skills, tools, and seniority</p>
                <button
                  onClick={() => setJd("")}
                  className="text-xs text-gray-600 hover:text-gray-900 underline-offset-4 hover:underline"
                >
                  Clear
                </button>
              </div>
            </div>
          </section>

          {/* Upload */}
          <section className="rounded-2xl border bg-white shadow-sm flex flex-col">
            <div className="p-5 border-b flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-lg bg-emerald-100 text-emerald-600 flex items-center justify-center">
                  <Upload className="h-4 w-4" />
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-gray-900">Resume Upload</h2>
                  <p className="text-xs text-gray-500">PDF or DOCX • up to 5MB each</p>
                </div>
              </div>
              <button
                onClick={demoLoad}
                className="hidden sm:inline-flex items-center gap-1.5 rounded-full bg-gray-900 px-3 py-1.5 text-xs font-medium text-white hover:bg-black transition"
              >
                <Sparkles className="h-3.5 w-3.5" /> Load demo
              </button>
            </div>

            <div className="p-5 flex-1 flex flex-col">
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`group relative flex flex-1 min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-6 text-center transition
                  ${isDragging ? "border-indigo-400 bg-indigo-50" : "border-gray-200 hover:border-gray-300 hover:bg-gray-50/50 bg-gray-50/30"}`}
              >
                <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-xl bg-white border shadow-sm">
                  <Upload className="h-6 w-6 text-gray-700" />
                </div>
                <p className="mt-3 text-sm font-medium text-gray-900">Drop resumes here or click to browse</p>
                <p className="mt-1 text-xs text-gray-500">Supports multiple PDF/DOCX files</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={onFileChange}
                  className="hidden"
                />
              </div>

              {/* File list */}
              {files.length > 0 && (
                <div className="mt-4 space-y-2 max-h-56 overflow-auto pr-1">
                  {files.map((f) => (
                    <div
                      key={f.id}
                      className="flex items-center gap-3 rounded-xl border bg-white px-3 py-2.5 shadow-sm"
                    >
                      <div
                        className={`h-8 w-8 rounded-lg flex items-center justify-center shrink-0
                        ${f.status === "parsed" ? "bg-emerald-100 text-emerald-600" : f.status === "error" ? "bg-red-100 text-red-600" : "bg-amber-100 text-amber-600"}`}
                      >
                        {f.status === "parsed" ? (
                          <CheckCircle2 className="h-4 w-4" />
                        ) : f.status === "error" ? (
                          <XCircle className="h-4 w-4" />
                        ) : (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-medium text-gray-900">{f.candidate_name || f.filename}</p>
                        <p className="text-xs text-gray-500 truncate">
                          {f.filename} • {f.char_count ? `${f.char_count} chars` : f.status} {f.error ? `• ${f.error}` : ""}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          removeFile(f.id);
                        }}
                        className="rounded-full p-1.5 hover:bg-gray-100 text-gray-500 hover:text-red-600 transition"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              <div className="mt-3 flex items-center justify-between text-xs text-gray-500">
                <span>
                  {totalParsed} of {files.length} parsed {isParsing && "• parsing..."}
                </span>
                {files.length > 0 && (
                  <button onClick={() => setFiles([])} className="hover:text-gray-900 underline-offset-4 hover:underline">
                    Clear all
                  </button>
                )}
              </div>
            </div>
          </section>
        </div>

        {/* Action bar */}
        <div className="rounded-2xl border bg-white p-4 shadow-sm flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-sm text-gray-600">
            <BarChart3 className="h-4 w-4" />
            <span>
              {totalParsed} candidate{totalParsed !== 1 ? "s" : ""} ready • Hybrid 0.4 TF-IDF + 0.6 Semantic
            </span>
          </div>
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <button
              onClick={handleClear}
              className="flex-1 sm:flex-none rounded-xl border bg-white px-5 py-2.5 text-sm font-medium text-gray-700 hover:bg-gray-50 transition"
            >
              Reset
            </button>
            <button
              onClick={handleMatch}
              disabled={isMatching || isParsing}
              className="flex-1 sm:flex-none inline-flex items-center justify-center gap-2 rounded-xl bg-indigo-600 px-6 py-2.5 text-sm font-semibold text-white shadow hover:bg-indigo-700 disabled:opacity-60 disabled:cursor-not-allowed transition"
            >
              {isMatching ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
              {isMatching ? "Ranking..." : `Match & Rank (${totalParsed})`}
            </button>
          </div>
        </div>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 flex items-start gap-3 text-sm text-red-700">
            <AlertTriangle className="h-5 w-5 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Results */}
        {results && (
          <section id="results" className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <Trophy className="h-5 w-5 text-amber-500" /> Ranking Results
                <span className="text-sm font-normal text-gray-500">({results.total_candidates} candidates)</span>
              </h2>
              <span className="text-xs text-gray-500 hidden sm:inline">
                Sorted by Total Score descending • {results.job_description_preview.slice(0, 80)}...
              </span>
            </div>

            {/* Analytics summary */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {[
                {
                  label: "Top Score",
                  value: results.results[0] ? `${results.results[0].total_score.toFixed(1)}%` : "-",
                },
                {
                  label: "Avg Score",
                  value: `${(results.results.reduce((a, r) => a + r.total_score, 0) / (results.results.length || 1)).toFixed(1)}%`,
                },
                {
                  label: "Matched Best",
                  value: results.results[0]?.candidate_name || "-",
                },
                {
                  label: "Gap Focus",
                  value: results.results[0]?.missing_skills?.[0] || "None",
                },
              ].map((stat) => (
                <div key={stat.label} className="rounded-xl border bg-white p-4">
                  <p className="text-xs text-gray-500">{stat.label}</p>
                  <p className="mt-1 text-sm font-semibold text-gray-900 truncate">{stat.value}</p>
                </div>
              ))}
            </div>

            {/* Desktop table */}
            <div className="hidden lg:block overflow-hidden rounded-2xl border bg-white shadow-sm">
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-xs uppercase tracking-wider text-gray-500">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium">#</th>
                      <th className="px-4 py-3 text-left font-medium">Candidate</th>
                      <th className="px-4 py-3 text-left font-medium w-[22%]">Match %</th>
                      <th className="px-4 py-3 text-left font-medium">Details</th>
                      <th className="px-4 py-3 text-left font-medium">Matched Skills</th>
                      <th className="px-4 py-3 text-left font-medium">Missing Skills</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y">
                    {results.results.map((r, idx) => (
                      <tr key={r.candidate_name + idx} className="hover:bg-gray-50/50">
                        <td className="px-4 py-4">
                          <span
                            className={`inline-flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold
                            ${idx === 0 ? "bg-amber-100 text-amber-700 border border-amber-200" : idx === 1 ? "bg-gray-100 text-gray-700 border" : idx === 2 ? "bg-orange-100 text-orange-700 border border-orange-200" : "bg-white text-gray-600 border"}`}
                          >
                            {idx + 1}
                          </span>
                        </td>
                        <td className="px-4 py-4">
                          <p className="font-medium text-gray-900">{r.candidate_name}</p>
                          <p className="text-xs text-gray-500">TF-IDF {r.tfidf_score.toFixed(1)} • Semantic {r.semantic_score.toFixed(1)}</p>
                        </td>
                        <td className="px-4 py-4">
                          <div className="space-y-1.5">
                            <div className="flex items-baseline justify-between">
                              <span className="text-sm font-semibold text-gray-900">{r.total_score.toFixed(1)}%</span>
                              <span className="text-xs text-gray-500">
                                {r.total_score >= 80 ? "Strong Fit" : r.total_score >= 60 ? "Moderate" : "Low"}
                              </span>
                            </div>
                            <ProgressBar value={r.total_score} />
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex flex-col gap-1 text-xs">
                            <span className="text-gray-600">TF-IDF {r.tfidf_score.toFixed(0)}%</span>
                            <ProgressBar value={r.tfidf_score} />
                            <span className="text-gray-600 mt-1">Semantic {r.semantic_score.toFixed(0)}%</span>
                            <ProgressBar value={r.semantic_score} />
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex flex-wrap gap-1.5 max-w-[260px]">
                            {r.matching_skills.length ? (
                              r.matching_skills.map((s) => <Badge key={s} variant="green">{s}</Badge>)
                            ) : (
                              <span className="text-xs text-gray-400">—</span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <div className="flex flex-wrap gap-1.5 max-w-[260px]">
                            {r.missing_skills.length ? (
                              r.missing_skills.map((s) => <Badge key={s} variant="red">{s}</Badge>)
                            ) : (
                              <span className="inline-flex items-center gap-1 text-xs text-emerald-600">
                                <CheckCircle2 className="h-3.5 w-3.5" /> None
                              </span>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Mobile cards */}
            <div className="grid grid-cols-1 gap-4 lg:hidden">
              {results.results.map((r, idx) => (
                <div key={r.candidate_name + idx} className="rounded-2xl border bg-white p-5 shadow-sm">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span
                        className={`h-8 w-8 rounded-full flex items-center justify-center text-sm font-bold border
                        ${idx === 0 ? "bg-amber-100 text-amber-700 border-amber-200" : "bg-white text-gray-700"}`}
                      >
                        {idx + 1}
                      </span>
                      <div>
                        <p className="font-semibold text-gray-900">{r.candidate_name}</p>
                        <p className="text-xs text-gray-500">TF-IDF {r.tfidf_score.toFixed(1)} • Semantic {r.semantic_score.toFixed(1)}</p>
                      </div>
                    </div>
                    <span className="text-sm font-bold text-gray-900">{r.total_score.toFixed(1)}%</span>
                  </div>
                  <div className="mt-4 space-y-3">
                    <ProgressBar value={r.total_score} />
                    <div>
                      <p className="text-xs font-medium text-gray-700 mb-1.5">Matched Skills</p>
                      <div className="flex flex-wrap gap-1.5">
                        {r.matching_skills.length ? (
                          r.matching_skills.map((s) => <Badge key={s} variant="green">{s}</Badge>)
                        ) : (
                          <span className="text-xs text-gray-400">None</span>
                        )}
                      </div>
                    </div>
                    <div>
                      <p className="text-xs font-medium text-gray-700 mb-1.5">Missing Skills</p>
                      <div className="flex flex-wrap gap-1.5">
                        {r.missing_skills.length ? (
                          r.missing_skills.map((s) => <Badge key={s} variant="red">{s}</Badge>)
                        ) : (
                          <span className="text-xs text-emerald-600 inline-flex items-center gap-1">
                            <CheckCircle2 className="h-3.5 w-3.5" /> No gaps
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="rounded-xl border bg-white px-4 py-3 flex flex-wrap items-center gap-4 text-xs text-gray-600">
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-full bg-emerald-500" /> Matched (green)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-full bg-red-500" /> Missing (red)
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="h-2 w-10 rounded bg-gray-200" /> Total = 0.4 TF-IDF + 0.6 Semantic
              </span>
            </div>
          </section>
        )}

        {/* Empty state */}
        {!results && (
          <div className="rounded-2xl border border-dashed bg-white/50 p-10 text-center">
            <div className="mx-auto h-12 w-12 rounded-xl bg-indigo-50 flex items-center justify-center text-indigo-600">
              <BarChart3 className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-sm font-semibold text-gray-900">No rankings yet</h3>
            <p className="mt-1 text-sm text-gray-500 max-w-md mx-auto">
              Upload resumes (PDF/DOCX) and enter a job description, then click <span className="font-medium">Match & Rank</span> to see
              hybrid scores and skill gaps.
            </p>
            <p className="mt-3 text-xs text-gray-400">
              API: <code className="bg-gray-100 rounded px-1.5 py-0.5">POST /api/upload-resume</code> +{" "}
              <code className="bg-gray-100 rounded px-1.5 py-0.5">POST /api/match</code> via Axios
            </p>
          </div>
        )}

        <footer className="pt-4 text-center text-xs text-gray-400">
          Built with React (Vite) + Tailwind CSS • FastAPI + scikit-learn + sentence-transformers/all-MiniLM-L6-v2
        </footer>
      </main>
    </div>
  );
}
