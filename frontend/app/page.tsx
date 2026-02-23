"use client";

import { useMemo, useState } from "react";

type Domain = "CSM" | "PM" | "CCE";

type Candidate = {
  text: string;
  total_score: number;
  breakdown: {
    domain: number;
    meaning: number;
    lex: number;
  };
  temperature: number;
};

type TranslateResponse = {
  best_candidate: string;
  best_score: number;
  score_breakdown: { domain: number; meaning: number; lex: number };
  candidates: Candidate[];
  prompt_action: string;
  used_fallback: boolean;
  num_attempted: number;
  num_returned: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function HomePage() {
  const [text, setText] = useState(
    "We propose a sparse graph method for improving generalization under distribution shift."
  );
  const [src, setSrc] = useState<Domain>("CSM");
  const [tgt, setTgt] = useState<Domain>("PM");
  const [k, setK] = useState(1);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TranslateResponse | null>(null);

  const disabled = useMemo(() => !text.trim() || loading, [text, loading]);

  const onTranslate = async () => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/translate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, src, tgt, k }),
      });

      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail ?? `Request failed (${res.status})`);
      }

      const data: TranslateResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto max-w-4xl p-6 md:p-10">
      <h1 className="text-3xl font-bold">SciBabel MVP</h1>
      <p className="mt-2 text-sm text-slate-600">
        Cross-domain scientific language translation with reward-based reranking.
      </p>

      <section className="mt-6 rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
        <label className="mb-2 block text-sm font-medium">Input text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={7}
          className="w-full rounded-lg border border-slate-300 p-3 text-sm outline-none focus:border-blue-500"
        />

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
          <div>
            <label className="mb-1 block text-sm font-medium">Source domain</label>
            <select
              value={src}
              onChange={(e) => setSrc(e.target.value as Domain)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="CSM">CSM</option>
              <option value="PM">PM</option>
              <option value="CCE">CCE</option>
            </select>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Target domain</label>
            <select
              value={tgt}
              onChange={(e) => setTgt(e.target.value as Domain)}
              className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
            >
              <option value="CSM">CSM</option>
              <option value="PM">PM</option>
              <option value="CCE">CCE</option>
            </select>
          </div>

          <div>
            <label className="mb-1 block text-sm font-medium">Candidates (k): {k}</label>
            <input
              type="range"
              min={1}
              max={8}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        <button
          onClick={onTranslate}
          disabled={disabled}
          className="mt-4 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-blue-300"
        >
          {loading ? "Translating..." : "Translate"}
        </button>

        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </section>

      {result && (
        <section className="mt-6 rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
          <h2 className="text-xl font-semibold">Best output</h2>
          <div className="mt-2 flex flex-wrap gap-2 text-xs">
            <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-700">
              attempted runs: {result.num_attempted}
            </span>
            <span className="rounded-full bg-slate-100 px-2 py-1 text-slate-700">
              returned candidates: {result.num_returned}
            </span>
            {result.used_fallback && (
              <span className="rounded-full bg-amber-100 px-2 py-1 text-amber-800">
                partial result (quota fallback)
              </span>
            )}
          </div>

          <p className="mt-2 whitespace-pre-wrap rounded-lg bg-slate-50 p-3 text-sm">
            {result.best_candidate}
          </p>

          <div className="mt-3 text-sm text-slate-700">
            <p>
              <span className="font-medium">Best score:</span> {result.best_score.toFixed(4)}
            </p>
            <p>
              <span className="font-medium">Breakdown:</span> domain {result.score_breakdown.domain.toFixed(4)}
              , meaning {result.score_breakdown.meaning.toFixed(4)}, lex {result.score_breakdown.lex.toFixed(4)}
            </p>
            <p>
              <span className="font-medium">Prompt action:</span> {result.prompt_action}
            </p>
          </div>

          <h3 className="mt-4 text-lg font-semibold">Top candidates</h3>
          <ol className="mt-2 space-y-2">
            {result.candidates.map((c, idx) => (
              <li key={`${idx}-${c.temperature}`} className="rounded-lg border border-slate-200 p-3 text-sm">
                <div className="mb-1 text-xs text-slate-500">
                  rank #{idx + 1} • total {c.total_score.toFixed(4)} • temp {c.temperature.toFixed(1)}
                </div>
                <p className="whitespace-pre-wrap">{c.text}</p>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
