"use client";

import { useMemo, useState } from "react";
import type { ReactNode } from "react";

type Domain = "CSM" | "PM" | "CHEM" | "CHEME";
type SrcDomain = Domain | "auto";
type AudienceLevel = "undergrad" | "grad" | "expert";

type Analog = { candidate: string; score: number };
type Evidence = { snippet: string; doc_id: string; source: string };

type AnnotatedTerm = {
  term: string;
  start: number;
  end: number;
  flagged: boolean;
  familiarity_tgt: number;
  distinctiveness_src: number;
  reason: string;
  analogs: Analog[];
  evidence: Evidence[];
  explain_available: boolean;
  short_explanation?: string;
};

type AnnotateResponse = {
  predicted_src?: string | null;
  predicted_src_confidence?: number | null;
  predicted_src_probs?: Record<string, number>;
  src_used?: string;
  src_warning: boolean;
  src_warning_reason?: string;
  is_ambiguous?: boolean;
  top2_gap?: number;
  suggested_src?: string;
  terms: AnnotatedTerm[];
};

type ExplainResponse = {
  term: string;
  short_explanation: string;
  long_explanation: string;
  closest_analog: string | null;
  caution_label: string;
  cache_hit: boolean;
  model: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

function renderHighlightedText(text: string, terms: AnnotatedTerm[], onSelect: (t: AnnotatedTerm) => void) {
  if (!terms.length) return <p className="whitespace-pre-wrap text-sm">{text}</p>;

  const sorted = [...terms].sort((a, b) => a.start - b.start);
  const chunks: ReactNode[] = [];
  let cursor = 0;

  sorted.forEach((t, idx) => {
    const s = Math.max(0, t.start);
    const e = Math.min(text.length, t.end);
    if (s > cursor) chunks.push(<span key={`plain-${idx}-${cursor}`}>{text.slice(cursor, s)}</span>);
    const piece = text.slice(s, e);
    chunks.push(
      <button
        type="button"
        key={`term-${idx}-${s}`}
        className={`rounded px-1 ${t.flagged ? "bg-amber-200 text-amber-950" : "bg-slate-200 text-slate-900"}`}
        onClick={() => onSelect(t)}
      >
        {piece}
      </button>
    );
    cursor = e;
  });

  if (cursor < text.length) chunks.push(<span key="tail">{text.slice(cursor)}</span>);
  return <p className="whitespace-pre-wrap text-sm leading-7">{chunks}</p>;
}

export default function HomePage() {
  const [text, setText] = useState("We optimize a graph neural network with sparse regularization under distribution shift.");
  const [src, setSrc] = useState<SrcDomain>("auto");
  const [tgt, setTgt] = useState<Domain>("PM");
  const [overrideSrc, setOverrideSrc] = useState<Domain | "">("");
  const [audience, setAudience] = useState<AudienceLevel>("grad");
  const [subtrack, setSubtrack] = useState("");
  const [maxTerms, setMaxTerms] = useState(8);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnnotateResponse | null>(null);
  const [selectedTerm, setSelectedTerm] = useState<AnnotatedTerm | null>(null);
  const [explainLong, setExplainLong] = useState<ExplainResponse | null>(null);
  const [shortCache, setShortCache] = useState<Record<string, string>>({});

  const disabled = useMemo(() => loading || !text.trim(), [loading, text]);

  const domains: Domain[] = ["CSM", "PM", "CHEM", "CHEME"] as Domain[];

  const top2 = useMemo(() => {
    const probs = result?.predicted_src_probs ?? {};
    return Object.entries(probs)
      .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
      .slice(0, 2);
  }, [result]);

  const confidenceLabel = (v: number | null | undefined): "High" | "Medium" | "Low" => {
    const x = typeof v === "number" ? v : 0;
    if (x >= 0.75) return "High";
    if (x >= 0.55) return "Medium";
    return "Low";
  };

  const shouldAskOverride = Boolean(
    src === "auto" && (result?.is_ambiguous || (typeof result?.predicted_src_confidence === "number" && result.predicted_src_confidence < 0.55))
  );

  const onAnnotate = async () => {
    setLoading(true);
    setError(null);
    setExplainLong(null);
    setSelectedTerm(null);

    try {
      const res = await fetch(`${API_BASE}/annotate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          src,
          tgt,
          audience_level: audience,
          subtrack,
          max_terms: maxTerms,
          include_short_explanations: false,
        }),
      });

      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail ?? `Request failed (${res.status})`);
      }
      const data: AnnotateResponse = await res.json();
      setResult(data);
      setOverrideSrc("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const onOverrideAndResubmit = async () => {
    if (!overrideSrc) return;
    setSrc(overrideSrc);
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/annotate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          src: overrideSrc,
          tgt,
          audience_level: audience,
          subtrack,
          max_terms: maxTerms,
          include_short_explanations: false,
        }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail ?? `Request failed (${res.status})`);
      }
      const data: AnnotateResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const fetchExplain = async (term: AnnotatedTerm, detail: "short" | "long") => {
    const key = `${term.term}::${detail}`;
    if (detail === "short" && shortCache[key]) return;

    const res = await fetch(`${API_BASE}/explain`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        term: term.term,
        src,
        tgt,
        audience_level: audience,
        subtrack,
        analogs: term.analogs.map((a) => a.candidate),
        detail,
      }),
    });

    if (!res.ok) {
      const payload = await res.json().catch(() => ({}));
      throw new Error(payload?.detail ?? `Explain failed (${res.status})`);
    }

    const exp: ExplainResponse = await res.json();
    if (detail === "short") {
      setShortCache((prev) => ({ ...prev, [key]: exp.short_explanation }));
      return;
    }
    setExplainLong(exp);
  };

  const applyReplacement = (candidate: string) => {
    if (!selectedTerm) return;
    const s = selectedTerm.start;
    const e = selectedTerm.end;
    const next = `${text.slice(0, s)}${candidate}${text.slice(e)}`;
    setText(next);
    setResult(null);
    setSelectedTerm(null);
    setExplainLong(null);
  };

  return (
    <main className="mx-auto max-w-6xl p-6 md:p-10">
      <h1 className="text-3xl font-bold">SciBabel Term Lens</h1>
      <p className="mt-2 text-sm text-slate-600">Detect unfamiliar terms, suggest analogs, and explain progressively in the target field.</p>

      <section className="mt-6 rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
        <label className="mb-2 block text-sm font-medium">Input text</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={7}
          className="w-full rounded-lg border border-slate-300 p-3 text-sm outline-none focus:border-blue-500"
        />

        <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-6">
          <div>
            <label className="mb-1 block text-sm font-medium">Interpret as:</label>
            <select value={tgt} onChange={(e) => setTgt(e.target.value as Domain)} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
              <option value="CSM">CSM</option>
              <option value="PM">PM</option>
              <option value="CHEM">CHEM</option>
              <option value="CHEME">CHEME</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">Detected as:</label>
            {src === "auto" && !shouldAskOverride ? (
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
                Auto (detected at run time)
              </div>
            ) : (
              <select value={src} onChange={(e) => setSrc(e.target.value as SrcDomain)} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
                <option value="auto">Auto</option>
                <option value="CSM">CSM</option>
                <option value="PM">PM</option>
                <option value="CHEM">CHEM</option>
                <option value="CHEME">CHEME</option>
              </select>
            )}
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">Audience</label>
            <select value={audience} onChange={(e) => setAudience(e.target.value as AudienceLevel)} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
              <option value="undergrad">undergrad</option>
              <option value="grad">grad</option>
              <option value="expert">expert</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">Subtrack</label>
            <input value={subtrack} onChange={(e) => setSubtrack(e.target.value)} placeholder="optional" className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
          </div>
          <div>
            <label className="mb-1 block text-sm font-medium">Max terms</label>
            <input type="number" min={1} max={20} value={maxTerms} onChange={(e) => setMaxTerms(Number(e.target.value))} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm" />
          </div>
          <div className="flex items-end">
            <button onClick={onAnnotate} disabled={disabled} className="w-full rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-blue-300">
              {loading ? "Annotating..." : "Annotate"}
            </button>
          </div>
        </div>

        <div className="mt-3 text-xs text-slate-500">
          Legacy translation flow available at <a className="text-blue-600 underline" href="/translate">/translate</a>
        </div>

        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

        {result && (
          <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
            <div className="flex flex-wrap items-center gap-3">
              <span><span className="font-medium">Detected as:</span> {result.predicted_src ?? "-"}</span>
              <span><span className="font-medium">Confidence:</span> {confidenceLabel(result.predicted_src_confidence)}{typeof result.predicted_src_confidence === "number" ? ` (${(result.predicted_src_confidence * 100).toFixed(1)}%)` : ""}</span>
              <span><span className="font-medium">Using source:</span> {result.src_used ?? src}</span>
            </div>
          </div>
        )}

        {shouldAskOverride && (
          <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            <p>
              Source domain unclear: {top2[0]?.[0] ?? "-"} {typeof top2[0]?.[1] === "number" ? top2[0][1].toFixed(2) : "-"}
              {" "}vs {top2[1]?.[0] ?? "-"} {typeof top2[1]?.[1] === "number" ? top2[1][1].toFixed(2) : "-"}. Please confirm.
            </p>
            <div className="mt-2 flex gap-2">
              <select value={overrideSrc} onChange={(e) => setOverrideSrc(e.target.value as Domain | "")} className="rounded-lg border border-amber-300 px-2 py-1 text-sm">
                <option value="">Select override</option>
                {top2.map(([d]) => (
                  <option key={`top-${d}`} value={d}>{d}</option>
                ))}
                {domains.map((d) => (
                  <option key={`all-${d}`} value={d}>{d}</option>
                ))}
              </select>
              <button type="button" onClick={onOverrideAndResubmit} className="rounded bg-amber-700 px-3 py-1 text-xs font-medium text-white">Confirm and re-run</button>
            </div>
          </div>
        )}
      </section>

      <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="lg:col-span-2 rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
          <h2 className="text-lg font-semibold">Annotated text</h2>
          <div className="mt-3 rounded-lg bg-slate-50 p-3">{renderHighlightedText(text, result?.terms ?? [], async (t) => {
            setSelectedTerm(t);
            setExplainLong(null);
            try {
              await fetchExplain(t, "short");
            } catch {
              // ignore short explain error in inline action
            }
          })}</div>

          {result?.src_warning && (
            <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
              This text looks more like {result.predicted_src}
              {typeof result.predicted_src_confidence === "number" ? ` (${(result.predicted_src_confidence * 100).toFixed(1)}%)` : ""}.
              {result.src_warning_reason === "mismatch" ? " Continue anyway?" : ""}
            </div>
          )}

          {result && (
            <div className="mt-4 text-sm">
              <span className="font-medium">Detected terms:</span> {result.terms.length}
            </div>
          )}
        </div>

        <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
          <h2 className="text-lg font-semibold">Term details</h2>
          {!selectedTerm && <p className="mt-3 text-sm text-slate-500">Click a highlighted term to inspect analogs and evidence.</p>}
          {selectedTerm && (
            <div className="mt-3 space-y-3 text-sm">
              <p>
                <span className="font-medium">Term:</span> {selectedTerm.term}
              </p>
              <p>
                <span className="font-medium">Flagged:</span> {selectedTerm.flagged ? "yes" : "no"} ({selectedTerm.reason})
              </p>
              <p>
                <span className="font-medium">Scores:</span> tgt familiarity {selectedTerm.familiarity_tgt.toFixed(3)}, src distinctiveness {selectedTerm.distinctiveness_src.toFixed(3)}
              </p>

              {shortCache[`${selectedTerm.term}::short`] && (
                <div className="rounded bg-blue-50 p-2 text-blue-900">{shortCache[`${selectedTerm.term}::short`]}</div>
              )}

              <div>
                <p className="font-medium">Analogs</p>
                <ul className="mt-1 space-y-1">
                  {selectedTerm.analogs.map((a) => (
                    <li key={a.candidate} className="flex items-center justify-between gap-2 rounded border border-slate-200 p-2">
                      <span>
                        {a.candidate} <span className="text-xs text-slate-500">({a.score.toFixed(3)})</span>
                      </span>
                      <button className="rounded bg-slate-900 px-2 py-1 text-xs text-white" onClick={() => applyReplacement(a.candidate)}>
                        Apply replacement
                      </button>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <p className="font-medium">Evidence snippets</p>
                <ul className="mt-1 space-y-2">
                  {selectedTerm.evidence.map((e, i) => (
                    <li key={`${e.doc_id}-${i}`} className="rounded border border-slate-200 p-2 text-xs">
                      <p>{e.snippet}</p>
                      <p className="mt-1 text-slate-500">
                        {e.source} • {e.doc_id}
                      </p>
                    </li>
                  ))}
                </ul>
              </div>

              <button
                className="rounded bg-blue-600 px-3 py-2 text-xs font-medium text-white"
                onClick={async () => {
                  try {
                    await fetchExplain(selectedTerm, "long");
                  } catch (e) {
                    setError(e instanceof Error ? e.message : "Explain failed");
                  }
                }}
              >
                Explain further
              </button>

              {explainLong && explainLong.term.toLowerCase() === selectedTerm.term.toLowerCase() && (
                <div className="rounded border border-blue-200 bg-blue-50 p-3 text-sm text-blue-950">
                  <p className="font-medium">Long explanation</p>
                  <p className="mt-1">{explainLong.long_explanation}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </section>
    </main>
  );
}
