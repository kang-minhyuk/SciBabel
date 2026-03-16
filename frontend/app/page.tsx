"use client";

import { useMemo, useState } from "react";
import type { ReactNode } from "react";

type Domain = "CSM" | "PM" | "CHEM" | "CHEME";
type SrcDomain = Domain | "auto";
type AudienceLevel = "undergrad" | "grad" | "expert";

type Analog = { candidate: string; score: number };
type Evidence = { snippet: string; doc_id: string; source: string };

type PdfTerm = {
  term_id: string;
  term: string;
  surface_term?: string;
  canonical_term?: string;
  start: number;
  end: number;
  flagged: boolean;
  familiarity_tgt?: number;
  distinctiveness_src?: number;
  reason?: string;
  analogs: Analog[];
  evidence: Evidence[];
};

type PdfBlock = {
  block_id: string;
  text: string;
  start: number;
  end: number;
};

type PdfPage = {
  page_num: number;
  text: string;
  blocks: PdfBlock[];
  terms: PdfTerm[];
};

type PdfAnnotateResponse = {
  document_id: string;
  filename: string;
  page_count: number;
  src_used?: string;
  predicted_src?: string | null;
  predicted_src_confidence?: number | null;
  src_warning?: boolean;
  src_warning_reason?: string;
  pages: PdfPage[];
  summary?: {
    flagged_term_count: number;
    pages_with_flags: number;
  };
};

type ExplainResponse = {
  term: string;
  short_explanation: string;
  long_explanation: string;
  closest_analog: string | null;
  caution_label: string;
  cache_hit: boolean;
  model: string;
  term_id?: string;
  page_num?: number;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "https://scibabel-backend-523773192713.us-central1.run.app";

function renderHighlightedText(text: string, terms: PdfTerm[], onSelect: (t: PdfTerm) => void) {
  if (!text.trim()) {
    return <p className="text-sm text-slate-500">No extractable text on this page.</p>;
  }
  if (!terms.length) {
    return <p className="whitespace-pre-wrap text-sm leading-7 text-slate-800">{text}</p>;
  }

  const sorted = [...terms].sort((a, b) => a.start - b.start);
  const chunks: ReactNode[] = [];
  let cursor = 0;

  sorted.forEach((term, idx) => {
    const s = Math.max(0, Math.min(text.length, term.start));
    const e = Math.max(s, Math.min(text.length, term.end));
    if (s > cursor) chunks.push(<span key={`plain-${idx}-${cursor}`}>{text.slice(cursor, s)}</span>);
    const phrase = text.slice(s, e);
    chunks.push(
      <button
        key={term.term_id || `term-${idx}`}
        type="button"
        className={`mx-0.5 rounded-md px-1.5 py-0.5 text-left transition-colors ${
          term.flagged
            ? "bg-amber-300 text-amber-950 underline decoration-2 underline-offset-2 hover:bg-amber-200"
            : "bg-slate-200 text-slate-900 hover:bg-slate-300"
        }`}
        onClick={() => onSelect(term)}
      >
        {phrase}
      </button>
    );
    cursor = e;
  });

  if (cursor < text.length) chunks.push(<span key="tail">{text.slice(cursor)}</span>);
  return <p className="whitespace-pre-wrap text-sm leading-7 text-slate-800">{chunks}</p>;
}

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [fileUrl, setFileUrl] = useState<string | null>(null);
  const [src, setSrc] = useState<SrcDomain>("auto");
  const [tgt, setTgt] = useState<Domain>("PM");
  const [audience, setAudience] = useState<AudienceLevel>("grad");
  const [maxTerms, setMaxTerms] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [doc, setDoc] = useState<PdfAnnotateResponse | null>(null);
  const [pageNum, setPageNum] = useState(1);
  const [selectedTerm, setSelectedTerm] = useState<PdfTerm | null>(null);
  const [explain, setExplain] = useState<ExplainResponse | null>(null);
  const [loadingExplain, setLoadingExplain] = useState(false);

  const domains: Domain[] = ["CSM", "PM", "CHEM", "CHEME"];
  const currentPage = useMemo(() => doc?.pages.find((p) => p.page_num === pageNum) ?? null, [doc, pageNum]);

  const onFileChange = (next: File | null) => {
    setFile(next);
    setDoc(null);
    setSelectedTerm(null);
    setExplain(null);
    if (!next) {
      setFileUrl(null);
      return;
    }
    setFileUrl(URL.createObjectURL(next));
  };

  const onAnnotatePdf = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setDoc(null);
    setSelectedTerm(null);
    setExplain(null);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("src", src);
      form.append("tgt", tgt);
      form.append("audience_level", audience);
      form.append("max_terms", String(maxTerms));
      form.append("same_field_mode", "normal");

      const res = await fetch(`${API_BASE}/pdf/annotate`, { method: "POST", body: form });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail ?? `PDF annotate failed (${res.status})`);
      }
      const out: PdfAnnotateResponse = await res.json();
      setDoc(out);
      setPageNum(1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const onExplain = async (term: PdfTerm) => {
    if (!doc || !currentPage) return;
    setLoadingExplain(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/pdf/explain`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: doc.document_id,
          page_num: currentPage.page_num,
          term_id: term.term_id,
          tgt,
          src: doc.src_used || src,
          audience_level: audience,
          detail: "long",
        }),
      });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload?.detail ?? `Explain failed (${res.status})`);
      }
      const out: ExplainResponse = await res.json();
      setExplain(out);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unexpected error");
    } finally {
      setLoadingExplain(false);
    }
  };

  const onCopyAnnotationJson = async () => {
    if (!doc) return;
    const raw = JSON.stringify(doc, null, 2);
    try {
      await navigator.clipboard.writeText(raw);
    } catch {
      setError("Failed to copy JSON to clipboard.");
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-amber-50 via-orange-50 to-white px-4 py-8 md:px-8">
      <div className="mx-auto max-w-7xl">
        <h1 className="text-3xl font-bold tracking-tight text-slate-900 md:text-4xl">SciBabel PDF Term Lens</h1>
        <p className="mt-2 max-w-3xl text-sm text-slate-600">
          Upload a paper PDF, detect cross-domain jargon locally, and request detailed explanations only when you click a term.
        </p>

        <section className="mt-6 rounded-2xl border border-orange-200 bg-white/90 p-4 shadow-sm md:p-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-6">
            <label className="md:col-span-2">
              <span className="mb-1 block text-xs font-semibold uppercase text-slate-500">PDF File</span>
              <input
                type="file"
                accept="application/pdf"
                onChange={(e) => onFileChange(e.target.files?.[0] ?? null)}
                className="w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"
              />
            </label>

            <label>
              <span className="mb-1 block text-xs font-semibold uppercase text-slate-500">Source</span>
              <select value={src} onChange={(e) => setSrc(e.target.value as SrcDomain)} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
                <option value="auto">auto</option>
                {domains.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </label>

            <label>
              <span className="mb-1 block text-xs font-semibold uppercase text-slate-500">Target</span>
              <select value={tgt} onChange={(e) => setTgt(e.target.value as Domain)} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
                {domains.map((d) => (
                  <option key={d} value={d}>{d}</option>
                ))}
              </select>
            </label>

            <label>
              <span className="mb-1 block text-xs font-semibold uppercase text-slate-500">Audience</span>
              <select value={audience} onChange={(e) => setAudience(e.target.value as AudienceLevel)} className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
                <option value="undergrad">undergrad</option>
                <option value="grad">grad</option>
                <option value="expert">expert</option>
              </select>
            </label>

            <label>
              <span className="mb-1 block text-xs font-semibold uppercase text-slate-500">Max Terms / Page</span>
              <input
                type="number"
                min={1}
                max={20}
                value={maxTerms}
                onChange={(e) => setMaxTerms(Number(e.target.value || 8))}
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm"
              />
            </label>
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              type="button"
              disabled={loading || !file}
              onClick={onAnnotatePdf}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-500"
            >
              {loading ? "Analyzing PDF..." : "Annotate PDF"}
            </button>
            {doc && (
              <>
                <button
                  type="button"
                  onClick={onCopyAnnotationJson}
                  className="rounded border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-700"
                >
                  Copy Annotation JSON
                </button>
                <p className="text-xs text-slate-600">doc: <span className="font-mono">{doc.document_id}</span> | pages: {doc.page_count} | flagged: {doc.summary?.flagged_term_count ?? 0}</p>
              </>
            )}
          </div>

          {doc && (
            <div className="mt-3 rounded-lg border border-sky-200 bg-sky-50 p-3 text-xs text-slate-700">
              <p className="font-semibold uppercase tracking-wide text-sky-900">QA Mode</p>
              <p className="mt-1">predicted_src: <span className="font-mono">{doc.predicted_src || "n/a"}</span></p>
              <p>confidence: {typeof doc.predicted_src_confidence === "number" ? doc.predicted_src_confidence.toFixed(3) : "n/a"}</p>
              <p>ambiguous: {doc.src_warning ? "yes" : "no"} ({doc.src_warning_reason || "none"})</p>
              <p>flagged_terms: {doc.summary?.flagged_term_count ?? 0}</p>
              <p>page_count: {doc.page_count}</p>
            </div>
          )}

          {error && <p className="mt-3 rounded-md border border-rose-300 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</p>}
        </section>

        <section className="mt-6 grid grid-cols-1 gap-4 lg:grid-cols-12">
          <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-5">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Document View</h2>
            {fileUrl ? (
              <object data={fileUrl} type="application/pdf" className="mt-3 h-[360px] w-full rounded-lg border border-slate-200">
                <p className="p-3 text-sm text-slate-600">PDF preview unavailable in this browser.</p>
              </object>
            ) : (
              <p className="mt-3 text-sm text-slate-500">Upload a PDF to preview it here.</p>
            )}

            {doc && (
              <div className="mt-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold text-slate-800">Extracted Text and Highlights</h3>
                  <div className="flex items-center gap-2">
                    <button type="button" onClick={() => setPageNum((p) => Math.max(1, p - 1))} disabled={pageNum <= 1} className="rounded border border-slate-300 px-2 py-1 text-xs disabled:opacity-40">Prev</button>
                    <span className="text-xs text-slate-600">Page {pageNum} / {doc.page_count}</span>
                    <button type="button" onClick={() => setPageNum((p) => Math.min(doc.page_count, p + 1))} disabled={pageNum >= doc.page_count} className="rounded border border-slate-300 px-2 py-1 text-xs disabled:opacity-40">Next</button>
                  </div>
                </div>
                <div className="mt-2 max-h-[360px] overflow-auto rounded-lg border border-slate-200 bg-slate-50 p-3">
                  {currentPage ? renderHighlightedText(currentPage.text, currentPage.terms, (t) => setSelectedTerm(t)) : null}
                </div>
              </div>
            )}
          </article>

          <article className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-7">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">Term Inspector</h2>
            {!doc && <p className="mt-3 text-sm text-slate-500">Run PDF annotation to see extracted terms.</p>}

            {doc && currentPage && (
              <div className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
                <div className="max-h-[420px] overflow-auto rounded-lg border border-slate-200 p-3">
                  <h3 className="mb-2 text-sm font-semibold text-slate-700">Page {currentPage.page_num} Terms</h3>
                  {currentPage.terms.length === 0 && <p className="text-sm text-slate-500">No terms found on this page.</p>}
                  {currentPage.terms.map((term) => (
                    <button
                      key={term.term_id}
                      type="button"
                      onClick={() => {
                        setSelectedTerm(term);
                        setExplain(null);
                      }}
                      className={`mb-2 block w-full rounded-lg border px-3 py-2 text-left text-sm transition ${selectedTerm?.term_id === term.term_id ? "border-amber-500 bg-amber-50" : "border-slate-200 bg-white hover:bg-slate-50"}`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium text-slate-900">{term.surface_term || term.term}</span>
                        <span className={`rounded px-2 py-0.5 text-[11px] ${term.flagged ? "bg-amber-200 text-amber-900" : "bg-slate-200 text-slate-700"}`}>{term.flagged ? "flagged" : "ok"}</span>
                      </div>
                      <p className="mt-1 text-xs text-slate-500">{term.canonical_term || term.term}</p>
                    </button>
                  ))}
                </div>

                <div className="rounded-lg border border-slate-200 p-3">
                  {!selectedTerm && <p className="text-sm text-slate-500">Click a highlighted term to inspect and explain.</p>}
                  {selectedTerm && (
                    <div>
                      <h3 className="text-base font-semibold text-slate-900">{selectedTerm.surface_term || selectedTerm.term}</h3>
                      <p className="mt-1 text-xs text-slate-500">canonical: {selectedTerm.canonical_term || selectedTerm.term}</p>
                      <p className="mt-2 text-xs text-slate-500">reason: {selectedTerm.reason || "n/a"}</p>

                      <button
                        type="button"
                        onClick={() => onExplain(selectedTerm)}
                        disabled={loadingExplain}
                        className="mt-3 rounded bg-slate-900 px-3 py-2 text-xs font-semibold text-white disabled:opacity-60"
                      >
                        {loadingExplain ? "Generating..." : "Explain This Term"}
                      </button>

                      {explain && (
                        <div className="mt-3 rounded-md border border-sky-200 bg-sky-50 p-3">
                          <p className="text-xs font-semibold uppercase text-sky-800">Explanation</p>
                          <p className="mt-2 text-sm text-slate-800">{explain.long_explanation || explain.short_explanation}</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            )}
          </article>
        </section>
      </div>
    </main>
  );
}
