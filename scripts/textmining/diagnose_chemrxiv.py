from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

API_PATH = "/engage/chemrxiv/public-api/v1/items"


def _ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _preview_body(resp: requests.Response, n: int = 300) -> str:
    text = (resp.text or "").strip()
    if not text:
        return "<empty>"
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[:n] + "..."


def detect_waf_signals(headers: dict[str, str], body_preview: str, content_type: str) -> dict[str, Any]:
    h = {k.lower(): str(v) for k, v in headers.items()}
    b = (body_preview or "").lower()
    cf_headers = any(k in h for k in ["cf-ray", "cf-cache-status"]) or "cloudflare" in h.get("server", "").lower()
    cf_body = any(x in b for x in ["attention required", "cloudflare", "captcha", "challenge"])
    html_on_api = "text/html" in (content_type or "").lower()
    waf_block = bool(cf_headers or cf_body or html_on_api)
    return {
        "waf_block": waf_block,
        "signals": {
            "cf_headers": cf_headers,
            "cf_body": cf_body,
            "html_on_api": html_on_api,
        },
    }


@dataclass
class ExpResult:
    experiment: str
    method: str
    requested_url: str
    final_url: str
    status_code: int | None
    elapsed_sec: float
    redirect_chain: list[dict[str, Any]]
    response_headers_subset: dict[str, str]
    content_type: str
    body_preview: str
    error: str | None
    cookies_set: list[str]
    waf_signals: dict[str, Any]


def _subset_headers(resp: requests.Response) -> dict[str, str]:
    wanted = [
        "server",
        "content-type",
        "location",
        "cf-ray",
        "cf-cache-status",
        "x-cache",
        "x-powered-by",
        "set-cookie",
        "retry-after",
    ]
    out: dict[str, str] = {}
    for k in wanted:
        if k in resp.headers:
            out[k] = str(resp.headers.get(k, ""))
    return out


def _redirect_chain(resp: requests.Response) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    for h in list(resp.history):
        chain.append(
            {
                "status_code": h.status_code,
                "url": h.url,
                "location": h.headers.get("Location", ""),
            }
        )
    return chain


def run_one(
    *,
    experiment: str,
    method: str,
    url: str,
    params: dict[str, Any] | None,
    headers: dict[str, str] | None,
    warmup_url: str | None = None,
    timeout: int = 20,
) -> ExpResult:
    s = requests.Session()
    if headers:
        s.headers.update(headers)

    cookies_set: list[str] = []
    if warmup_url:
        try:
            s.get(warmup_url, timeout=timeout)
            cookies_set = [c.name for c in s.cookies]
        except Exception:
            cookies_set = [c.name for c in s.cookies]

    t0 = time.time()
    try:
        if method.upper() == "HEAD":
            resp = s.head(url, params=params, timeout=timeout, allow_redirects=True)
        else:
            resp = s.get(url, params=params, timeout=timeout, allow_redirects=True)
        elapsed = time.time() - t0
        content_type = str(resp.headers.get("Content-Type", ""))
        preview = _preview_body(resp)
        waf = detect_waf_signals(dict(resp.headers), preview, content_type)
        return ExpResult(
            experiment=experiment,
            method=method.upper(),
            requested_url=requests.Request(method.upper(), url, params=params).prepare().url or url,
            final_url=resp.url,
            status_code=int(resp.status_code),
            elapsed_sec=round(elapsed, 3),
            redirect_chain=_redirect_chain(resp),
            response_headers_subset=_subset_headers(resp),
            content_type=content_type,
            body_preview=preview,
            error=None,
            cookies_set=cookies_set,
            waf_signals=waf,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        return ExpResult(
            experiment=experiment,
            method=method.upper(),
            requested_url=requests.Request(method.upper(), url, params=params).prepare().url or url,
            final_url="",
            status_code=None,
            elapsed_sec=round(elapsed, 3),
            redirect_chain=[],
            response_headers_subset={},
            content_type="",
            body_preview="",
            error=str(exc),
            cookies_set=cookies_set,
            waf_signals={"waf_block": False, "signals": {}},
        )


def summarize_conclusion(results: list[ExpResult]) -> str:
    statuses = [r.status_code for r in results if r.status_code is not None]
    any_200 = any(s == 200 for s in statuses)
    any_403 = any(s == 403 for s in statuses)
    any_401 = any(s == 401 for s in statuses)
    waf_hits = [r for r in results if bool(r.waf_signals.get("waf_block"))]
    html_api = [r for r in results if "text/html" in (r.content_type or "").lower()]
    auth_hints = [
        r
        for r in results
        if any(x in (r.body_preview or "").lower() for x in ["unauthorized", "forbidden", "login", "authentication"]) and not r.waf_signals.get("waf_block")
    ]

    warmup_rows = [r for r in results if r.experiment.startswith("E_")]
    warmup_success = any((r.status_code == 200 and len(r.cookies_set) > 0) for r in warmup_rows)
    if warmup_success:
        return "Likely needs auth"

    if any_403 and waf_hits:
        return "Likely WAF block"
    if any_401 or auth_hints:
        return "Likely needs auth"
    if any_403 and not any_200:
        return "Likely platform-side block or API access policy"
    if any_200:
        return "Endpoint reachable; likely client header/flow issue in failing mode"
    if html_api and not any_200:
        return "Likely endpoint/param change (non-JSON API behavior)"
    return "Inconclusive; check network/proxy and endpoint documentation"


def next_steps(conclusion: str) -> list[str]:
    if "WAF" in conclusion:
        return [
            "Programmatic access is likely blocked by platform protections.",
            "Consider alternate public data sources (e.g., OpenAlex/Semantic Scholar) or manual exports.",
            "Contact ChemRxiv/OpenEngage support for API access guidance.",
        ]
    if "endpoint/param" in conclusion:
        return [
            "Check official ChemRxiv/OpenEngage API documentation for endpoint/param changes.",
            "Validate expected response schema and required query parameters.",
        ]
    if "header/flow" in conclusion:
        return [
            "Integrate the successful request pattern (headers/session warmup) into fetcher.",
            "Keep strict diagnostics for regressions.",
        ]
    return [
        "Capture additional traces from another network/IP.",
        "Verify DNS/proxy/firewall effects and endpoint ownership.",
    ]


def guidance_matrix(warmup_worked: bool) -> list[str]:
    lines = [
        "### If WAF detected",
        "- Programmatic access may be blocked by platform protections.",
        "- Recommend using OpenAlex/Semantic Scholar or manual exports.",
        "",
        "### If endpoint/param mismatch suspected",
        "- Verify official ChemRxiv/OpenEngage documentation for endpoint and pagination params.",
        "",
        "### If cookie warmup works",
    ]
    if warmup_worked:
        lines.append("- Warmup appears effective; integrate warmup session flow into fetcher (do not use fallback here).")
    else:
        lines.append("- Warmup did not improve status in this run; likely not a cookie bootstrap issue.")
    return lines


def run_diagnostics(out_md: str | None = None, timeout: int = 20) -> tuple[Path, Path, str]:
    stamp = _ts()
    out_dir = Path("reports/chemrxiv")
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / f"requests_{stamp}.jsonl"
    md_path = out_dir / (f"diagnosis_{stamp}.md" if out_md is None else Path(out_md).name)

    https_url = f"https://chemrxiv.org{API_PATH}"
    http_url = f"http://chemrxiv.org{API_PATH}"
    params_skip_limit = {"limit": 1, "skip": 0}

    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    ref_headers = {
        **browser_headers,
        "Referer": "https://chemrxiv.org/engage/chemrxiv/public-dashboard",
        "Origin": "https://chemrxiv.org",
    }
    ref_headers_http = {
        **browser_headers,
        "Referer": "http://chemrxiv.org/engage/chemrxiv/public-dashboard",
        "Origin": "http://chemrxiv.org",
    }

    results: list[ExpResult] = []

    # A/B
    results.append(run_one(experiment="A_https_minimal", method="GET", url=https_url, params=params_skip_limit, headers=None, timeout=timeout))
    results.append(run_one(experiment="B_http_minimal", method="GET", url=http_url, params=params_skip_limit, headers=None, timeout=timeout))

    # C
    results.append(run_one(experiment="C_https_browser_headers", method="GET", url=https_url, params=params_skip_limit, headers=browser_headers, timeout=timeout))
    results.append(run_one(experiment="C_http_browser_headers", method="GET", url=http_url, params=params_skip_limit, headers=browser_headers, timeout=timeout))

    # D
    results.append(run_one(experiment="D_https_referer_origin", method="GET", url=https_url, params=params_skip_limit, headers=ref_headers, timeout=timeout))
    results.append(run_one(experiment="D_http_referer_origin", method="GET", url=http_url, params=params_skip_limit, headers=ref_headers_http, timeout=timeout))

    # E
    results.append(
        run_one(
            experiment="E_https_warmup_cookie_session",
            method="GET",
            url=https_url,
            params=params_skip_limit,
            headers=browser_headers,
            warmup_url="https://chemrxiv.org/engage/chemrxiv/public-dashboard",
            timeout=timeout,
        )
    )
    results.append(
        run_one(
            experiment="E_http_warmup_cookie_session",
            method="GET",
            url=http_url,
            params=params_skip_limit,
            headers=browser_headers,
            warmup_url="http://chemrxiv.org/engage/chemrxiv/public-dashboard",
            timeout=timeout,
        )
    )

    # F
    results.append(run_one(experiment="F_https_head", method="HEAD", url=https_url, params=params_skip_limit, headers=browser_headers, timeout=timeout))

    # G
    for i in range(3):
        results.append(
            run_one(
                experiment=f"G_https_retry_{i+1}",
                method="GET",
                url=https_url,
                params=params_skip_limit,
                headers=browser_headers,
                timeout=timeout,
            )
        )
        time.sleep(2**i)

    # H
    variants = [
        ("H_param_skip_limit", {"skip": 0, "limit": 1}),
        ("H_param_offset_limit", {"offset": 0, "limit": 1}),
        ("H_param_page_size", {"page": 1, "size": 1}),
    ]
    for name, p in variants:
        results.append(run_one(experiment=name, method="GET", url=https_url, params=p, headers=browser_headers, timeout=timeout))

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    conclusion = summarize_conclusion(results)
    warmup_rows = [r for r in results if r.experiment.startswith("E_")]
    warmup_worked = any(r.status_code == 200 for r in warmup_rows)
    lines = [
        "# ChemRxiv API Diagnostic Report",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Raw metadata: {jsonl_path}",
        "",
        "## Conclusion",
        f"- {conclusion}",
        "",
        "## Experiment Results",
        "",
    ]
    for r in results:
        lines.extend(
            [
                f"### {r.experiment}",
                f"- method: {r.method}",
                f"- requested_url: {r.requested_url}",
                f"- final_url: {r.final_url or '<none>'}",
                f"- status_code: {r.status_code}",
                f"- elapsed_sec: {r.elapsed_sec}",
                f"- cookies_set: {r.cookies_set}",
                f"- content_type: {r.content_type}",
                f"- waf_signals: {r.waf_signals}",
                f"- redirect_chain: {json.dumps(r.redirect_chain, ensure_ascii=False)}",
                f"- response_headers_subset: {json.dumps(r.response_headers_subset, ensure_ascii=False)}",
                f"- body_preview: {r.body_preview or '<none>'}",
                f"- error: {r.error or '<none>'}",
                "",
            ]
        )

    lines.append("## Next-step guidance")
    for s in next_steps(conclusion):
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Conditional guidance")
    lines.extend(guidance_matrix(warmup_worked))

    md_full = out_dir / f"diagnosis_{stamp}.md"
    md_full.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if out_md:
        out_custom = Path(out_md)
        out_custom.parent.mkdir(parents=True, exist_ok=True)
        out_custom.write_text(md_full.read_text(encoding="utf-8"), encoding="utf-8")
        md_path = out_custom

    return jsonl_path, md_path, conclusion


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose ChemRxiv API 403 behavior")
    parser.add_argument("--out", default="", help="Optional markdown report output path")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    jsonl_path, md_path, conclusion = run_diagnostics(out_md=args.out or None, timeout=args.timeout)
    print(f"ChemRxiv diagnosis complete: {md_path}")
    print(f"Raw request metadata: {jsonl_path}")
    print(f"Conclusion: {conclusion}")


if __name__ == "__main__":
    main()
