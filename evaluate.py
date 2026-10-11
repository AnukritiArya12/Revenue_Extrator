"""Compare the system's answers with an answer file (written by a person) and report accuracy.

    python -m eval.evaluate                  # uses data/ground_truth.csv
    python -m eval.evaluate answers.csv      # any answer file, e.g. from a financial portal
    python -m eval.evaluate --search         # also test the search step: plain BM25 vs the improved search

Columns of the answer file (names are detected automatically):
    company | name                       required
    value | total_revenue | answer       required: the correct value. NONE = the system must say NOT FOUND
    year | fy | report                   optional: the report, e.g. FY26
    metric | item                        optional: default revenue ("profit after tax", "net_loss" also work)
    unit                                 optional: lets the file use another unit (crore, lakh crore, million, billion ...)
    scope                                optional: consolidated (default) or standalone

Printed: one line per answer, then accuracy overall / by metric / by confidence level, and the failures.
Saved:   data/processed/evaluation_report.csv
Result words: MATCH | CLOSE (under 0.5%, not counted) | DIFFERENT | NOT FOUND | no report in the system
"""
import csv
import re
import sys
from collections import defaultdict

from src.extract.engine import UNIT_FACTOR, extract_financial_metric, retrieve
from src.extract.metrics import METRICS
from src.extract.query import find_metric, label_year, list_reports
from src.retrieve.bm25_search import ReportNotFound, search

COLUMNS = {"company": ["company", "name", "company_name"],
           "year": ["year", "fy", "fiscal_year", "report"],
           "value": ["value", "total_revenue", "answer", "sales", "revenue"],
           "metric": ["metric", "item"], "unit": ["unit", "units"], "scope": ["scope"]}
NONE_WORDS = {"none", "not found", "notfound", "n/a", "na"}
SUFFIX = {"limited", "ltd", "incorporated", "inc", "corporation", "corp", "plc", "company", "co"}
UNIT_WORDS = {"cr": "crore", "crores": "crore", "lakhs": "lakh", "millions": "million", "mn": "million",
              "billions": "billion", "bn": "billion", "thousands": "thousand"}
EXTRA_FACTOR = {"lakh crore": 1e12, "lakh crores": 1e12, "trillion": 1e12}


def column(header, wanted):
    low = {h.lower().strip(): h for h in header}
    return next((low[n] for n in COLUMNS[wanted] if n in low), None)


def norm_name(s):
    return "".join(w for w in re.sub(r"[^a-z0-9 ]", "", s.lower()).split() if w not in SUFFIX)


def same_company(a, b):
    a, b = norm_name(a), norm_name(b)
    return bool(a and b) and (a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a))))


def number(text):
    """'2,67,022' -> 267022.0 ; '(5,000)' and '-5,000' -> -5000.0 ; anything else -> None"""
    t = str(text or "").strip()
    digits = re.sub(r"[^\d.]", "", t)
    if not digits:
        return None
    try:
        v = float(digits)
    except ValueError:
        return None
    return -v if t.startswith("(") or t.startswith("-") else v


def factor(unit):
    u = " ".join((unit or "").lower().split())
    u = UNIT_WORDS.get(u, u)
    return EXTRA_FACTOR.get(u) or UNIT_FACTOR.get(u)


def compare(answer, answer_unit, value, unit):
    fa, fs = factor(answer_unit), factor(unit)
    if fa and fs:
        rel = abs(answer * fa - value * fs) / max(abs(answer * fa), 1)
        return "MATCH" if rel <= 5e-5 else f"CLOSE ({rel * 100:.2f}% off)" if rel <= 5e-3 else f"DIFFERENT ({rel * 100:.1f}% off)"
    return "MATCH" if abs(answer - value) < 1e-6 else "DIFFERENT (no unit in the file: compared as printed)"


def pct(a, b):
    return f"{100 * a / b:.0f}%" if b else "-"


def print_report(records):
    scored = [r for r in records if r["correct"] is not None]
    ok = sum(r["correct"] for r in scored)
    print(f"\nACCURACY  {ok}/{len(scored)} correct ({pct(ok, len(scored))})")
    print("\nBy metric")
    by_metric = defaultdict(list)
    for r in scored:
        by_metric[r["metric"]].append(r["correct"])
    for m, v in sorted(by_metric.items()):
        print(f"  {m:<24}{sum(v)}/{len(v)}  {pct(sum(v), len(v))}")
    print("\nBy confidence  (a useful HIGH should be more accurate than MEDIUM and LOW)")
    by_conf = defaultdict(list)
    for r in scored:
        by_conf[r["confidence"]].append(r["correct"])
    for level in ("HIGH", "MEDIUM", "LOW", "NONE"):
        if level in by_conf:
            v = by_conf[level]
            print(f"  {level:<10}{sum(v)}/{len(v)} correct  ({pct(sum(v), len(v))})")
    flagged = [r for r in scored if r["status"] == "VERIFY"]
    print(f"\nFlagged VERIFY: {len(flagged)}   (of which correct anyway: {sum(r['correct'] for r in flagged)})")
    failures = [r for r in scored if not r["correct"]]
    if failures:
        print("\nFAILURES")
        for r in failures:
            print(f"  {r['company']} {r['metric']}: answer {r['expected']}, system {r['system'] or 'NOT FOUND'} "
                  f"(page {r['page']}, line '{r['label']}', {r['confidence']}) -> {r['result']}")
            if r["warnings"]:
                print(f"      why: {r['warnings']}")


def search_test(checks):
    """Does the search put a chunk with the right number in the top 5? plain BM25 vs the improved search."""
    print("\nSearch step (is the right number in the top-5 chunks?)")
    print(f"{'setup':<44}{'hit@1':<10}{'hit@5':<10}")
    setups = (("BM25 plain", lambda c, rep, m: search(c, rep, " ".join(a for t in METRICS[m]["aliases"] for a in t), k=5)),
              ("BM25 + synonyms + phrase bonus + page boost", lambda c, rep, m: retrieve(c, rep, METRICS[m], k=5)))
    for name, run in setups:
        top1 = top5 = 0
        for company, report, metric, raw in checks:
            pat = re.compile(r"(?<![\d.])" + re.escape(re.sub(r"[^\d.]", "", raw)) + r"(?!\d)")
            hits = [bool(pat.search(c["text"].replace(",", ""))) for _, c in run(company, report, metric)]
            top1 += hits[:1] == [True]
            top5 += any(hits)
        print(f"{name:<44}{top1}/{len(checks):<8}{top5}/{len(checks):<8}")

        
def run_all_reports():
    import csv
    import os

    output_path = "data/processed/all_reports_revenue.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reports = list_reports()
    rows = []

    for company, report in reports:
        print(f"Extracting revenue: {company} - {report}")

        try:
            result = extract_financial_metric(
                company, report, "revenue"
            )

            rows.append({
                "company": company,
                "report": report,
                "metric": "revenue",
                "status": result.get("status", ""),
                "value": result.get("value", ""),
                "value_float": result.get("value_float", ""),
                "unit": result.get("unit", ""),
                "currency": result.get("currency", ""),
                "year": result.get("year", ""),
                "scope": result.get("scope", ""),
                "page": result.get("page", ""),
                "label": result.get("label", ""),
                "evidence": " || ".join(
                    result.get("evidence", []) or []
                ),
                "evidence_row": result.get("evidence_row", ""),
                "confidence": result.get("confidence", ""),
                "needs_verification": result.get(
                    "needs_verification", ""
                ),
                "message": result.get("message", ""),
                "reasons": " | ".join(
                    result.get("reasons", []) or []
                ),
            })

        except Exception as error:
            rows.append({
                "company": company,
                "report": report,
                "metric": "revenue",
                "status": "ERROR",
                "message": str(error),
            })

    columns = [
        "company", "report", "metric", "status",
        "value", "value_float", "unit", "currency",
        "year", "scope", "page", "label", "evidence",
        "evidence_row", "confidence", "needs_verification",
        "message", "reasons"
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(1 for row in rows if row["status"] == "OK")
    print(f"\nTotal reports: {len(reports)}")
    print(f"Successful extractions: {successful}")
    print(f"Other results/errors: {len(rows) - successful}")
    print(f"Saved to: {output_path}")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    path = args[0] if args else "data/ground_truth.csv"
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = [r for r in reader if any((v or "").strip() for v in r.values())]
    except FileNotFoundError:
        sys.exit(f"Answer file not found: {path}")
    cols = {k: column(header, k) for k in COLUMNS}
    if not (cols["company"] and cols["value"]):
        sys.exit(f"Need a company column and a value column; found: {header}")

    reports = list_reports()
    records, checks = [], []
    print(f"{'company':<16}{'metric':<22}{'answer':<14}{'system':<14}{'unit':<9}{'page':<6}{'conf':<8}result")
    for row in rows:
        name, raw = row[cols["company"]].strip(), (row[cols["value"]] or "").strip()
        get = lambda k: (row.get(cols[k]) or "").strip() if cols[k] else ""
        digits = re.sub(r"\D", "", get("year"))[-2:]
        found_reports = [(c, y) for c, y in reports if same_company(c, name) and (not digits or re.sub(r"\D", "", y)[-2:] == digits)]
        metric_text = get("metric") or "revenue"
        metric = metric_text.lower().replace(" ", "_")
        metric = metric if metric in METRICS else find_metric(metric_text)
        rec = {"company": name, "report": "", "metric": metric or metric_text, "expected": raw, "system": "",
               "unit": "", "page": "", "label": "", "status": "", "confidence": "", "result": "", "warnings": "",
               "correct": None}
        if not found_reports or metric is None:
            rec["result"] = "no report in the system" if not found_reports else f"unknown metric '{metric_text}'"
            records.append(rec)
            print(f"{name:<16}{rec['metric']:<22}{raw:<14}{'-':<14}{'-':<9}{'-':<6}{'-':<8}{rec['result']}")
            continue
        company, report = max(found_reports, key=lambda cy: label_year(cy[1]) or 0)
        try:
            r = extract_financial_metric(company, report, metric, scope=get("scope") or "consolidated")
        except ReportNotFound:
            rec["result"] = "no report in the system"
            records.append(rec)
            continue
        found = r["status"] in ("OK", "VERIFY")
        if raw.lower() in NONE_WORDS:
            correct = not found
            result = "OK (correctly NOT FOUND)" if correct else "WRONG (should be NOT FOUND)"
        elif number(raw) is None:
            correct, result = None, "no answer given (check by hand)"
        else:
            result = compare(number(raw), get("unit"), r["value_float"], r["unit"]) if found else "NOT FOUND"
            correct = result == "MATCH"
            result = ("OK  " if correct else "WRONG ") + result
            checks.append((company, report, metric, raw))
        rec.update({"report": report, "system": r["value"] or "", "unit": r["unit"] or "", "page": r["page"] or "",
                    "label": r["label"] or "", "status": r["status"], "confidence": r["confidence"],
                    "result": result, "correct": correct, "warnings": " | ".join(r["warnings"] or [r["message"]]).strip(" |")})
        records.append(rec)
        flag = "  (VERIFY)" if r["status"] == "VERIFY" else ""
        print(f"{name:<16}{metric:<22}{raw:<14}{(r['value'] or '-'):<14}{str(r['unit']):<9}{str(r['page']):<6}{r['confidence']:<8}{result}{flag}")

    print_report(records)
    out = "data/processed/evaluation_report.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        fields = ["company", "report", "metric", "expected", "system", "unit", "page", "label", "status",
                  "confidence", "result", "warnings"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved -> {out}")
    if "--search" in sys.argv and checks:
        search_test(checks)

if __name__ == "__main__":
    import sys

    if "--all-reports" in sys.argv:
        run_all_reports()
    else:
        main()
