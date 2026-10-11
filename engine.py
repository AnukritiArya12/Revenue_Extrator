"""ONE generic pipeline for every financial metric.

    extract_financial_metric(company, report, metric, year=None, scope="consolidated")

The metric comes from the user's question (query.py); its rules are DATA in metrics.py.
Nothing here mentions revenue or profit. The steps are the same for every metric:

  1. candidate pages   BM25 search + every page tagged as an income statement
  2. read rows         label + numbers per row (statement_parser)
  3. match             does the row label equal one of the metric's aliases?
  4. year column       read the year headings, take the column of the requested year
  5. validate + rank   statement page, scope, arithmetic proof, year found, ...
  6. answer            best candidate + evidence, or VERIFY if two rivals are equally plausible
"""
import logging
import re
from collections import defaultdict

from src.extract.metrics import IDENTITIES, METRICS
from src.extract.statement_parser import STOP, find_proofs, parse_rows, to_float
from src.ingest.section_tagger import STATEMENT_HEAD
from src.retrieve.bm25_search import ReportNotFound, load_chunks
from src.retrieve.hybrid_search import HybridSearch

log = logging.getLogger("extractor")

hybrid_search = HybridSearch()


STATEMENT_SECTIONS = ("income_statement", "highlights")
MAX_LEFT_NOISE = 5            # extra words allowed in front of an alias ("Total net revenues")
AMBIGUITY_GAP = 3             # two different values within this score gap -> VERIFY

YEAR_RE = re.compile(r"(20\d{2})\s*[-\u2013/]\s*(\d{2})\b|\b(20\d{2})\b")
UNIT_RE = re.compile(r"\b(crores?|crs?\b\.?|lakhs?|millions?|billions?|thousands?)\b", re.I)
UNIT_NORMAL = {"crore": "crore", "crores": "crore", "cr": "crore", "crs": "crore",
               "lakh": "lakh", "lakhs": "lakh", "million": "million", "millions": "million",
               "billion": "billion", "billions": "billion", "thousand": "thousand",
               "thousands": "thousand"}
UNIT_FACTOR = {"thousand": 1e3, "lakh": 1e5, "million": 1e6, "crore": 1e7, "billion": 1e9}
YOY_LIMIT = 5.0               # a year-on-year change above 5x (or below 1/5) looks like a wrong column or unit
LEVELS = ["LOW", "MEDIUM", "HIGH"]


# ------------------------------------------------------------------ small helpers
def search_query(cfg: dict, company: str = None) -> str:
    """Words for the BM25 search: the metric's own names, its synonyms, and statement words."""
    words = [a for tier in cfg["aliases"] for a in tier] + cfg.get("search_terms", [])
    prefix = f"{company} " if company else ""
    return prefix + " ".join(words) + " statement of profit and loss income"


def retrieve(company: str, report: str, cfg: dict, k: int = 5, boost: bool = True):
    """Hybrid BM25 + semantic retrieval with metric aliases."""
    phrases = [a for tier in cfg["aliases"] for a in tier]
    if company:
        phrases.append(company)

    results = hybrid_search.search(
        company=company,
        year=report,
        query=search_query(cfg, company),
        k=k,
        boost=boost,
        phrases=phrases,
    )

    return [(r.get("rerank_score", r.get("hybrid_score", 0.0)), r) for r in results]


_HEADER_NOISE_RE = re.compile(
    r"^(in\s+(?:millions?|thousands?|billions?|crores?|lakhs?)\b"
    r"|except\s+(?:for\s+)?per\s+share\s+amounts?\b"
    r"|for\s+the\s+years?\s+ended\b"
    r"|continuing\s+operations\b"
    r"|consolidated\s+statement\s+of\b|\s+)+",
    re.I
)

_NOTE_TAIL_WORDS = {
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "notes", "note", "and",
    "contd", "continued", "see", "unaudited", "audited", "reconciliation"
}


def norm_label(label: str):
    """Lower-case words of a label, without leading enumerators or table header noise."""
    cl = _HEADER_NOISE_RE.sub("", label.strip())
    words = re.sub(r"[^a-z ]", " ", cl.lower().replace("-", " ")).split()
    while len(words) > 1 and (len(words[0]) == 1 or re.fullmatch(r"[ivxlc]{1,4}", words[0])):
        words.pop(0)
    return words


def match_row(label: str, cfg: dict):
    """Return the alias TIER (0 = strongest) if the label is one of the metric's names, else None."""
    words = norm_label(label)
    if not words or any(w in cfg["exclude_words"] for w in words):
        return None
    for tier, aliases in enumerate(cfg["aliases"]):
        for alias in aliases:
            a = alias.split()
            alen = len(a)
            for idx in range(len(words) - alen + 1):
                if words[idx: idx + alen] == a:
                    left_noise = idx
                    if left_noise > 8:
                        continue
                    right_words = words[idx + alen:]
                    if not right_words or all(rw in _NOTE_TAIL_WORDS or len(rw) <= 2 for rw in right_words):
                        return tier
    return None


def header_years(text: str, start: int, n_cols: int):
    """Years in the nearest lines before `start`, in order, without repeats."""
    lines = text[:start].splitlines()
    found = []
    for line in reversed(lines[-10:]):
        matches = []
        for m in YEAR_RE.finditer(line):
            y = int(m.group(1)[:2] + m.group(2)) if m.group(1) else int(m.group(3))
            matches.append(y)
        if len(matches) >= n_cols:
            return matches[-n_cols:]
        for m in reversed(matches):
            if m not in found:
                found.insert(0, m)
        if len(found) >= n_cols:
            return found[-n_cols:]
    if found:
        return found[-n_cols:]
    window = text[max(0, start - 500): start]
    years = []
    for m in YEAR_RE.finditer(window):
        y = int(m.group(1)[:2] + m.group(2)) if m.group(1) else int(m.group(3))
        if y not in years:
            years.append(y)
    return years[-n_cols:]


def detect_unit(text: str, start: int):
    """Unit as printed (million / crore ...). Unit words next to 'share' are ignored, because
    '(In millions, except number of shares, which are reflected in thousands)' has two units."""
    before = text[max(0, start - 1500): start]
    for scope_text in (before, text):
        hits = []
        for m in UNIT_RE.finditer(scope_text):
            if "share" in scope_text[max(0, m.start() - 60): m.start()].lower():
                continue
            hits.append(m.group(1))
        if hits:
            return UNIT_NORMAL.get(hits[-1].lower().rstrip("."))
    return None


def detect_currency(text: str, unit):
    if unit in ("crore", "lakh"):
        return "INR"
    counts = [(len(re.findall(r"\$|US\$|\bUSD\b", text)), "USD"),
              (len(re.findall(r"\u20b9|\bRs\.?\b|\bINR\b", text)), "INR"),
              (len(re.findall(r"\u00a3|\bGBP\b", text)), "GBP"),
              (len(re.findall(r"\u20ac|\bEUR\b", text)), "EUR")]
    best = max(counts)
    return best[1] if best[0] > 0 else None


def detect_scope(text: str, start: int):
    before = text[max(0, start - 1500): start].lower()
    c, s = before.rfind("consolidated"), before.rfind("standalone")
    if c == -1 and s == -1:
        return "unknown"
    return "consolidated" if c > s else "standalone"


def build_page_texts(chunks):
    """page -> full text (text and table chunks in order), page -> section type."""
    parts, tables, section = defaultdict(list), defaultdict(list), {}
    for c in chunks:
        curr_sec = section.get(c["page"], "other")
        if c["section_type"] in STATEMENT_SECTIONS:
            section[c["page"]] = c["section_type"]
        elif curr_sec == "other":
            section[c["page"]] = c["section_type"]

        if c["chunk_type"] == "text":
            parts[c["page"]].append((int(c["chunk_id"].rsplit("_c", 1)[1]), c["text"]))
        elif c["chunk_type"] == "table":
            tables[c["page"]].append(c["text"])

    page_texts = {}
    for p in set(parts.keys()).union(tables.keys()):
        t_text = "\n".join(t for _, t in sorted(parts.get(p, [])))
        tb_text = "\n\n".join(tables.get(p, []))
        if tb_text and tb_text not in t_text:
            page_texts[p] = (t_text + "\n\n" + tb_text).strip()
        else:
            page_texts[p] = t_text
    return page_texts, section


def is_valid_statement_head(text: str, start: int) -> bool:
    """Ignore statement head phrases that appear inside row descriptions (e.g. 'reclassified to statement of Profit & Loss')."""
    prefix = text[max(0, start - 45): start].lower()
    return not bool(re.search(r"\b(reclassified\s+to|transferred\s+to|charged\s+to|carried\s+to|recogni[sz]ed\s+in)\b", prefix))


# ------------------------------------------------------------------ steps 2-4
def statement_rows(text: str, cfg: dict):
    """Rows of the statement on this page: from the statement title down (to the first
    expense row when the metric lives in the income region)."""
    rows = parse_rows(text)
    anchors = [m.start() for m in STATEMENT_HEAD.finditer(text) if is_valid_statement_head(text, m.start())] + [0]
    for anchor in anchors:
        sub = [r for r in rows if r.start >= anchor]
        if cfg["region"] == "income":
            cut = []
            for r in sub:
                if STOP.match(r.label):
                    break
                cut.append(r)
            sub = cut
        if any(match_row(r.label, cfg) is not None for r in sub):
            return sub
    return []


def pick_column(row, years, target_year):
    """Which number in the row belongs to the requested year? -> (token, year, status)"""
    n = len(row.tokens)
    if years and len(years) == n:
        year = max(years) if target_year is None else target_year
        if year in years:
            return row.tokens[years.index(year)], year, "ok"
        return None, None, "year_not_in_row"
    if target_year is None:
        return row.tokens[0], None, "assumed_first_column"
    return None, None, "year_unknown"


def statement_title(text: str, row_start: int, scope: str):
    """e.g. 'Consolidated Statement of Profit and Loss' (the title as printed in the report)."""
    heads = [m for m in STATEMENT_HEAD.finditer(text) if m.start() <= row_start and is_valid_statement_head(text, m.start())]
    if not heads:
        return None
    segment = text[heads[-1].start(): heads[-1].start() + 70]
    title = re.split(r"[\n(\d$\u20b9]|\s{2,}|\s+for\s+the\s+(?:year|period)", segment)[0].strip()
    title = " ".join(title.title().split()) if title.isupper() else " ".join(title.split())
    title = title[0].upper() + title[1:]
    return (scope.capitalize() + " " if scope in ("consolidated", "standalone") else "") + title


def evidence_lines(rows, idx):
    lo, hi = max(0, idx - 2), min(len(rows), idx + 3)
    return [("» " if j == idx else "  ") + rows[j].label + " | " + " | ".join(rows[j].tokens)
            for j in range(lo, hi)]


# ------------------------------------------------------------------ steps 1, 3-5
def find_candidates(texts, sections, parse_pages, cfg, year, scope_pref, page_rank, company: str = None):
    cands = []
    for page in parse_pages:
        text = texts.get(page, "")
        rows = statement_rows(text, cfg)
        if not rows:
            continue
        find_proofs(rows)
        for i, row in enumerate(rows):
            tier = match_row(row.label, cfg)
            if tier is None:
                continue
            anchor = rows[0].start if cfg["pages"] == "statement" else row.start
            years = header_years(text, anchor, len(row.tokens))
            token, year_used, ystatus = pick_column(row, years, year)
            scope = detect_scope(text, anchor)
            score = 50 - 10 * tier
            score += {"income_statement": 20, "highlights": 10}.get(sections.get(page), 0)
            if scope_pref is None:
                score += 4
            else:
                score += 8 if scope == scope_pref else 4 if scope == "unknown" else 0
            score += 5 if row.is_total else 0
            score += 5 if ystatus == "ok" else 0

            # Disambiguate Parent Company vs Subsidiary / Segment / MD&A disclosure
            head_window = text[max(0, anchor - 600): anchor + 400].lower()
            is_sub = bool(re.search(
                r"\b(retail ventures|platforms limited|subsidiary|subsidiaries|segment reporting|segment information|disaggregated revenue|directors'\s+report|board's\s+report|management\s+discussion|discussion\s+and\s+analysis)\b",
                head_window
            ))
            if is_sub and scope_pref == "consolidated":
                score -= 15

            # Parent company alignment
            if company and company.lower() in head_window and not is_sub:
                score += 5

            # Verified audited statement title check
            if any(is_valid_statement_head(text, m.start()) for m in STATEMENT_HEAD.finditer(head_window)):
                score += 5

            if token is None:
                score -= 100
            cands.append({"page": page, "text": text, "rows": rows, "idx": i, "row": row,
                          "tier": tier, "years": years, "token": token, "year_used": year_used,
                          "ystatus": ystatus, "scope": scope, "score": score,
                          "anchor": anchor, "rank": page_rank.get(page, 99)})
    cands.sort(
        key=lambda c: (c["score"], -c["rank"], -c["page"], -c["idx"]),
        reverse=True
    )
    return cands


def find_note_value(texts, note_cfg, scope, statement_value, target_year):
    """A note row such as 'Revenue recognised' (contracted price - reductions = revenue
    recognised) that is a verified total and within `tolerance` of the statement value."""
    regex = re.compile(note_cfg["regex"], re.I)
    best = None
    for page, text in texts.items():
        if not regex.search(text):                 # cheap pre-filter
            continue
        rows = parse_rows(text)
        find_proofs(rows)
        for row in rows:
            if not (regex.fullmatch(" ".join(norm_label(row.label))) and row.is_total):
                continue
            if detect_scope(text, row.start) not in (scope, "unknown"):
                continue
            years = header_years(text, row.start, len(row.tokens))
            token, _, status = pick_column(row, years, target_year)
            if token is None:
                continue
            rel = abs(to_float(token) - statement_value) / max(abs(statement_value), 1)
            if rel <= note_cfg["tolerance"] and (best is None or rel < best[0]):
                best = (rel, page, text, row, token)
    return best


# ------------------------------------------------------------------ validation helpers
def grade(points: int) -> str:
    return "HIGH" if points >= 8 else "MEDIUM" if points >= 5 else "LOW"


def cap(result: dict, level: str, why: str):
    """Confidence can be at most `level`; the reason is kept for the audit trail."""
    result["warnings"].append(why)
    if LEVELS.index(result["confidence"]) > LEVELS.index(level):
        result["confidence"] = level
        result["reasons"].append(f"cap {level}: {why}")


def check_identity(metric: str, best: dict, text: str):
    """Cross-line arithmetic on the same statement, e.g. PBT = Total income - Total expenses.
    -> (None, "") if not applicable, else ("verified" | "not_reconciled", description)."""
    years, n, year = best["years"], len(best["row"].tokens), best["year_used"]
    if not (year and len(years) == n and year in years):
        return None, ""
    idx = years.index(year)
    rows = [r for r in parse_rows(text) if r.start >= best["rows"][0].start]
    for ident in IDENTITIES:
        if ident["result"] != metric:
            continue
        total, found = 0.0, True
        for term_metric, sign in ident["terms"]:
            hit = next((r for r in rows if match_row(r.label, METRICS[term_metric]) is not None
                        and len(r.values) == n), None)
            if hit is None:
                found = False
                break
            total += sign * hit.values[idx]
        if not found:
            continue
        reported = best["row"].values[idx]
        if abs(total - reported) <= max(1.01, 1e-6 * abs(reported)):
            return "verified", ident["name"]
        return "not_reconciled", f"{ident['name']}: computed {total:,.2f} but reported {reported:,.2f}"
    return None, ""


def plausibility(cfg: dict, best: dict):
    """Business sanity rules. Each problem returned makes the answer VERIFY."""
    years, row, year = best["years"], best["row"], best["year_used"]
    if not (year and len(years) == len(row.tokens) and year in years):
        return []
    values = dict(zip(years, row.values))
    v, problems = values[year], []
    if cfg.get("positive_only") and v <= 0:
        problems.append(f"value {row.tokens[years.index(year)]} is not positive; this metric should be")
    prev = values.get(year - 1)
    if not cfg.get("volatile") and prev and v > 0 and prev > 0 and not (1 / YOY_LIMIT <= v / prev <= YOY_LIMIT):
        problems.append(f"year-on-year change of {v / prev:.1f}x looks implausible; check the unit or the column")
    return problems


# ------------------------------------------------------------------ the one public function
def extract_financial_metric(company: str, report: str, metric: str, year: int = None,
                             scope: str = "consolidated", _fallback_from: str = None):
    """company/report = names as stored by ingestion (e.g. 'TCS', 'FY26').
    metric = a key of METRICS. year = end year of the financial year (2026 for FY 2025-26);
    None means the latest year in the statement. scope = 'consolidated' | 'standalone' | None.
    Raises ReportNotFound if the report has not been ingested."""
    cfg = METRICS[metric]
    result = {"company": company, "report": report, "metric": metric, "metric_used": None,
              "status": "NOT_FOUND", "value": None, "value_float": None, "value_absolute": None,
              "unit": None, "currency": None, "year": None, "year_columns": [], "statement": None,
              "scope": None, "page": None, "label": None, "evidence": [], "evidence_row": None,
              "confidence": "NONE", "score": 0, "reasons": [], "warnings": [], "alternatives": [],
              "needs_verification": False, "interpretation": None, "message": ""}

    chunks = load_chunks(company, report)
    if not chunks:
        raise ReportNotFound(f"No report ingested for {company} {report}. See: python -m src.tools reports")
    texts, sections = build_page_texts(chunks)
    retrieved = []
    for _, c in retrieve(company, report, cfg):
        if c["page"] not in retrieved:
            retrieved.append(c["page"])
    statement_pages = [p for p in sorted(sections) if sections[p] in STATEMENT_SECTIONS]
    parse_pages = statement_pages if cfg["pages"] == "statement" \
        else list(dict.fromkeys(retrieved + statement_pages))
    log.debug("%s %s %s: search found pages %s; reading pages %s", company, report, metric, retrieved, parse_pages)

    cands = find_candidates(texts, sections, parse_pages, cfg, year, scope,
                            {p: i for i, p in enumerate(retrieved)}, company=company)
    for c in cands[:5]:
        log.debug("  candidate page %s '%s' %s score %s tier %s scope %s",
                  c["page"], c["row"].label, c["token"], c["score"], c["tier"], c["scope"])

    # -- no row is labelled like this metric: try the sector fallback (e.g. revenue -> total income)
    if not cands:
        fb = cfg.get("fallback")
        if fb and _fallback_from is None:
            r = extract_financial_metric(company, report, fb["metric"], year, scope, _fallback_from=metric)
            if r["status"] in ("OK", "VERIFY"):
                r["metric"], r["metric_used"] = metric, fb["metric"]
                cap(r, "MEDIUM", f"{metric.replace('_', ' ')} is not reported ({fb['reason']}); "
                                 f"{fb['metric'].replace('_', ' ')} is used instead")
                return r
        result["message"] = "no row in the report is labelled like this metric"
        return result
    best = cands[0]
    if best["token"] is None:
        result["status"] = "YEAR_NOT_IN_REPORT"
        result["year_columns"] = best["years"]
        result["message"] = f"the metric exists, but not for year {year}; columns present: {best['years']}"
        return result

    text, row, anchor = best["text"], best["row"], best["anchor"]
    token, page = best["token"], best["page"]
    points, reasons, caps = 0, [], []

    def add(p, why):
        nonlocal points
        points += p
        reasons.append(f"{p:+d} {why}")

    # -- year
    if best["year_used"]:
        add(1, f"year headings {best['years']} -> column of {best['year_used']}")
    else:
        caps.append("year headings not found: the first column was assumed")
    digits = re.sub(r"\D", "", report)
    if best["year_used"] and digits and str(best["year_used"])[-2:] == digits[-2:]:
        add(1, f"year {best['year_used']} is the report's own year ({report})")
    elif best["year_used"]:
        reasons.append(f"+0 year {best['year_used']} comes from a comparative column of report {report}")

    # -- unit / currency / scope / statement page
    unit = detect_unit(text, anchor)
    currency = detect_currency(text, unit)
    add(1 if unit else 0, f"unit read from the report: {unit}" if unit else "unit not found")
    add(1 if currency else 0, f"currency: {currency}" if currency else "currency not found")
    if not unit:
        caps.append("unit not found in the report")
    if best["scope"] == "consolidated":
        add(1, "consolidated statement")
    else:
        reasons.append(f"+0 scope: {best['scope']}")
    if scope and best["scope"] not in (scope, "unknown"):
        caps.append(f"{scope} statement requested but only the {best['scope']} one has this line")
    section = sections.get(page)
    add({"income_statement": 2, "highlights": 1}.get(section, 0), f"page {page} is tagged '{section}'")

    # -- arithmetic proof of a total, and cross-line identity
    if row.is_total:
        parts = " ".join(("+ " if s_ > 0 else "- ") + best["rows"][k].label for k, s_ in row.proof).lstrip("+ ")
        add(2, f"arithmetic proof: {row.label} = {parts} (verified on every year column)")
    else:
        add(1, "single line item (not a sum of other rows)")
    status_id, ident = check_identity(metric, best, text)
    if status_id == "verified":
        add(2, f"identity verified: {ident}")
    elif status_id == "not_reconciled":
        add(-1, f"identity not reconciled: {ident}")
        reasons[-1] += " (exceptional items or associates may explain the gap)"

    # -- cross-reference: same number on other retrieved pages?
    flat = token.replace(",", "").strip("()-")
    pat = re.compile(rf"(?<![\d.]){re.escape(flat)}(?!\d)")
    confirm = [p for p in retrieved if p != page and pat.search(texts.get(p, "").replace(",", ""))]
    add(2 if confirm else 0, f"same number also on page(s) {confirm}" if confirm
        else "number not found on other retrieved pages")

    # -- business-rule plausibility (checked on the statement value, before any note override)
    problems = plausibility(cfg, best)

    # -- optional note override (e.g. TCS "Revenue recognised")
    label = row.label
    if cfg.get("note_override"):
        hit = find_note_value(texts, cfg["note_override"], best["scope"], to_float(token), best["year_used"])
        if hit:
            _, npage, ntext, nrow, ntoken = hit
            if ntoken != token:
                reasons.append(f"+0 statement (page {page}) shows {token}; the note differs by "
                               f"{abs(to_float(ntoken) - to_float(token)):g} - the note value is used")
            reasons.append(f"+0 note on page {npage}: '{nrow.label}' is a verified total ({ntoken})")
            token, page, label = ntoken, npage, nrow.label + " (reconciliation note)"
            unit = detect_unit(ntext, nrow.start) or unit
            note_rows = parse_rows(ntext)
            find_proofs(note_rows)
            best = dict(best, rows=note_rows)
            best["idx"] = next((i for i, r in enumerate(best["rows"]) if r.start == nrow.start), 0)
            row = nrow

    # -- rivals: a different value with (almost) the same score, which is not a part of the best row
    parts_of_best = {k for k, _ in row.proof} if best["page"] == page else set()
    rivals = [c for c in cands[1:]
              if c["token"] is not None
              and abs(to_float(c["token"]) - to_float(token)) > 0.005 * max(abs(to_float(token)), 1)
              and c["score"] >= cands[0]["score"] - AMBIGUITY_GAP
              and not (c["page"] == best["page"] and c["idx"] in parts_of_best)]
    seen, alts = set(), []
    for c in cands[1:]:
        key = (c["row"].label, c["token"], c["page"])
        if c["token"] is not None and key not in seen:
            seen.add(key)
            alts.append({"label": c["row"].label, "value": c["token"], "page": c["page"],
                         "scope": c["scope"], "score": c["score"]})
    if rivals:
        r = rivals[0]
        problems.insert(0, f"'{r['row'].label}' = {r['token']} (page {r['page']}) is almost as plausible as "
                           f"'{label}' = {token}")

    value_float = to_float(token)
    result.update({
        "status": "VERIFY" if problems else "OK",
        "value": token, "value_float": value_float, "unit": unit, "currency": currency,
        "value_absolute": value_float * UNIT_FACTOR[unit] if unit in UNIT_FACTOR else None,
        "year": best["year_used"], "year_columns": best["years"],
        "statement": statement_title(text, anchor, best["scope"]) if page == best["page"]
        else "Notes to the financial statements",
        "scope": best["scope"], "page": page, "label": label,
        "evidence": evidence_lines(best["rows"], best["idx"]),
        "evidence_row": f"{label} {token}", "score": points, "reasons": reasons,
        "confidence": grade(points), "warnings": [], "alternatives": alts[:5],
        "needs_verification": bool(problems),
    })
    for why in caps:
        cap(result, "MEDIUM", why)
    if problems:
        result["confidence"] = "LOW"
        result["message"] = "verification required: " + "; ".join(problems)
        result["warnings"].extend(problems)
    if cfg.get("negative_is"):
        word = cfg["negative_is"]
        result["interpretation"] = (f"{word} of {abs(value_float):,.2f}".rstrip("0").rstrip(".")
                                    if value_float < 0
                                    else f"no {word}: the reported value is positive ({token})")
    log.info("%s %s %s -> %s %s (page %s, %s, %s)", company, report, metric, token, unit,
             page, result["status"], result["confidence"])
    return result
