"""Deterministic cross-checks for BRD items 1-4 (bước B6 "Kiểm tra cụ thể").

Everything the BRD asks to *compare, count or date* happens here, in code:
"MST có khớp tuyệt đối", "thay đổi trong 12 tháng gần nhất", "tỷ lệ sở hữu có
đủ", "người đại diện thay đổi gần đây". A model asked to do these in a prompt
gets them right most of the time, and nothing downstream can tell the times it
did not — the same reason the retail calculators keep DSR out of prompts.

Input is a list of *records*, one per (document, pass), each tagged with its
source:

    {"source": "customer" | "field" | "external" | "internal",
     "document": "GCN_DKDN.pdf", "document_type": "gcn_dkdn",
     "pass": "legal_basic", "data": {...pass JSON...}}

Output per item: a ``summary`` (the BRD "template" view, each value carrying
the source it came from), ``findings`` (severity + message + evidence) and a
one-line ``conclusion``.

Severities: ``alert`` (cảnh báo — blocks or needs escalation), ``warning``
(cần làm rõ), ``info`` (ghi nhận), ``ok`` (đã kiểm tra, khớp).
"""

from __future__ import annotations

import calendar
import re
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Iterable

from src.utils.common import normalize_text

SEVERITY_ORDER = {"alert": 3, "warning": 2, "info": 1, "ok": 0}
LOOKBACK_DAYS = 365

# Which source wins when sources disagree on a value shown in the summary.
# The national register beats the customer's paper, which beats the field
# report, which beats whatever the bank's own system last stored.
SOURCE_PRIORITY = ("external", "customer", "field", "internal")
SOURCE_LABELS = {
    "external": "Cổng ĐKDN/cơ quan thuế",
    "customer": "KH cung cấp",
    "field": "Báo cáo thực địa",
    "internal": "Dữ liệu nội bộ TCB",
}

ITEM_TITLES = {
    1: "Thông tin cơ bản về pháp lý doanh nghiệp",
    2: "Cổ đông, thành viên góp vốn",
    3: "Người đại diện pháp luật",
    4: "Nợ thuế, BHXH, phạt hành chính, tranh chấp",
}

CHANGE_FIELD_LABELS = {
    "name": "tên DN",
    "capital": "vốn điều lệ",
    "address": "địa chỉ trụ sở",
    "legal_rep": "người đại diện",
    "members": "thành viên góp vốn/cổ đông",
    "business_lines": "ngành nghề",
    "other": "khác",
}

STATUS_LABELS = {
    "dang_hoat_dong": "Đang hoạt động",
    "tam_ngung": "Tạm ngừng hoạt động",
    "ngung_hoat_dong": "Ngừng hoạt động",
    "giai_the": "Giải thể",
    "pha_san": "Phá sản",
    "khac": "Khác",
}

LEGAL_FORM_LABELS = {
    "cong_ty_co_phan": "Công ty cổ phần",
    "tnhh_mot_thanh_vien": "Công ty TNHH một thành viên",
    "tnhh_hai_thanh_vien_tro_len": "Công ty TNHH hai thành viên trở lên",
    "doanh_nghiep_tu_nhan": "Doanh nghiệp tư nhân",
    "cong_ty_hop_danh": "Công ty hợp danh",
    "khac": "Khác",
}

_COMPANY_FORM_TOKENS = (
    "cong ty", "trach nhiem huu han", "tnhh", "co phan", "cp", "mot thanh vien",
    "hai thanh vien tro len", "mtv", "doanh nghiep tu nhan", "dntn", "hop danh",
    "jsc", "joint stock company", "company limited", "co ltd", "ltd", "company",
)
_PERSON_TITLE_TOKENS = ("ong", "ba", "mr", "mrs", "ms")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    item: int
    code: str
    severity: str
    message: str
    evidence: list[str] = field(default_factory=list)


def parse_date(value: Any, end_of_period: bool = False) -> date | None:
    """Read dd/mm/yyyy, yyyy-mm-dd, mm/yyyy or yyyy. None if unreadable.

    ``end_of_period`` resolves a month or year to its last day, so a change
    dated only "03/2026" still counts as inside a window that ends mid-March —
    the conservative reading for a "changed in the last 12 months" warning.
    """

    text = str(value or "").strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    try:
        match = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", text)
        if match:
            day, month, year = map(int, match.groups())
            return date(year, month, day)
        match = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
        if match:
            year, month, day = map(int, match.groups())
            return date(year, month, day)
        match = re.fullmatch(r"(\d{1,2})[/.\-](\d{4})", text)
        if match:
            month, year = map(int, match.groups())
            day = calendar.monthrange(year, month)[1] if end_of_period else 1
            return date(year, month, day)
        match = re.fullmatch(r"(\d{4})", text)
        if match:
            year = int(match.group(1))
            return date(year, 12, 31) if end_of_period else date(year, 1, 1)
    except ValueError:
        return None
    return None


def within_lookback(value: Any, as_of: date) -> bool:
    when = parse_date(value, end_of_period=True)
    return bool(when and as_of - timedelta(days=LOOKBACK_DAYS) <= when <= as_of)


def _strip_tokens(text: str, tokens: Iterable[str]) -> str:
    padded = f" {text} "
    for token in sorted(tokens, key=len, reverse=True):
        padded = padded.replace(f" {token} ", " ")
    return " ".join(padded.split())


def norm_company(name: Any) -> str:
    return _strip_tokens(normalize_text(str(name or "")), _COMPANY_FORM_TOKENS)


def norm_person(name: Any) -> str:
    return _strip_tokens(normalize_text(str(name or "")), _PERSON_TITLE_TOKENS)


def norm_tax_code(code: Any) -> str:
    return re.sub(r"[^0-9\-]", "", str(code or ""))


def norm_address(address: Any) -> str:
    return normalize_text(str(address or ""))


def _num(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def fmt_money(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "—"
    return f"{number:,.0f}".replace(",", ".") + " đ"


def _cite(record: dict[str, Any]) -> str:
    return f"{record.get('document', '?')} ({SOURCE_LABELS.get(record.get('source'), record.get('source'))})"


def _records(records: list[dict[str, Any]], pass_name: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("pass") == pass_name and isinstance(r.get("data"), dict)]


def _by_priority(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {source: i for i, source in enumerate(SOURCE_PRIORITY)}
    return sorted(records, key=lambda r: order.get(r.get("source"), 99))


def _pick(records: list[dict[str, Any]], key: str) -> tuple[Any, str]:
    """Highest-priority non-empty value of ``key`` and the record it came from."""

    for record in _by_priority(records):
        value = record["data"].get(key)
        if value not in (None, "", [], {}):
            return value, _cite(record)
    return None, ""


def _distinct(records: list[dict[str, Any]], key: str, normalizer) -> dict[str, list[str]]:
    """Normalised value → one evidence line: the value as printed, then every document stating it.

    Grouping by value keeps the evidence readable: "A — GCN; tra cứu" and
    "B — nội bộ" shows at a glance which source is the odd one out.
    """

    groups: dict[str, tuple[Any, list[str]]] = {}
    for record in records:
        raw = record["data"].get(key)
        if raw in (None, "", []):
            continue
        normalized = normalizer(raw)
        if normalized:
            groups.setdefault(normalized, (raw, []))[1].append(_cite(record))
    shown = lambda raw: fmt_money(raw) if key == "charter_capital" else raw
    return {norm: [f"{shown(raw)} — {'; '.join(cites)}"] for norm, (raw, cites) in groups.items()}


def _watchlist_hits(
    watchlist: list[dict[str, Any]],
    *,
    names: Iterable[str] = (),
    ids: Iterable[str] = (),
    company: bool = False,
) -> list[str]:
    norm = norm_company if company else norm_person
    wanted_names = {norm(n) for n in names if norm(n)}
    wanted_ids = {norm_tax_code(i) for i in ids if norm_tax_code(i)}
    hits = []
    for entry in watchlist or []:
        entry_name = norm(entry.get("name"))
        entry_ids = {norm_tax_code(entry.get("tax_code")), norm_tax_code(entry.get("id_number"))} - {""}
        if (entry_name and entry_name in wanted_names) or (entry_ids & wanted_ids):
            hit = f"{entry.get('name') or entry.get('tax_code')} — danh sách {entry.get('list', 'cảnh báo')}"
            if hit not in hits:
                hits.append(hit)
    return hits


def _rep_names(record: dict[str, Any]) -> list[str]:
    reps = record["data"].get("legal_representatives") or []
    names = []
    for rep in reps:
        name = rep.get("name") if isinstance(rep, dict) else rep
        if name:
            names.append(str(name))
    return names


def _rep_groups(records: list[dict[str, Any]]) -> dict[frozenset, list[str]]:
    """Set of representative names → one evidence line listing each document once."""

    groups: dict[frozenset, tuple[list[str], list[str]]] = {}
    for record in records:
        names = _rep_names(record)
        key = frozenset(norm_person(n) for n in names) - {""}
        if key:
            shown, cites = groups.setdefault(key, (names, []))
            if _cite(record) not in cites:
                cites.append(_cite(record))
    return {key: [f"{', '.join(shown)} — {'; '.join(cites)}"] for key, (shown, cites) in groups.items()}


def _conclude(findings: list[Finding], ok_text: str) -> str:
    worst = max((SEVERITY_ORDER[f.severity] for f in findings), default=0)
    if worst == 3:
        return "CẢNH BÁO — " + "; ".join(f.message for f in findings if f.severity == "alert")
    if worst == 2:
        return "Cần làm rõ — " + "; ".join(f.message for f in findings if f.severity == "warning")
    return ok_text


# --------------------------------------------------------------------------- #
# Item 1 — Thông tin cơ bản về pháp lý doanh nghiệp
# --------------------------------------------------------------------------- #
def check_legal_basic(records, as_of: date, watchlist) -> dict[str, Any]:
    recs = _records(records, "legal_basic")
    findings: list[Finding] = []
    add = lambda *a, **k: findings.append(Finding(1, *a, **k))

    if not recs:
        add("no_data", "alert", "Không trích xuất được thông tin pháp lý cơ bản từ hồ sơ nào")
        return {"summary": {}, "findings": findings, "conclusion": _conclude(findings, "")}

    # MST — must match exactly across every source.
    tax_codes = _distinct(recs, "tax_code", norm_tax_code)
    if not tax_codes:
        add("tax_code_missing", "warning", "Không đọc được MST/mã số doanh nghiệp trên hồ sơ")
    elif len(tax_codes) > 1:
        add("tax_code_mismatch", "alert", "MST không khớp tuyệt đối giữa các nguồn dữ liệu",
            [e for v in tax_codes.values() for e in v])
    else:
        add("tax_code_match", "ok", "MST khớp giữa các nguồn", next(iter(tax_codes.values())))

    names = _distinct(recs, "company_name", norm_company)
    if len(names) > 1:
        add("name_mismatch", "warning", "Tên doanh nghiệp không khớp giữa các nguồn",
            [e for v in names.values() for e in v])
    elif names:
        add("name_match", "ok", "Tên doanh nghiệp khớp giữa các nguồn", next(iter(names.values())))

    addresses = _distinct(recs, "head_office_address", norm_address)
    if len(addresses) > 1:
        add("address_mismatch", "warning", "Địa chỉ trụ sở chính ghi khác nhau giữa các nguồn",
            [e for v in addresses.values() for e in v])

    founded = _distinct(recs, "establishment_date", lambda v: str(parse_date(v) or v))
    if len(founded) > 1:
        add("establishment_mismatch", "warning", "Ngày thành lập không khớp giữa các nguồn",
            [e for v in founded.values() for e in v])

    capital = _distinct(recs, "charter_capital", lambda v: str(_num(v)) if _num(v) is not None else "")
    if len(capital) > 1:
        add("capital_mismatch", "warning", "Vốn điều lệ không khớp giữa các nguồn (có thể do tăng/giảm vốn)",
            [e for v in capital.values() for e in v])

    rep_sets = _rep_groups(recs)
    if len(rep_sets) > 1:
        add("rep_mismatch", "warning", "Người đại diện theo pháp luật không khớp giữa các nguồn",
            [e for v in rep_sets.values() for e in v])

    # Operating status — only the register says this; the GCN never does.
    statuses = [(r["data"].get("operating_status"), r) for r in _by_priority(recs)
                if r["data"].get("operating_status")]
    if not statuses:
        add("status_unknown", "warning",
            "Chưa có kết quả tra cứu trạng thái hoạt động (Cổng ĐKDN/cơ quan thuế/masothue.com)")
    else:
        bad = [(s, r) for s, r in statuses if s != "dang_hoat_dong"]
        if bad:
            add("status_not_active", "alert",
                "Trạng thái doanh nghiệp: " + ", ".join(sorted({STATUS_LABELS.get(s, s) for s, _ in bad})),
                [_cite(r) for _, r in bad])
        else:
            add("status_active", "ok", "Doanh nghiệp đang hoạt động theo thông tin thuế/ĐKDN",
                [_cite(r) for _, r in statuses])

    # Changes in the last 12 months.
    changes = []
    for record in recs:
        data = record["data"]
        for change in data.get("change_history") or []:
            if isinstance(change, dict) and within_lookback(change.get("date"), as_of):
                fields_changed = [CHANGE_FIELD_LABELS.get(f, f) for f in change.get("fields_changed") or []]
                what = ", ".join(fields_changed) or "chưa rõ nội dung"
                changes.append(f"{change.get('date')}: thay đổi {what} — {_cite(record)}")
        if within_lookback(data.get("latest_change_date"), as_of):
            changes.append(f"{data.get('latest_change_date')}: đăng ký thay đổi lần "
                           f"{data.get('registration_change_number') or '?'} — {_cite(record)}")
    if changes:
        add("changes_12m", "warning",
            "Doanh nghiệp có đăng ký thay đổi trong 12 tháng trước thời điểm vay — đối chiếu báo cáo thực địa, bocaodientu.com.vn",
            sorted(set(changes)))

    # Internal system up to date?
    internal = [r for r in recs if r.get("source") == "internal"]
    reference = [r for r in recs if r.get("source") in ("external", "customer")]
    if not internal:
        add("internal_absent", "info", "Không có dữ liệu nội bộ (KH chưa phải ETC TCB hoặc chưa kết nối BCDE/BEP)")
    elif reference:
        stale = []
        checks = (
            ("tax_code", norm_tax_code, "MST"),
            ("company_name", norm_company, "tên DN"),
            ("head_office_address", norm_address, "địa chỉ"),
            ("charter_capital", lambda v: str(_num(v)), "vốn điều lệ"),
        )
        for key, normalizer, label in checks:
            ref_value, ref_cite = _pick(reference, key)
            int_value, _ = _pick(internal, key)
            if ref_value not in (None, "") and int_value not in (None, "") and normalizer(ref_value) != normalizer(int_value):
                stale.append(f"{label}: nội bộ '{int_value}' ≠ '{ref_value}' ({ref_cite})")
        if stale:
            add("internal_stale", "warning", "Thông tin hệ thống nội bộ chưa cập nhật theo GCN ĐKDN/external data", stale)
        else:
            add("internal_current", "ok", "Thông tin hệ thống nội bộ khớp GCN ĐKDN/external data")

    # Site visit: does the company actually operate at the address?
    visits = [(r["data"].get("site_visit") or {}, r) for r in recs if r.get("source") == "field"]
    confirmed = [(v, r) for v, r in visits if isinstance(v, dict) and v.get("actual_operation_confirmed") is not None]
    if not confirmed:
        add("site_unconfirmed", "warning", "Chưa có xác nhận doanh nghiệp hoạt động thực tế tại địa chỉ (AI Site Visit/báo cáo thực địa)")
    elif any(v.get("actual_operation_confirmed") is False for v, _ in confirmed):
        add("site_not_operating", "alert", "Báo cáo thực địa không xác nhận doanh nghiệp hoạt động thực tế tại địa chỉ",
            [_cite(r) for v, r in confirmed if v.get("actual_operation_confirmed") is False])
    else:
        add("site_operating", "ok", "Báo cáo thực địa xác nhận doanh nghiệp hoạt động thực tế tại địa chỉ",
            [_cite(r) for _, r in confirmed])

    company_names = [r["data"].get("company_name") for r in recs]
    tax_ids = [r["data"].get("tax_code") for r in recs]
    hits = _watchlist_hits(watchlist, names=company_names, ids=tax_ids, company=True)
    if hits:
        add("watchlist_company", "alert", "Doanh nghiệp thuộc danh sách cảnh báo AML/BL/WL", hits)

    summary = {}
    for key, label in (
        ("company_name", "Tên doanh nghiệp"), ("company_name_foreign", "Tên nước ngoài"),
        ("abbreviation", "Tên viết tắt"), ("tax_code", "Mã số doanh nghiệp/MST"),
        ("legal_form", "Loại hình"), ("establishment_date", "Ngày thành lập"),
        ("registration_change_number", "Lần đăng ký thay đổi"), ("latest_change_date", "Ngày thay đổi gần nhất"),
        ("charter_capital", "Vốn điều lệ"), ("head_office_address", "Địa chỉ trụ sở chính"),
        ("legal_representatives", "Người đại diện theo pháp luật"), ("operating_status", "Trạng thái hoạt động"),
    ):
        value, cite = _pick(recs, key)
        if key == "charter_capital":
            value = fmt_money(value) if value is not None else None
        elif key == "operating_status" and value:
            value = STATUS_LABELS.get(value, value)
        elif key == "legal_form" and value:
            value = LEGAL_FORM_LABELS.get(value, value)
        elif isinstance(value, list):
            value = ", ".join(str(v.get("name") if isinstance(v, dict) else v) for v in value)
        summary[label] = {"value": value, "source": cite}

    return {
        "summary": summary,
        "findings": findings,
        "conclusion": _conclude(findings, "Pháp lý cơ bản phù hợp giữa các nguồn; doanh nghiệp đang hoạt động"),
    }


# --------------------------------------------------------------------------- #
# Item 2 — Cổ đông, thành viên góp vốn
# --------------------------------------------------------------------------- #
_MEMBER_DOC_PRIORITY = ("so_dang_ky_co_dong", "gcn_dkdn", "tra_cuu_dkdn", "dieu_le", "quyet_dinh_bo_nhiem", "chuyen_nhuong_von")


def _primary_member_record(recs):
    with_members = [r for r in recs if r["data"].get("members")]
    if not with_members:
        return None
    order = {t: i for i, t in enumerate(_MEMBER_DOC_PRIORITY)}
    return min(
        with_members,
        key=lambda r: (
            order.get(r.get("document_type"), 50),
            -(parse_date(r["data"].get("document_date")) or date.min).toordinal(),
        ),
    )


def check_shareholders(records, as_of: date, watchlist, borrower_names: list[str]) -> dict[str, Any]:
    recs = _records(records, "shareholders")
    findings: list[Finding] = []
    add = lambda *a, **k: findings.append(Finding(2, *a, **k))

    primary = _primary_member_record(recs)
    if primary is None:
        add("no_members", "alert", "Không trích xuất được danh sách cổ đông/thành viên góp vốn")
        return {"summary": {"members": []}, "findings": findings, "conclusion": _conclude(findings, "")}

    data = primary["data"]
    members = [m for m in data.get("members") or [] if isinstance(m, dict)]
    charter = _num(data.get("charter_capital"))
    if charter is None:
        lb_capital, _ = _pick(_records(records, "legal_basic"), "charter_capital")
        charter = _num(lb_capital)

    rows = []
    for member in members:
        pct = _num(member.get("ownership_pct"))
        pct_source = "như in"
        if pct is None and charter and _num(member.get("capital_amount")) is not None:
            pct = round(_num(member["capital_amount"]) / charter * 100, 2)
            pct_source = "tính toán"
        rows.append({
            "name": member.get("name", ""),
            "member_type": member.get("member_type", ""),
            "id_number": member.get("id_number", ""),
            "capital_amount": member.get("capital_amount"),
            "ownership_pct": pct,
            "pct_source": pct_source,
            "acquisition_date": member.get("acquisition_date", ""),
            "acquisition_method": member.get("acquisition_method", ""),
            "source": _cite(primary),
        })

    pcts = [r["ownership_pct"] for r in rows]
    if pcts and all(p is not None for p in pcts):
        total = round(sum(pcts), 2)
        if abs(total - 100) > 0.5:
            add("ownership_not_100", "warning", f"Tổng tỷ lệ sở hữu = {total}% (≠ 100%)", [_cite(primary)])
        else:
            add("ownership_100", "ok", f"Tổng tỷ lệ sở hữu = {total}%", [_cite(primary)])
    else:
        add("ownership_incomplete", "warning", "Thiếu tỷ lệ sở hữu/vốn góp của một số thành viên", [_cite(primary)])
    capitals = [_num(r["capital_amount"]) for r in rows]
    if charter and capitals and all(c is not None for c in capitals):
        if abs(sum(capitals) - charter) > 0.005 * charter:
            add("capital_sum_mismatch", "warning",
                f"Tổng vốn góp {fmt_money(sum(capitals))} ≠ vốn điều lệ {fmt_money(charter)}", [_cite(primary)])

    # Member lists across sources.
    member_sets: dict[frozenset, list[str]] = {}
    for record in recs:
        names = frozenset(norm_person(m.get("name")) for m in record["data"].get("members") or []
                          if isinstance(m, dict)) - {""}
        if names:
            member_sets.setdefault(names, []).append(_cite(record))
    if len(member_sets) > 1:
        primary_set = max(member_sets, key=len)
        evidence = []
        for names, cites in member_sets.items():
            missing = primary_set - names
            extra = names - primary_set
            if missing or extra:
                evidence.append(f"{', '.join(cites)}: thiếu {sorted(missing) or '—'}, thêm {sorted(extra) or '—'}")
        add("member_mismatch", "warning", "Danh sách thành viên/cổ đông không khớp giữa các nguồn", evidence)

    # Transfers / ownership changes in the last 12 months.
    recent = []
    for record in recs:
        for transfer in record["data"].get("transfers") or []:
            if isinstance(transfer, dict) and within_lookback(transfer.get("date"), as_of):
                recent.append(f"{transfer.get('date')}: {transfer.get('transferor') or '?'} → "
                              f"{transfer.get('transferee') or '?'} ({_cite(record)})")
        for member in record["data"].get("members") or []:
            if (isinstance(member, dict) and member.get("acquisition_method") == "nhan_chuyen_nhuong"
                    and within_lookback(member.get("acquisition_date"), as_of)):
                recent.append(f"{member.get('acquisition_date')}: {member.get('name')} nhận chuyển nhượng ({_cite(record)})")
    for record in _records(records, "legal_basic"):
        for change in record["data"].get("change_history") or []:
            if (isinstance(change, dict) and "members" in (change.get("fields_changed") or [])
                    and within_lookback(change.get("date"), as_of)):
                recent.append(f"{change.get('date')}: thay đổi thành viên góp vốn ({_cite(record)})")
    if recent:
        add("transfer_12m", "warning", "Có chuyển nhượng vốn/thay đổi thành viên trong 12 tháng gần thời điểm vay",
            sorted(set(recent)))

    # The individual borrower(s) of this loan.
    borrower_rows = []
    for borrower in borrower_names or []:
        key = norm_person(borrower)
        match = next((r for r in rows if norm_person(r["name"]) == key), None)
        if match is None:
            add("borrower_not_member", "warning",
                f"KH cá nhân vay vốn '{borrower}' không có tên trong danh sách thành viên/cổ đông", [_cite(primary)])
            continue
        borrower_rows.append(match)
        new_entry = within_lookback(match["acquisition_date"], as_of)
        acquired_recently = new_entry and match["acquisition_method"] in ("nhan_chuyen_nhuong", "tang_von", "thua_ke_tang_cho", "")
        if not acquired_recently:
            # A transfer row naming the borrower as transferee counts too.
            for record in recs:
                for transfer in record["data"].get("transfers") or []:
                    if (isinstance(transfer, dict) and norm_person(transfer.get("transferee")) == key
                            and within_lookback(transfer.get("date"), as_of)):
                        acquired_recently = True
                        match = {**match, "acquisition_date": transfer.get("date")}
        pct_text = f"{match['ownership_pct']}%" if match["ownership_pct"] is not None else "chưa rõ tỷ lệ"
        if acquired_recently:
            add("borrower_new_member", "alert",
                f"KH cá nhân vay vốn {match['name']} mới tiếp nhận phần vốn góp trong 12 tháng "
                f"({match['acquisition_date'] or 'ngày chưa rõ'}; sở hữu {pct_text})", [match["source"]])
        else:
            add("borrower_member", "info",
                f"KH cá nhân vay vốn {match['name']} sở hữu {pct_text}"
                f"{', từ ' + match['acquisition_date'] if match['acquisition_date'] else ''}", [match["source"]])

    corporate = [r["name"] for r in rows if r["member_type"] == "phap_nhan"]
    if corporate:
        add("corporate_members", "info",
            "Có cổ đông pháp nhân — cần làm rõ chủ sở hữu các tầng trên (minh bạch sở hữu nhiều tầng)", corporate)

    relations = []
    for record in recs:
        for rel in record["data"].get("relationships") or []:
            if isinstance(rel, dict):
                relations.append(f"{rel.get('member_a')} – {rel.get('member_b')}: {rel.get('relation')} ({_cite(record)})")
        for note in record["data"].get("beneficial_owner_notes") or []:
            relations.append(f"{note} ({_cite(record)})")
    if relations:
        add("relationships", "info", "Quan hệ giữa các chủ sở hữu / chủ sở hữu thực tế", relations)

    hits = _watchlist_hits(watchlist, names=[r["name"] for r in rows], ids=[r["id_number"] for r in rows])
    hits += [h for h in _watchlist_hits(watchlist, names=corporate, company=True) if h not in hits]
    if hits:
        add("watchlist_member", "alert", "Cổ đông/thành viên góp vốn thuộc danh sách cảnh báo AML/WL/BL", hits)

    return {
        "summary": {"charter_capital": charter, "members": rows, "borrowers": borrower_rows},
        "findings": findings,
        "conclusion": _conclude(findings, "Cơ cấu sở hữu đầy đủ, khớp giữa các nguồn, không có chuyển nhượng gần thời điểm vay"),
    }


# --------------------------------------------------------------------------- #
# Item 3 — Người đại diện pháp luật
# --------------------------------------------------------------------------- #
def check_legal_rep(records, as_of: date, watchlist) -> dict[str, Any]:
    recs = _records(records, "legal_rep")
    basics = _records(records, "legal_basic")
    findings: list[Finding] = []
    add = lambda *a, **k: findings.append(Finding(3, *a, **k))

    rep_sets = _rep_groups(_by_priority(recs + basics))
    current: dict[str, dict[str, Any]] = {}
    for record in _by_priority(recs + basics):
        for rep in record["data"].get("legal_representatives") or []:
            rep = rep if isinstance(rep, dict) else {"name": rep}
            key = norm_person(rep.get("name"))
            if key and key not in current:
                current[key] = {**rep, "source": _cite(record)}

    if not current:
        add("no_rep", "alert", "Không xác định được người đại diện theo pháp luật")
    elif len(rep_sets) > 1:
        add("rep_mismatch", "warning", "Người đại diện theo pháp luật không khớp giữa các nguồn",
            [e for v in rep_sets.values() for e in v])
    else:
        add("rep_match", "ok", "Người đại diện theo pháp luật khớp giữa các nguồn", next(iter(rep_sets.values())))

    changes = []
    for record in recs:
        for rep in record["data"].get("legal_representatives") or []:
            if isinstance(rep, dict) and within_lookback(rep.get("appointed_date"), as_of):
                changes.append(f"{rep.get('name')} được bổ nhiệm {rep.get('appointed_date')} ({_cite(record)})")
        for prev in record["data"].get("previous_representatives") or []:
            if isinstance(prev, dict) and within_lookback(prev.get("until_date"), as_of):
                changes.append(f"{prev.get('name')} thôi đại diện {prev.get('until_date')} ({_cite(record)})")
    for record in basics:
        for change in record["data"].get("change_history") or []:
            if (isinstance(change, dict) and "legal_rep" in (change.get("fields_changed") or [])
                    and within_lookback(change.get("date"), as_of)):
                changes.append(f"{change.get('date')}: thay đổi người đại diện ({_cite(record)})")
    if changes:
        add("rep_changed_12m", "alert", "Người đại diện theo pháp luật thay đổi trong 12 tháng gần nhất", sorted(set(changes)))

    operator, operator_cite = None, ""
    people_met = []
    for record in recs:
        candidate = record["data"].get("actual_operator") or {}
        if isinstance(candidate, dict) and candidate.get("name") and operator is None:
            operator, operator_cite = candidate, _cite(record)
        for person in record["data"].get("people_met") or []:
            if isinstance(person, dict) and person.get("name"):
                flags = []
                if person.get("decides_business"):
                    flags.append("quyết định kinh doanh")
                if person.get("understands_operations_finance"):
                    flags.append("hiểu hoạt động & tài chính")
                people_met.append(f"{person['name']} ({person.get('role') or '—'})"
                                  f"{': ' + ', '.join(flags) if flags else ''} — {_cite(record)}")
    if operator is None:
        add("operator_unknown", "info", "Chưa có thông tin người điều hành thực tế (AI Site Visit/RBO interview)")
    elif norm_person(operator["name"]) not in current:
        add("operator_differs", "warning",
            f"Người điều hành thực tế ({operator['name']}) khác người đại diện theo pháp luật",
            [f"{operator_cite}: {operator.get('evidence') or operator.get('role') or ''}".strip()])
    else:
        add("operator_is_rep", "ok", f"Người điều hành thực tế là người đại diện theo pháp luật ({operator['name']})",
            [operator_cite])
    if people_met:
        add("people_met", "info", "Người tiếp xúc tại thực địa", people_met)

    hits = _watchlist_hits(watchlist, names=[r.get("name") for r in current.values()],
                           ids=[r.get("id_number") for r in current.values()])
    if operator:
        hits += _watchlist_hits(watchlist, names=[operator.get("name")])
    if hits:
        add("watchlist_rep", "alert", "Người đại diện/điều hành thuộc danh sách cảnh báo AML/WL/BL", hits)

    return {
        "summary": {
            "legal_representatives": list(current.values()),
            "actual_operator": {**operator, "source": operator_cite} if operator else None,
        },
        "findings": findings,
        "conclusion": _conclude(findings, "Người đại diện khớp đúng, ổn định; trùng người điều hành thực tế"),
    }


# --------------------------------------------------------------------------- #
# Item 4 — Nợ thuế, BHXH, phạt hành chính, tranh chấp
# --------------------------------------------------------------------------- #
_PAID = {"da_nop"}


def _open_debts(rows):
    return [r for r in rows if isinstance(r, dict) and r.get("status") not in _PAID
            and ((_num(r.get("amount")) or 0) > 0 or r.get("status") in ("con_no", "dang_cuong_che"))]


def check_liabilities(records, as_of: date) -> dict[str, Any]:
    recs = _records(records, "liabilities")
    findings: list[Finding] = []
    add = lambda *a, **k: findings.append(Finding(4, *a, **k))
    summary: dict[str, list[dict[str, Any]]] = {
        "tax_debts": [], "social_insurance_debts": [], "clearance_confirmations": [],
        "administrative_penalties": [], "enforcement_actions": [], "inspections": [],
        "disputes": [], "site_risks": [],
    }
    for record in recs:
        for key in summary:
            for row in record["data"].get(key) or []:
                entry = dict(row) if isinstance(row, dict) else {"note": row}
                entry["source"] = _cite(record)
                summary[key].append(entry)

    if not recs:
        add("no_documents", "warning",
            "Chưa có hồ sơ/tra cứu về nợ thuế, BHXH, xử phạt, tranh chấp — cần tra cứu cổng thuế, BHXH tỉnh thành")
        return {"summary": summary, "findings": findings, "conclusion": _conclude(findings, "")}

    for key, label, scope in (("tax_debts", "nợ thuế", "thue"), ("social_insurance_debts", "nợ BHXH", "bhxh")):
        debts = _open_debts(summary[key])
        cleared = [c for c in summary["clearance_confirmations"] if c.get("scope") == scope]
        if debts:
            total = sum(_num(d.get("amount")) or 0 for d in debts)
            long_overdue = [d for d in debts if (_num(d.get("overdue_days")) or 0) > 90
                            or (_num(d.get("months_overdue")) or 0) > 3]
            message = f"Doanh nghiệp còn {label}: {fmt_money(total)}"
            if long_overdue:
                message += f" ({len(long_overdue)} khoản quá hạn > 90 ngày)"
            add(f"{scope}_debt", "alert", message,
                [f"{fmt_money(d.get('amount'))} — {d.get('as_of') or ''} — {d['source']}" for d in debts])
            if cleared:
                add(f"{scope}_conflict", "alert", f"Mâu thuẫn: có xác nhận hoàn thành nghĩa vụ nhưng nguồn khác ghi {label}",
                    [c["source"] for c in cleared])
        elif cleared:
            add(f"{scope}_clear", "ok", f"Có xác nhận không còn {label}",
                [f"{c.get('as_of') or ''} — {c['source']}" for c in cleared])
        else:
            add(f"{scope}_unverified", "info", f"Chưa có xác nhận/tra cứu {label}")

    for penalty in summary["administrative_penalties"]:
        status = penalty.get("remediation_status") or ""
        desc = (f"{penalty.get('decision_no') or 'QĐ ?'} ngày {penalty.get('date') or '?'} — "
                f"{penalty.get('domain') or ''}: {penalty.get('violation') or ''} ({fmt_money(penalty.get('amount'))})")
        if status == "da_khac_phuc":
            add("penalty_remediated", "info", "Vi phạm đã khắc phục: " + desc, [penalty["source"]])
        else:
            add("penalty_open", "warning",
                f"Vi phạm hành chính {'chưa rõ tình trạng khắc phục' if not status else status.replace('_', ' ')}: {desc}",
                [penalty["source"]])

    if summary["enforcement_actions"]:
        add("enforcement", "alert", "Doanh nghiệp có quyết định cưỡng chế",
            [f"{e.get('decision_no') or ''} {e.get('date') or ''}: {e.get('measure') or ''} — {e['source']}"
             for e in summary["enforcement_actions"]])

    for dispute in summary["disputes"]:
        desc = (f"{dispute.get('case_no') or 'vụ việc'} tại {dispute.get('forum') or '?'} — vai trò "
                f"{dispute.get('role') or '?'}, với {dispute.get('counterparty') or '?'}: {dispute.get('subject') or ''} "
                f"({fmt_money(dispute.get('claim_amount'))}; {dispute.get('status') or 'chưa rõ tình trạng'})")
        severity = "alert" if dispute.get("affects_operations_or_collateral") else "warning"
        add("dispute", severity,
            "Tranh chấp ảnh hưởng hoạt động/TSBĐ/dòng tiền: " + desc if severity == "alert" else "Tranh chấp: " + desc,
            [dispute["source"]])

    if summary["site_risks"]:
        add("site_risk", "warning", "Rủi ro môi trường/PCCC ghi nhận tại địa điểm hoạt động",
            [f"{r.get('note')} — {r['source']}" for r in summary["site_risks"]])
    for inspection in summary["inspections"]:
        if inspection.get("findings"):
            add("inspection", "info",
                f"Kiểm tra/thanh tra {inspection.get('date') or ''} ({inspection.get('agency') or ''}): {inspection['findings']}",
                [inspection["source"]])

    return {
        "summary": summary,
        "findings": findings,
        "conclusion": _conclude(findings, "Không ghi nhận nợ thuế, nợ BHXH, xử phạt chưa khắc phục hay tranh chấp"),
    }


# --------------------------------------------------------------------------- #
def run_all_checks(
    records: list[dict[str, Any]],
    as_of: date | None = None,
    borrower_names: list[str] | None = None,
    watchlist: list[dict[str, Any]] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run items 1-4. ``as_of`` is the loan date the 12-month windows end at."""

    as_of = as_of or date.today()
    results = {
        1: check_legal_basic(records, as_of, watchlist or []),
        2: check_shareholders(records, as_of, watchlist or [], borrower_names or []),
        3: check_legal_rep(records, as_of, watchlist or []),
        4: check_liabilities(records, as_of),
    }
    for item, result in results.items():
        result["title"] = ITEM_TITLES[item]
        result["findings"] = sorted(
            (asdict(f) for f in result["findings"]),
            key=lambda f: -SEVERITY_ORDER[f["severity"]],
        )
    return results
