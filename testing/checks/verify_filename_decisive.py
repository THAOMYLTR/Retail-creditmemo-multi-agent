"""Verify retail documents classify correctly, and filename_decisive never relabels.

Three properties, each tied to how the rule pass is used:

1. Realistic retail filenames (CCCD photo, payslips, bank statement, house
   purchase contract, CIC report...) land on the intended matrix type. A miss
   here is a missing or colliding keyword in document_matrix.yaml.
2. Realistic document bodies with an uninformative filename ("scan001.pdf") still
   land on the right type — the keyword pass is what saves the LLM call.
3. The filename_decisive shortcut may only change *whether* the LLM is called,
   never the label the rule returns, and across every pair of types it never
   skips the LLM while contradicting the filename.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.agents.document_classification import (
    document_type_scores,
    filename_keyword_owner,
    rule_classify_document,
)
from src.config import Config
from src.matrix.document_matrix import load_matrix

TH = Config().document_classifier_rule_confidence_threshold
matrix = load_matrix()
tids = list(matrix.types)
failures = []


def calls_llm(rule, *, with_new_rule=True):
    if not rule["document_type"]:
        return True
    if rule["confidence"] >= TH or rule["routing_unambiguous"]:
        return False
    return not (with_new_rule and rule["filename_decisive"])


# --- 1. realistic retail filenames -------------------------------------------
FILENAMES = {
    "CCCD_Nguyen_Van_A.jpg": "giay_to_tuy_than",
    "Xac nhan cu tru CT07.pdf": "xac_nhan_cu_tru",
    "Giay dang ky ket hon.pdf": "tinh_trang_hon_nhan",
    "HDLD_cong_ty_ABC.pdf": "hop_dong_lao_dong",
    "Xac nhan luong thang 6.pdf": "xac_nhan_luong",
    "Bang luong 6 thang.xlsx": "xac_nhan_luong",
    "Sao ke tai khoan VCB 01-06.2026.pdf": "sao_ke_tai_khoan",
    "Sao ke luong.pdf": "sao_ke_tai_khoan",
    "Quyet toan thue TNCN 2025.pdf": "thue_thu_nhap_ca_nhan",
    "Hop dong cho thue can ho.pdf": "hop_dong_cho_thue",
    "Giay chung nhan dang ky ho kinh doanh.pdf": "ho_so_ho_kinh_doanh",
    "So tiet kiem.jpg": "thu_nhap_tai_san_khac",
    "Giay de nghi vay von.pdf": "de_nghi_vay_von",
    "Hop dong mua ban can ho.pdf": "hop_dong_mua_ban_nha_dat",
    "Hop dong dat coc.pdf": "hop_dong_mua_ban_nha_dat",
    "Hop dong mua ban xe o to.pdf": "hop_dong_mua_ban_xe",
    "So do thua dat.jpg": "giay_chung_nhan_qsdd",
    "Dang ky xe.jpg": "dang_ky_xe",
    "Chung thu tham dinh gia.pdf": "chung_thu_dinh_gia",
    "Hop dong the chap.pdf": "tai_san_bao_dam_khac",
    "CIC_KHCN.pdf": "cic_khach_hang_vay",
    "CIC vo chong.pdf": "cic_nguoi_dong_vay",
    "Bao cao tham dinh KHCN.pdf": "bao_cao_tham_dinh",
}
print("1. Tên file thực tế của hồ sơ bán lẻ")
for filename, expected in FILENAMES.items():
    rule = rule_classify_document(filename, "")
    ok = rule["document_type"] == expected
    print(f"   {'✅' if ok else '❌'} {filename:42} -> {rule['document_type']}")
    if not ok:
        failures.append(f"filename {filename}: {rule['document_type']} != {expected}")

# --- 2. realistic bodies under an uninformative filename ---------------------
BODIES = {
    "de_nghi_vay_von": (
        "GIẤY ĐỀ NGHỊ VAY VỐN KIÊM PHƯƠNG ÁN TRẢ NỢ\nThông tin khách hàng: Nguyễn "
        "Văn A\nNơi công tác: Công ty ABC theo hợp đồng lao động\nThu nhập hằng "
        "tháng: lương 35 triệu\nMục đích vay: mua căn hộ\nTài sản bảo đảm: thế "
        "chấp căn hộ hình thành từ vốn vay\nNguồn trả nợ: từ lương"
    ),
    "sao_ke_tai_khoan": (
        "SAO KÊ TÀI KHOẢN\nSố tài khoản: 0123xxx\nSố dư đầu kỳ 12.000.000\n"
        "05/01/2026 LUONG T12 CONG TY ABC 35.000.000\n10/01/2026 TRA GOP FE "
        "CREDIT 4.000.000\nSố dư cuối kỳ"
    ),
    "xac_nhan_luong": (
        "GIẤY XÁC NHẬN THU NHẬP\nCông ty ABC xác nhận ông Nguyễn Văn A hiện đang "
        "công tác tại công ty theo hợp đồng lao động không xác định thời hạn, "
        "mức lương hằng tháng 35.000.000 đồng"
    ),
    "cic_khach_hang_vay": (
        "TRUNG TÂM THÔNG TIN TÍN DỤNG QUỐC GIA\nBÁO CÁO THÔNG TIN TÍN DỤNG\nDư "
        "nợ thẻ tín dụng\nNhóm nợ: Nợ đủ tiêu chuẩn\nKhông có nợ xấu trong 5 năm"
    ),
    "giay_chung_nhan_qsdd": (
        "GIẤY CHỨNG NHẬN QUYỀN SỬ DỤNG ĐẤT QUYỀN SỞ HỮU NHÀ Ở VÀ TÀI SẢN KHÁC "
        "GẮN LIỀN VỚI ĐẤT\nThửa đất số 12, tờ bản đồ số 5"
    ),
    "bao_cao_tham_dinh": (
        "BÁO CÁO THẨM ĐỊNH KHÁCH HÀNG CÁ NHÂN\nXác minh nơi ở: cán bộ đến tận "
        "nơi\nXác minh nơi làm việc: gọi điện phòng nhân sự\nĐề xuất cấp tín dụng"
    ),
}
print("\n2. Nội dung thực tế, tên file không mang thông tin (scan001.pdf)")
for expected, body in BODIES.items():
    rule = rule_classify_document("scan001.pdf", body)
    ok = rule["document_type"] == expected
    print(
        f"   {'✅' if ok else '❌'} {expected:24} -> {rule['document_type']:24} "
        f"conf={rule['confidence']:.2f} gọi LLM={'CÓ' if calls_llm(rule) else 'KHÔNG'}"
    )
    if not ok:
        failures.append(f"body {expected}: {rule['document_type']}")

# --- 3. filename_decisive never relabels ------------------------------------
saved = wrong = still = 0
for a in tids:
    keywords = list(matrix.types[a].keywords)
    fn = f"{keywords[0]}.pdf"
    fn_hits = {t: s for t, s in document_type_scores(fn, "").items() if s}
    if a not in fn_hits:
        continue
    fn_top = max(fn_hits, key=lambda t: fn_hits[t])
    for b in tids:
        if b == a:
            continue
        rule = rule_classify_document(fn, "\n".join(list(matrix.types[b].keywords)[:3]))
        if not calls_llm(rule, with_new_rule=False):
            continue
        if calls_llm(rule, with_new_rule=True):
            still += 1
        elif rule["document_type"] == fn_top:
            saved += 1
        else:
            wrong += 1
            failures.append(f"scan: {fn} -> {rule['document_type']} != {fn_top}")

print(f"\n3. Quét {len(tids)}x{len(tids)} cặp loại tài liệu (ca trước đây phải gọi LLM)")
print(f"   bỏ được LLM, đúng ý tên file : {saved}")
print(f"   bỏ LLM nhưng mâu thuẫn (SAI) : {wrong}  (phải là 0)")
print(f"   vẫn gọi LLM                  : {still}")

changed = 0
for fn in FILENAMES:
    for body in ("", *BODIES.values()):
        rule = rule_classify_document(fn, body)
        if rule["filename_decisive"] and rule["document_type"] != filename_keyword_owner(fn):
            changed += 1
            failures.append(f"label: {fn} decisive but label != owner")
print(f"   nhãn bị đổi bởi luật filename_decisive: {changed}  (phải là 0)")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("All checks passed.")
