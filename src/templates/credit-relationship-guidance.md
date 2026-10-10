---
name: credit-relationship
description: >-
  Hướng dẫn phân tích quan hệ tín dụng của khách hàng cá nhân: khoản vay và thẻ tín dụng tại các TCTD, nhóm nợ và lịch sử trả nợ, tỷ lệ sử dụng hạn mức thẻ, đối chiếu CIC với dữ liệu nội bộ T24 và với nghĩa vụ khách hàng kê khai, và các dấu hiệu cảnh báo.
  Dùng khi tạo phần quan hệ tín dụng của tờ trình thẩm định.
---

#### NGUYÊN TẮC CHUNG
- Bố cục được cung cấp là khung tham khảo, không phải biểu mẫu bắt buộc điền kín.
- Không lặp giá trị của kỳ này sang kỳ khác để lấp ô trống.
- Bảng biểu chỉ hiển thị những TCTD/thẻ/kỳ có số liệu.
- Nếu không có dữ liệu T24 hoặc CIC, ghi rõ "Không có dữ liệu trong hồ sơ" và nêu
  giới hạn này trong phần Nhận định — không suy đoán dư nợ hay nhóm nợ.
- Các cột số tiền đã ghi đơn vị ở dòng "(Đơn vị: triệu VNĐ)" — mỗi ô CHỈ ghi số
  trần, làm tròn 2 chữ số thập phân (vd "1.574,22"), KHÔNG lặp lại đơn vị trong ô.

#### TRỌNG TÂM PHÂN TÍCH
- Mục 1: khoản vay, thẻ tín dụng và lịch sử trả nợ tại Techcombank lấy từ dữ liệu
  nội bộ (T24) qua công cụ hoặc tài liệu được cung cấp. Không có thì ghi rõ.
- Mục 2.1: dư nợ vay theo từng TCTD, tách ngắn hạn và trung dài hạn. Ghi chú riêng
  các khoản vay tại CÔNG TY TÀI CHÍNH (vay tiêu dùng) — nhiều khoản vay tiêu dùng
  song song là dấu hiệu cần nêu ở Mục 4.
- Mục 2.2: mọi thẻ tín dụng kể cả thẻ dư nợ bằng 0. Tỷ lệ sử dụng hạn mức = dư nợ
  thẻ / hạn mức thẻ, CHỈ tính khi có đủ cả hai số; thiếu một trong hai thì để trống.
  Tỷ lệ sử dụng cao kéo dài là dấu hiệu căng thẳng tài chính.
- Mục 2.3: lấy từ báo cáo CIC tài sản bảo đảm (R20/R21) nếu có. "Mã loại tài sản"
  chép nguyên mã, KHÔNG tự diễn giải nghĩa mã. Tài sản chưa giải chấp đang bảo đảm
  cho khoản vay khác thì không thể dùng làm tài sản bảo đảm cho khoản vay này.
- Mục 2.4: chép nguyên văn kết luận của CIC về nợ cần chú ý, nợ xấu, chậm thanh
  toán thẻ; điểm/xếp hạng tín dụng nếu báo cáo có in.
- Mục 2.5 — biểu đồ do hệ thống chèn tự động từ dữ liệu đã trích xuất — KHÔNG tự vẽ,
  KHÔNG chép lại bảng số vào báo cáo. Việc của bạn là phần Nhận định:
  - Nêu xu hướng dư nợ đọc được (tăng/giảm, đỉnh ở tháng nào, dư nợ cuối kỳ so với
    đầu kỳ), kèm con số và tháng cụ thể.
  - Đối chiếu dư nợ với thu nhập có chứng từ trong cùng các tháng: dư nợ tăng nhanh
    trong khi thu nhập không đổi là dấu hiệu cần nêu ở Mục 4.
  - Tháng nào thiếu số liệu thì nói rõ là thiếu, KHÔNG coi là dư nợ bằng 0.
- Mục 3: báo cáo CIC của vợ/chồng, người đồng vay, người bảo lãnh (nếu hồ sơ có).
  Nêu nhóm nợ và tổng dư nợ của từng người; nợ xấu của người đồng vay ảnh hưởng
  trực tiếp đến khoản vay. Không có hồ sơ thì xoá mục này.
- Mục 4 — dấu hiệu cảnh báo: nợ quá hạn, nợ cần chú ý/nợ xấu trong lịch sử, chậm
  thanh toán thẻ, nhiều khoản vay tiêu dùng song song, dư nợ tăng nhanh gần đây,
  khoản vay trên CIC mà khách hàng KHÔNG kê khai ở giấy đề nghị vay vốn (đối chiếu
  với danh sách nợ hiện hữu trong [DỮ LIỆU GIẤY ĐỀ NGHỊ VAY VỐN] nếu được cung cấp
  hoặc nội dung giấy đề nghị), hoặc bản ghi CIC không nhất quán với T24.
- Chỉ nhận xét và đánh giá trên dữ liệu được cung cấp. KHÔNG tự tạo ra số liệu quan
  hệ tín dụng.
