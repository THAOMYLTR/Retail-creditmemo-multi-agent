---
name: income-analysis
description: >-
  Hướng dẫn phân tích thu nhập và khả năng trả nợ của khách hàng cá nhân: nhận diện và đánh giá chứng từ từng nguồn thu nhập, tính ổn định, đối chiếu thu nhập kê khai với thu nhập có chứng từ, chi phí sinh hoạt, nghĩa vụ trả nợ hiện hữu, DSR và thu nhập còn lại. Dùng khi lập phần thu nhập và khả năng trả nợ của tờ trình thẩm định.
---

#### NGUYÊN TẮC CHUNG
- Bố cục là khung tham khảo, không phải biểu mẫu bắt buộc điền kín.
- Mọi con số về thu nhập ghi nhận, chi phí sinh hoạt, nghĩa vụ trả nợ, kỳ trả nợ
  khoản vay mới, DSR, thu nhập còn lại và tuổi khi đáo hạn ĐÃ ĐƯỢC TÍNH SẴN trong
  [BẢNG TÍNH KHẢ NĂNG TRẢ NỢ]. CHÉP đúng các con số đó; TUYỆT ĐỐI không tự tính
  lại, không làm tròn khác đi.
- Cột "Nguồn" của bảng tính phải được nhắc lại khi bình luận: "chính sách" là tham
  số chính sách (vd mức sàn chi phí sinh hoạt, tỷ lệ nghĩa vụ thẻ), KHÔNG phải số
  liệu của khách hàng; "đề nghị vay" là số khách hàng tự kê khai, chưa chứng minh.
- Mọi dòng "CẢNH BÁO" dưới bảng tính phải được nêu lại trong Nhận định của mục
  liên quan — không được bỏ qua cảnh báo nào.
- Bảng chỉ hiển thị nguồn thu nhập, tháng và khoản mục có số liệu thật. Không lặp
  số của tháng này sang tháng khác để lấp ô trống.
- Ô có giá trị đúng bằng không viết `-`; ô THIẾU dữ liệu để trống.

#### THỨ BẬC CHỨNG TỪ THU NHẬP
Cùng một khoản thu nhập có thể xuất hiện trên nhiều giấy tờ. Độ tin cậy giảm dần:
1. Sao kê tài khoản thể hiện tiền THỰC NHẬN hằng tháng.
2. Xác nhận lương/bảng lương của đơn vị sử dụng lao động; sổ sách hộ kinh doanh.
3. Chứng từ thuế TNCN (thu nhập cả năm).
4. Hợp đồng lao động/hợp đồng cho thuê — chỉ là mức THỎA THUẬN, chưa phải tiền đã nhận.
5. Thu nhập khách hàng tự kê khai trên giấy đề nghị — không phải chứng từ.
Bảng tính đã áp dụng thứ bậc này. Cột "Chứng từ" ở mục 1 ghi đúng loại chứng từ
mà bảng tính đã dùng cho nguồn đó.

#### TRỌNG TÂM PHÂN TÍCH
- Mục 1: mỗi nguồn thu nhập một dòng, lấy từ các dòng "Thu nhập bình quân tháng —
  ..." của bảng tính. Nêu rõ nguồn nào chỉ có hợp đồng hoặc chỉ có kê khai.
  - Mục 1.1: lấy từ khối [DỮ LIỆU THU NHẬP ĐÃ TRÍCH XUẤT] — các dòng
    "monthly_income" theo từng tháng. Đánh giá tính đều đặn: tháng giảm mạnh, tháng
    thiếu, khoản thưởng bất thường làm bình quân cao lên. Lương nhận bằng tiền mặt
    (không qua tài khoản) thì ghi rõ là chưa có sao kê đối chiếu.
  - Mục 1.2: đối chiếu thu nhập khách hàng kê khai trên giấy đề nghị với thu nhập
    có chứng từ. Chênh lệch lớn phải giải thích được bằng hồ sơ; không giải thích
    được thì nêu là điểm cần làm rõ.
- Mục 2: chép từ bảng tính. Đặc biệt lưu ý:
  - Khoản trả nợ định kỳ thấy trên sao kê ("recurring_debits") mà khách hàng không
    kê khai là dấu hiệu khoản vay chưa khai báo — nêu rõ tên bên nhận và số tiền.
  - Nghĩa vụ ước tính từ CIC và nghĩa vụ thẻ là GIẢ ĐỊNH theo chính sách, phải nói rõ.
- Mục 3: sơ đồ dòng tiền hằng tháng — các nguồn thu nhập ghi nhận đi vào khách
  hàng, rồi chia ra chi phí sinh hoạt, trả nợ hiện hữu, trả nợ khoản vay mới và thu
  nhập còn lại. Nhãn trên mũi tên là số tiền triệu VNĐ lấy ĐÚNG từ bảng tính (vd
  `-->|32,35 triệu|`), TỐI ĐA 10 TỪ — hệ thống vẽ ĐẦY ĐỦ nhãn bạn viết, không cắt
  bớt chữ nào, nên nhãn dài làm sơ đồ xấu chứ không bị giấu đi. Bắt buộc dùng
  Mermaid, hàng rào ```mermaid không thụt đầu dòng; KHÔNG tự chọn màu, KHÔNG viết
  classDef. Thu nhập còn lại ÂM thì vẫn vẽ và ghi số âm. Không đủ số liệu thì bỏ
  sơ đồ.
- Mục 4: chép giá trị, ngưỡng chính sách và đánh giá (Đạt/Vượt ngưỡng) từ bảng tính.
  Nhận định nêu DSR so với ngưỡng, mức thu nhập còn lại so với chi phí sinh hoạt,
  và mức chống chịu nếu lãi suất tăng (chỉ nói định tính — không tự tính kịch bản).
- Mục 5: kết luận ngắn gọn về mức độ chứng minh thu nhập, tính ổn định và khả năng
  trả nợ. KHÔNG đề xuất số tiền vay — đó là việc của phần đề xuất khoản vay.
