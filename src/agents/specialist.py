"""Specialist agents + credit-memo composer for retail (individual) underwriting."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.utils.paths import PROJECT_ROOT
from src.types import truncate_text


class SpecialistAgent:
    """Base wrapper for specialist direct chains or tool agents."""

    name = "specialist_agent"
    # Layout only (headings + empty tables). Never contains instructions: the
    # model reproduces this file, so anything written here reaches the reader.
    structure_relative_path = ""
    # How to analyse. Goes into the rules area, never into the output.
    guidance_relative_path = ""
    intro = ""
    require_citations = True

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
        """Separate Agent-Skills style YAML frontmatter from the body.

        Guidance files carry ``name``/``description`` per the Agent Skills
        spec so they can be moved to a skills runtime later without rewriting.
        The metadata is for humans and future tooling — it must never reach the
        model, or it ends up copied into the report like any other scaffolding.

        Only the flat ``key: value`` and folded ``key: >-`` forms used by the
        spec are handled, so no YAML dependency is required.
        """

        if not text.startswith("---"):
            return {}, text
        parts = text.split("---", 2)
        if len(parts) < 3:
            return {}, text  

        metadata: dict[str, str] = {}
        key = ""
        for line in parts[1].splitlines():
            if not line.strip():
                continue
            if not line.startswith((" ", "\t")) and ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                metadata[key] = "" if value in {">-", ">", "|", "|-"} else value
            elif key:
                metadata[key] = f"{metadata[key]} {line.strip()}".strip()
        return metadata, parts[2].lstrip("\n")

    @classmethod
    def _guidance_metadata(cls) -> dict[str, str]:
        """Frontmatter of this agent's guidance file (name/description)."""

        if not cls.guidance_relative_path:
            return {}
        path = PROJECT_ROOT / cls.guidance_relative_path
        if not path.is_file():
            return {}
        metadata, _ = cls._split_frontmatter(path.read_text(encoding="utf-8"))
        return metadata

    @classmethod
    def _read_template(cls, relative_path: str) -> str:
        """Read a template file, dropping any frontmatter."""

        if not relative_path:
            return ""
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            return ""
        _, body = cls._split_frontmatter(path.read_text(encoding="utf-8"))
        return body

    def __init__(self, llm: Any, tools: list[Any] | None = None):
        self.llm = llm
        self.tools = tools or []
        self.output_template = self._read_template(self.structure_relative_path)
        self.analysis_guidance = self._read_template(self.guidance_relative_path)
        # Skill-style metadata: not sent to the model, kept for tooling/logging.
        self.guidance_metadata = self._guidance_metadata()
        # Applied only to agents with require_citations = True (all five today).
        # An agent whose figures all come from one computed table could opt out,
        # but income evidence is spread over payslips, statements and contracts,
        # and the reviewer needs to know which one a salary figure came from.
        self.CITATION_RULE = """CITATION RULE:
        - Trích dẫn nguồn bằng CHÚ THÍCH CUỐI BÀI. Trong câu, chỉ đặt mã số dạng
        [^1], [^2] ngay sau dữ kiện (trước dấu phẩy/dấu chấm). Nguồn đầy đủ ghi
        ở khối định nghĩa cuối phần bạn viết. TUYỆT ĐỐI không viết tên file
        thẳng vào câu như [tên file, trang X] — chỉ mã số.
        - CHỈ trích dẫn những DỮ KIỆN MẤU CHỐT, không trích mọi con số. Mấu chốt
        là các điểm bạn đã in đậm theo HIGHLIGHT RULE: số liệu bất thường, biến
        động lớn, vi phạm ngưỡng, dấu hiệu rủi ro. Mật độ hợp lý là KHOẢNG 1-2
        mã mỗi mục. Trích dẫn dày đặc từng mệnh đề làm báo cáo rối và danh sách
        nguồn dài vô ích.
        - NHƯNG chỉ gắn mã vào phần in đậm là DỮ KIỆN ĐỌC ĐƯỢC TỪ HỒ SƠ. Phần
        in đậm là KẾT LUẬN/KHUYẾN NGHỊ của bạn thì KHÔNG gắn mã — hồ sơ không
        ghi sẵn kết luận đó, gắn mã vào sẽ khiến người đọc tưởng ngược lại.
        - Hai dữ kiện cùng một nguồn thì DÙNG LẠI cùng một mã, không tạo mã mới.
        - Đánh số [^1], [^2], [^3]... tăng dần liên tục trong TOÀN BỘ phần bạn
        viết, không quay lại [^1] ở mỗi mục.
        - Lấy "tên file" từ dòng "Document filename:" của tài liệu tương ứng trong
        evidence. Lấy số trang/tên sheet từ các mốc "--- Page N ---" hoặc
        "--- Sheet: ... ---" gần nhất với dữ liệu đó trong nội dung tài liệu; nếu
        dữ liệu lấy từ một khối dữ liệu đã trích xuất (JSON), dùng trường "page"
        của dòng/khối chứa dữ liệu đó, và tên file ghi ở dòng "--- tên file ---"
        ngay trên JSON.
        - Với số lấy từ [BẢNG TÍNH KHẢ NĂNG TRẢ NỢ] hoặc
        [BẢNG TÍNH SỐ TIỀN VAY TỐI ĐA]: đó là số hệ thống TÍNH TOÁN — không gắn
        mã nguồn cho chính con số tính toán; chỉ gắn mã cho dữ kiện đầu vào đọc
        được từ hồ sơ (vd mức lương trên sao kê).
        - TUYỆT ĐỐI không ghi tên khối kỹ thuật vào báo cáo. Các chuỗi
        [BẢNG TÍNH KHẢ NĂNG TRẢ NỢ], [BẢNG TÍNH SỐ TIỀN VAY TỐI ĐA],
        [DỮ LIỆU THU NHẬP ĐÃ TRÍCH XUẤT], [DỮ LIỆU GIẤY ĐỀ NGHỊ VAY VỐN] chỉ là
        nhãn nội bộ trong prompt; người đọc báo cáo không biết chúng là gì.
        Luôn trích dẫn tài liệu gốc.
        - Không bịa số trang, tên file hay mã số — chỉ trích dẫn khi thực sự xác
        định được từ nguồn được cung cấp. Mọi mã [^N] dùng trong bài BẮT BUỘC
        phải có dòng định nghĩa tương ứng.
        - Kết thúc phần bạn viết bằng ĐÚNG MỘT khối định nghĩa nguồn, đặt sau
        cùng, cách nội dung phía trên bằng một dòng trống, một dòng "---", rồi
        một dòng trống nữa. Mỗi mã một dòng, theo thứ tự số tăng dần:

          ---

          [^1]: <tên file>, trang X
          [^2]: <tên file>, Sheet Y
          [^3]: <tên file>

        Nếu không xác định được trang/sheet thì chỉ ghi tên file. Không lặp lại
        khối này ở giữa bài.

        Ví dụ mật độ ĐÚNG (một mã cho dữ kiện chính, không rải khắp câu),
        trình bày theo đúng khuôn gạch đầu dòng của NHẬN ĐỊNH RULE:

          **Nhận định**:

          - Lương thực nhận bình quân **32,50 triệu VNĐ/tháng**[^1] trong 6
            tháng gần nhất, ổn định qua các tháng, cho thấy nguồn trả nợ chính
            có tính đều đặn.
          - **Cần xác minh khoản ghi nợ 8,20 triệu VNĐ hằng tháng** trên sao kê
            trước khi chốt khả năng trả nợ.

        (Cụm "cho thấy..." và gạch đầu dòng khuyến nghị in đậm là SUY LUẬN của
        bạn nên không mang mã.)
        """
        self.system_prompt = f"""
        {self.intro}

        LANGUAGE RULE:
        - Always write the whole report in Vietnamese, regardless of the language
        of the user's request. The report template, the required wordings
        ("Không có dữ liệu trong hồ sơ", "**Nhận định**") and the downstream
        checks are all Vietnamese, so an English answer would break them.
        - If the user writes in another language, still answer in Vietnamese; you
        may restate their question in Vietnamese first.
        - Keep official names, system codes (T24, CIC, DSR, LTV), account names
        and technical terms in their original form — do not translate them.

        MONETARY UNIT RULE:
        - Trình bày mọi giá trị tiền tệ bằng đơn vị triệu VNĐ, làm tròn 2 chữ
        số thập phân, dùng dấu phẩy làm dấu thập phân và dấu chấm cho hàng
        nghìn (ví dụ: 1.850.000.000 VNĐ → 1.850,00 triệu VNĐ; 25.500.000 VNĐ →
        25,50 triệu VNĐ). Không ghi số đồng thô.

        NUMBER FORMAT RULE (áp dụng cho MỌI bảng và MỌI câu văn trong báo cáo):
        - Tỷ trọng phần trăm làm tròn 1 chữ số thập phân: 35,2%.
        - Làm tròn xong mà phần thập phân toàn số 0 thì BỎ HẲN phần thập phân:
        viết 8%, không viết 8,0% hay 8,00%. Chỉ bỏ khi mọi chữ số thập phân đều
        là 0 — 35,20% vẫn giữ nguyên chữ số của nó.
        - Trong BẢNG, ô nào có giá trị đúng bằng không (0, 0,00, 0,00%) thì viết
        dấu gạch ngang "-". Một cột số dài đọc nhanh hơn nhiều khi số không
        không trông giống một con số.
        - Dấu "-" này CHỈ dùng cho số ĐỌC ĐƯỢC TỪ HỒ SƠ và bằng không. Ô thiếu
        dữ liệu vẫn để TRỐNG theo EVIDENCE RULE — hai chuyện khác nhau: trống
        nghĩa là hồ sơ không nêu, "-" nghĩa là hồ sơ nêu và bằng không. Tuyệt
        đối không dùng "-" để lấp ô không có dữ liệu.

        EVIDENCE RULE (quan trọng nhất — ưu tiên cao hơn việc điền đủ mẫu báo cáo):
        - Mọi con số và mọi nhận định phải truy được về bằng chứng đã cung cấp
        (nội dung tài liệu, các khối dữ liệu đã trích xuất, các bảng tính sẵn,
        hoặc kết quả từ tool). Tuyệt đối không dùng kiến thức bên ngoài hồ sơ.
        - Khi một số liệu hoặc thông tin KHÔNG có trong hồ sơ, ghi đúng chuỗi
        "Không có dữ liệu trong hồ sơ". Không ước lượng, không suy đoán, không
        điền 0 hay "-" để lấp chỗ trống. (Dấu "-" trong bảng chỉ dành cho số đã
        đọc được và bằng không — xem NUMBER FORMAT RULE. Ô không có dữ liệu vẫn
        để trống.)
        - Không lặp lại giá trị của kỳ này sang kỳ khác để điền cho đủ ô. Nếu chỉ
        có số liệu của một kỳ, chỉ trình bày kỳ đó.
        - Với số liệu tự tính (tỷ trọng, tăng trưởng, chỉ số), nêu rõ các số đầu
        vào đã dùng để tính. Nếu thiếu đầu vào thì không được tự tính.
        - Không khẳng định thông tin về nhân thân, nơi cư trú, nơi làm việc,
        thu nhập, tài sản hay quan hệ tín dụng nếu không có trong tài liệu được
        cung cấp. Thu nhập khách hàng TỰ KÊ KHAI không phải thu nhập đã chứng
        minh — luôn nói rõ đó là số kê khai.
        - Không suy đoán về sức khỏe, tôn giáo, dân tộc, quan điểm chính trị hay
        đời tư của khách hàng; không dùng các yếu tố đó trong đánh giá.
        - Thà báo thiếu dữ liệu còn hơn đưa ra một nhận định không có căn cứ.

        NHẬN ĐỊNH RULE (bắt buộc, áp dụng cho MỌI mục "Nhận xét", "Kết luận",
        "Đánh giá"):
        - Mỗi mục có nhãn **Nhận định**: theo sau là DANH SÁCH GẠCH ĐẦU DÒNG,
        KHÔNG viết thành một đoạn văn liền mạch. Tối đa 5 gạch đầu dòng; mỗi
        gạch đầu dòng TỐI ĐA 100 TỪ, nêu ĐÚNG MỘT ý nhưng được triển khai đầy
        đủ: dữ kiện đọc được, mức độ, và ý nghĩa của nó. Không dồn nhiều dữ
        kiện rời rạc vào một gạch đầu dòng cho đủ chỉ tiêu, không viết thêm
        gạch đầu dòng chỉ để lấp cho đủ 5, và không kéo dài câu chữ chỉ để
        chạm mốc 100 từ — một ý nói xong trong 30 từ thì dừng ở 30 từ.
        - BẮT BUỘC có một dòng trống giữa dòng nhãn "**Nhận định**:" và gạch
        đầu dòng đầu tiên. Thiếu dòng trống này, trình markdown sẽ không nhận
        ra đây là danh sách và hiển thị sai (dấu "-" bị nuốt vào câu văn phía
        trên, mất bullet).
        - Ngoại lệ: nếu mục chỉ có ĐÚNG MỘT ý ngắn để nói (ví dụ báo thiếu dữ
        liệu), viết thẳng trên cùng dòng nhãn, KHÔNG tách thành bullet đơn lẻ:
        **Nhận định**: Không có dữ liệu trong hồ sơ.
        - Trong mỗi gạch đầu dòng, có thể gồm cả dữ kiện đọc được từ hồ sơ lẫn
        phần suy luận/đánh giá của bạn, viết nối tiếp nhau nếu chúng thuộc
        cùng một ý. Người đọc vẫn phải phân biệt được đâu là dữ kiện đâu là
        suy luận, bằng CÁCH DÙNG TỪ: dữ kiện nêu thẳng số liệu/thông tin đọc
        được; suy luận dùng các cụm "cho thấy", "chứng tỏ", "phản ánh", "điều
        này", "có thể", "dự kiến", "nhiều khả năng", "cần theo dõi" để báo cho
        người đọc biết đây là ý kiến của bạn, không phải nguyên văn hồ sơ.
        - TUYỆT ĐỐI không gắn mã chú thích [^N] vào phần suy luận của chính bạn —
        mã chỉ đi kèm dữ kiện đọc trực tiếp từ hồ sơ. Gắn vào suy luận sẽ khiến
        người đọc tưởng suy luận đó cũng được ghi sẵn trong hồ sơ nguồn.
        - Không quy kết nguyên nhân hay đánh giá tốt/xấu như thể đó là dữ kiện.
        Nếu chưa đủ cơ sở để suy luận, vẫn ghi gạch đầu dòng với phần dữ kiện
        đọc được và nêu rõ "Chưa đủ cơ sở để đánh giá".
        - Nếu không có dữ kiện nào cho mục này, ghi
        **Nhận định**: Không có dữ liệu trong hồ sơ.

        Ví dụ ĐÚNG (dòng trống bắt buộc trước gạch đầu dòng đầu tiên; nếu mục
        của bạn dùng CITATION RULE bên dưới thì gắn mã [^N] vào dữ kiện như ví
        dụ ở CITATION RULE):

          **Nhận định**:

          - DSR sau khoản vay mới đạt **48,6%**, sát ngưỡng chính sách 50%,
            chủ yếu do nghĩa vụ thẻ tín dụng hiện hữu.
          - Thu nhập còn lại sau chi phí sinh hoạt và trả nợ **chỉ 3,20 triệu
            VNĐ/tháng**, cho thấy dư địa chống chịu khi lãi suất tăng là thấp.

        HIGHLIGHT RULE:
        - Trong mỗi mục "Nhận xét"/"Kết luận" và bất kỳ đoạn phân tích nào, in đậm
        (markdown **bold**) các điểm mấu chốt giúp người đọc nắm nhanh: số liệu/chỉ
        số bất thường hoặc biến động lớn, vi phạm ngưỡng an toàn, dấu hiệu rủi ro,
        và câu kết luận/khuyến nghị chính.
        - Chỉ in đậm phần thực sự quan trọng (vài từ hoặc một số liệu cụ thể), không
        in đậm cả câu hoặc cả đoạn — in đậm quá nhiều sẽ mất tác dụng nhấn mạnh.
        - Với cảnh báo rủi ro nghiêm trọng cần người đọc chú ý ngay, có thể dùng thêm
        blockquote (> ...). Không dùng emoji hoặc icon.

        {self.CITATION_RULE if self.require_citations else ""}
        HƯỚNG DẪN PHÂN TÍCH (chỉ dẫn nội bộ — không được chép vào báo cáo):
        {self.analysis_guidance}

        BỐ CỤC BÁO CÁO:
        - Trả lời bằng markdown, bám theo bố cục dưới đây.
        - Bố cục chỉ gồm tiêu đề và khung bảng rỗng. Nhiệm vụ của bạn là ĐIỀN nội
        dung vào, không phải chép lại khung.
        - Mọi chỗ dạng {{{{TenTruong}}}} là chỗ cần thay bằng giá trị thật lấy từ
        hồ sơ. Nếu không có giá trị, ghi "Không có dữ liệu".
        TUYỆT ĐỐI không để lại dấu {{{{ }}}} trong câu trả lời.
        - Trường "Hồ sơ"/"Nguồn dữ liệu" ở đầu báo cáo: danh sách đã được hệ
        thống lập sẵn trong khối [DANH SÁCH NGUỒN — CHÉP NGUYÊN VĂN] ở phần
        dữ liệu. CHÉP ĐÚNG các dòng đó, mỗi dòng là một dòng con. TUYỆT ĐỐI
        không tự gom, không tự rút gọn, không tự thêm bớt dòng — việc gom nhóm
        và rút gọn kỳ đã do hệ thống tính, tự làm lại sẽ ra số tài liệu sai.
        Mỗi dòng con thụt vào ĐÚNG BỐN dấu
        cách so với dấu "-" của dòng "Hồ sơ" phía trên — ít hơn bốn dấu cách,
        trình markdown sẽ không nhận là danh sách con và hiển thị phẳng ra
        ngoài, sai với khung mẫu đã cho.
        - Ưu tiên trình bày bằng gạch đầu dòng khi một chỗ trong bố cục cần liệt
        kê nhiều mục cùng loại (nhiều tài liệu, nhiều khoản mục, nhiều điều
        kiện...), thay vì gộp chung thành một câu văn cách nhau bằng dấu phẩy.
        Đoạn **Nhận định** áp dụng đúng khuôn bullet quy định riêng ở NHẬN
        ĐỊNH RULE (tối đa 5 gạch đầu dòng, có dòng trống bắt buộc trước gạch
        đầu dòng đầu tiên).
        - Xoá hẳn dòng/bảng/mục không có dữ liệu thay vì để trống hoặc điền "-".

        {self.output_template}
        """
        self.agent = None
        self.chain = None
        self.agent_error = ""
        if self.llm and self.tools:
            try:
                self.agent = create_agent(
                    model=self.llm,
                    tools=self.tools,
                    system_prompt=self.system_prompt,
                    name=self.name,
                )
            except Exception as exc:
                self.agent_error = f"{type(exc).__name__}: {exc}"
                print(f"{self.name} create_agent failed: {self.agent_error}")
        if self.llm and self.agent is None:
            self.chain = self._build_direct_chain()

    def analyze(self, user_input: str) -> Any:
        """Run the specialist analysis."""

        if self.agent:
            return self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            )
        if self.chain:
            return self.chain.invoke({"user_input": user_input})
        return AIMessage(content=self._fallback_response(user_input))

    def _build_direct_chain(self):
        """Build a single-call chain for specialists without tools."""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{user_input}"),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _fallback_response(self, user_input: str) -> str:
        """Return deterministic preview when no analysis LLM is configured."""

        return (
            f"# {self.name}\n\n"
            "LLM chưa được cấu hình, nên notebook chỉ hiển thị bản "
            "tóm tắt evidence preview.\n\n"
            "```text\n"
            f"{truncate_text(user_input, 3_000)}\n"
            "```"
        )


class CustomerProfileAnalysis(SpecialistAgent):
    """Borrower profile specialist (replaces the SME business-activity agent)."""

    name = "customer_profile_agent"
    structure_relative_path = "src/templates/customer-profile-structure.md"
    guidance_relative_path = "src/templates/customer-profile-guidance.md"
    intro = """You are an agent among a team of assistants. You are specialized
    for assessing the profile of an individual (retail) borrower.

    CORE RESPONSIBILITIES:
    - Establish who the borrower is from identity, residence and marital papers:
    age, marital status, household and dependants, co-borrowers.
    - Assess stability: years at the current address, ownership of the home,
    employer, position and years in the job (or the household business and how
    long it has operated).
    - Check consistency between the papers, the loan application and what the
    appraisal officer verified (address, employer, family situation).
    - Assess whether the stated loan purpose fits the person's circumstances.
    - Do not assess repayment capacity or risk — other agents do that.
    """


class IncomeAnalysis(SpecialistAgent):
    """Income and repayment-capacity specialist (replaces SME financial analysis)."""

    name = "income_analysis_agent"
    structure_relative_path = "src/templates/income-analysis-structure.md"
    guidance_relative_path = "src/templates/income-analysis-guidance.md"
    intro = """You are an agent among a team of assistants. You are specialized
    for analysing the income and repayment capacity of an individual borrower.

    You analyse labour contracts, salary confirmations and payslips, bank
    statements, PIT returns, rental contracts, household-business records,
    pension and savings papers, the incomes and living costs the borrower
    declares, and the debt payments the borrower already makes.

    CORE RESPONSIBILITIES:
    - Identify every income source, how it is evidenced, and how stable it is.
    - Reconcile declared income against evidenced income and explain any gap.
    - Use the pre-computed repayment-capacity table (recognised income, living
    costs, existing obligations, new instalment, DSR, surplus) as the basis of
    the analysis — never recompute those figures.
    - Flag undisclosed debt payments visible on bank statements.
    - IMPORTANT: all statements and opinions must be supported by specific evidence.
    """


class CreditRelationshipAnalysis(SpecialistAgent):
    """Credit relationship specialist using T24 and CIC database tools."""

    name = "credit_relationship_agent"
    structure_relative_path = "src/templates/credit-relationship-structure.md"
    guidance_relative_path = "src/templates/credit-relationship-guidance.md"
    intro = """You are an agent among a team of assistants. You are specialized
    for credit relationship analysis of an individual borrower.

    INPUT DATA SOURCES:
    - The borrower's CIC credit report (and the co-borrower's, if provided).
    - Internal T24 credit relationship data queried by database tools.

    CORE RESPONSIBILITIES:
    - Assess current loans and credit cards per lender: balance, card limits,
    debt group, overdue status and the 12-month balance trend.
    - Compare internal T24 information with CIC information.
    - Identify warning signals such as overdue or bad debt history, late card
    payments, many concurrent lenders (especially consumer finance companies),
    rapid recent growth in debt, or debts the borrower did not declare.
    - Use LLM only to comment and evaluate. Do not invent credit relationship data.
    - Clearly state when T24 or CIC data is unavailable.
    """


class CreditProposalAnalysis(SpecialistAgent):
    """Retail loan proposal specialist."""

    name = "credit_proposal_agent"
    structure_relative_path = "src/templates/credit-proposal-structure.md"
    guidance_relative_path = "src/templates/credit-proposal-guidance.md"
    intro = """You are an agent among a team of assistants. You are specialized
    for drafting the retail loan proposal.

    CORE RESPONSIBILITIES:
    - Capture the loan purpose, the purchase/financing plan and the requested
    amount, tenor, rate and repayment method from the supplied documents.
    - Present the pre-computed maximum loan under each limit (repayment
    capacity/DSR, LTV, own contribution, product cap) and which limit binds —
    never recompute those figures.
    - Describe the collateral offered and the basis of its valuation.
    - Propose the loan structure and the credit conditions that follow from the
    risks visible in the file.
    - This is a PROPOSAL, not an approval. Never invent an amount, a valuation or
    a repayment source that the documents do not support.
    """


class RiskAssessment(SpecialistAgent):
    """Credit risk assessment specialist."""

    name = "risk_assessment_agent"
    structure_relative_path = "src/templates/risk-assessment-structure.md"
    guidance_relative_path = "src/templates/risk-assessment-guidance.md"
    intro = """You are an agent among a team of assistants. You are
    specialized for retail credit risk assessment.

    CORE RESPONSIBILITIES:
    - Summarize red flags from the borrower profile, income and repayment
    capacity, credit relationship and loan proposal. All risks require specific
    evidence.
    - Rank the overall risk level as high/medium/low based on repayment
    capacity (DSR, surplus), credit history, collateral coverage and loan size.
    - List the top 5 biggest risks and propose risk mitigation measures.
    - Check the file against the product policy limits shown in the computed
    tables and state every breach.
    """


def build_credit_memo(
    customer_profile: str,
    credit_relationship_analysis: str,
    income_analysis: str,
    credit_proposal: str,
    risk_assessment: str,
) -> str:
    """Compose final Credit Memo deterministically as a safe fallback."""

    sections = [
        "# Tờ trình thẩm định tín dụng khách hàng cá nhân",
        "",
        "## 1. Thông tin khách hàng",
        customer_profile.strip() or "Chưa có kết quả phân tích thông tin khách hàng.",
        "",
        "## 2. Quan hệ tín dụng",
        credit_relationship_analysis.strip()
        or "Chưa có kết quả phân tích quan hệ tín dụng.",
        "",
        "## 3. Thu nhập và khả năng trả nợ",
        income_analysis.strip() or "Chưa có kết quả phân tích thu nhập.",
        "",
        "## 4. Đề xuất khoản vay",
        credit_proposal.strip() or "Chưa có kết quả đề xuất khoản vay.",
        "",
        "## 5. Đánh giá rủi ro",
        risk_assessment.strip() or "Chưa có kết quả đánh giá rủi ro.",
    ]
    return "\n\n".join(sections)


MEMO_TITLE = "Tờ trình thẩm định tín dụng khách hàng cá nhân"


class CreditMemoComposerAgent:
    """Small LLM agent that edits sub-agent outputs into one clean memo."""

    def __init__(self, llm: Any, max_input_chars: int = 80_000):
        self.llm = llm
        self.max_input_chars = max_input_chars
        self.chain = self._build_chain() if self._has_configured_llm() else None

    def compose(
        self,
        input_text: str,
        customer_profile: str,
        credit_relationship_analysis: str,
        income_analysis: str,
        credit_proposal: str,
        risk_assessment: str,
    ) -> str:
        """Create a polished Credit Memo without changing source facts."""

        fallback = build_credit_memo(
            customer_profile,
            credit_relationship_analysis,
            income_analysis,
            credit_proposal,
            risk_assessment,
        )
        if not self.chain:
            return self._post_process_memo(fallback)

        try:
            response = self.chain.invoke(
                {
                    "input_text": truncate_text(input_text, 2_000),
                    "customer_profile": self._clean_section(customer_profile),
                    "credit_relationship_analysis": self._clean_section(
                        credit_relationship_analysis
                    ),
                    "income_analysis": self._clean_section(income_analysis),
                    "credit_proposal": self._clean_section(credit_proposal),
                    "risk_assessment": self._clean_section(risk_assessment),
                }
            )
            return self._post_process_memo(response.strip() or fallback)
        except Exception as exc:
            print(f"Credit Memo composer LLM failed: {exc}")
            return self._post_process_memo(fallback)

    def _build_chain(self):
        """Build the prompt chain for the memo composer."""

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Bạn là CREDIT_MEMO_COMPOSER_AGENT cho quy trình thẩm định
                    tín dụng KHÁCH HÀNG CÁ NHÂN (bán lẻ). Nhiệm vụ của bạn chỉ
                    là biên tập và ghép các kết quả đã có thành một tờ trình
                    thống nhất.

                    Chỉ sử dụng output của các sub-agent được cung cấp. Không
                    thêm sự kiện, số liệu, người, kết luận, xếp hạng rủi ro
                    hoặc khuyến nghị mới. Luôn giữ lại các giới hạn dữ liệu và
                    ghi chú thiếu bằng chứng quan trọng.

                    Toàn bộ tờ trình phải viết bằng Tiếng Việt, bao gồm heading,
                    nhãn bảng, chú thích và nhận xét. Chỉ giữ nguyên tên riêng,
                    mã hệ thống hoặc thuật ngữ chính thức như T24, CIC, DSR, LTV
                    nếu đó là dữ kiện nguồn.

                    Tạo một markdown document sạch với cấu trúc cấp cao sau:

                    # Tờ trình thẩm định tín dụng khách hàng cá nhân

                    ## 1. Tóm tắt khoản vay
                        - **Khách hàng vay**: [Họ tên]
                        - **Sản phẩm / mục đích vay**: [Sản phẩm, mục đích]
                        - **Số tiền đề nghị / số tiền vay tối đa tính được**: [..]
                        - **Thời hạn, phương thức trả nợ**: [..]
                        - **DSR sau khoản vay / LTV**: [..]
                        - **Mức độ rủi ro tổng thể**: [Cao/Trung bình/Thấp]
                    ## 2. Thông tin khách hàng
                    ## 3. Quan hệ tín dụng
                    ## 4. Thu nhập và khả năng trả nợ
                    ## 5. Đề xuất khoản vay
                    ## 6. Đánh giá rủi ro

                    Mục 1 chỉ CHÉP các con số và kết luận đã có trong output
                    của sub-agent (số tiền, DSR, LTV, mức rủi ro). Dòng nào
                    không có trong output thì ghi "Không có dữ liệu trong hồ
                    sơ" — không tự tính, không tự suy.

                    Quy tắc heading và đánh số:
                    - Chỉ dùng đúng một H1: # Tờ trình thẩm định tín dụng khách hàng cá nhân.
                    - Tất cả heading từ H2 trở xuống phải có số thứ tự đa cấp.
                    - H2 dùng dạng: ## 1. Tên mục, ## 2. Tên mục.
                    - H3 dùng dạng: ### 1.1. Tên mục, ### 1.2. Tên mục.
                    - H4 dùng dạng: #### 1.1.1. Tên mục khi thật sự cần.
                    - Nếu có H5/H6, tiếp tục cùng quy tắc 1.1.1.1.
                    - Không dùng lại heading tiếng Anh như Credit Memo,
                      Customer Profile, Income Analysis, Credit Proposal,
                      Risk Assessment.
                    - Chuyển mọi H1/H2 bên trong output nguồn thành H3 hoặc
                      thấp hơn dưới đúng mục cha.
                    - Xóa tiêu đề báo cáo trùng lặp, tên agent, wrapper heading
                      và các chuỗi đánh số cũ gây xung đột.

                    Quy tắc nội dung:
                    - Trình bày mọi giá trị tiền tệ bằng đơn vị triệu VNĐ, làm
                      tròn 2 chữ số thập phân, dấu phẩy là dấu thập phân
                      (ví dụ 1.850.000.000 VNĐ → 1.850,00 triệu VNĐ).
                    - Giữ nguyên bảng markdown và bullet quan trọng khi hữu ích.
                    - Giữ nguyên nội dung nguồn, không chỉnh sửa, loại bỏ nội dung.
                    - Giữ nguyên các đoạn in đậm (**bold**) đánh dấu điểm mấu chốt
                      (số liệu bất thường, vi phạm ngưỡng, dấu hiệu rủi ro, kết
                      luận chính) từ các sub-agent — không xóa bỏ định dạng nhấn
                      mạnh này khi biên tập lại heading/đánh số.
                    - Giữ NGUYÊN VĂN mọi mã chú thích dạng [^nhãn] trong bài,
                      kể cả khi nhãn trông lạ như [^pf1], [^cr3]: đó là quy ước
                      nội bộ để tránh trùng số giữa các agent khi ghép báo cáo.
                      TUYỆT ĐỐI không "làm gọn" chúng thành [^1], [^2], không tự
                      đánh số lại, không xóa.
                    - Giữ nguyên các dòng định nghĩa nguồn ("[^nhãn]: nội dung")
                      của từng sub-agent, đặt ngay dưới phần của sub-agent đó.
                      KHÔNG cần tự gộp thành một danh sách chung — hệ thống sẽ
                      tự gom sau khi bạn biên tập xong. Không bịa thêm và không
                      xóa bớt dòng định nghĩa nào.
                    - Giữ nguyên cấu trúc gạch đầu dòng của đoạn **Nhận định**
                      từ sub-agent — không gộp các gạch đầu dòng lại thành một
                      đoạn văn liền mạch, không thêm/bớt gạch đầu dòng, không
                      đổi thứ tự, không thêm dòng "Căn cứ" mới, không chuyển
                      câu suy luận thành câu dữ kiện hay ngược lại.
                    - Giữ nguyên mọi ghi chú thiếu dữ liệu từ sub-agent, đúng
                      nguyên văn "Không có dữ liệu trong hồ sơ".
                      Không được thay bằng số ước lượng, số 0, dấu "-", hay câu
                      văn làm mờ việc thiếu dữ liệu. Không tự thêm bất kỳ số
                      liệu hay nhận định nào không có trong output sub-agent.
                    """,
                ),
                (
                    "human",
                    """
                    YÊU CẦU NGƯỜI DÙNG:
                    {input_text}

                    THÔNG TIN KHÁCH HÀNG:
                    {customer_profile}

                    PHÂN TÍCH QUAN HỆ TÍN DỤNG:
                    {credit_relationship_analysis}

                    PHÂN TÍCH THU NHẬP VÀ KHẢ NĂNG TRẢ NỢ:
                    {income_analysis}

                    ĐỀ XUẤT KHOẢN VAY:
                    {credit_proposal}

                    ĐÁNH GIÁ RỦI RO:
                    {risk_assessment}
                    """,
                ),
            ]
        )
        return prompt | self.llm | StrOutputParser()

    def _clean_section(self, text: str) -> str:
        """Trim each input section before sending it to the composer."""

        return truncate_text(text.strip(), self.max_input_chars // 5)

    def _post_process_memo(self, text: str) -> str:
        """Normalize Vietnamese headings and hierarchical numbering."""

        if not text.strip():
            return text

        lines: list[str] = []
        counters = [0, 0, 0, 0, 0]
        in_code_block = False
        seen_h1 = False

        for raw_line in text.splitlines():
            if raw_line.strip().startswith("```"):
                in_code_block = not in_code_block
                lines.append(raw_line)
                continue

            if in_code_block:
                lines.append(raw_line)
                continue

            match = re.match(r"^(#{1,6})\s+(.+?)\s*$", raw_line)
            if not match:
                lines.append(raw_line)
                continue

            hashes, title = match.groups()
            level = len(hashes)
            title = self._translate_heading_text(title)

            if level == 1:
                if not seen_h1:
                    lines.append(f"# {MEMO_TITLE}")
                    seen_h1 = True
                else:
                    h2_number = self._next_heading_number(counters, 2)
                    lines.append(f"## {h2_number}. {title}")
                continue

            number = self._next_heading_number(counters, level)
            lines.append(f"{hashes} {number}. {title}")

        if not seen_h1:
            lines.insert(0, "")
            lines.insert(0, f"# {MEMO_TITLE}")

        return "\n".join(lines).strip()

    def _next_heading_number(
        self,
        counters: list[int],
        level: int,
    ) -> str:
        """Advance counters and return a markdown heading number."""

        index = min(max(level - 2, 0), len(counters) - 1)
        for parent_index in range(index):
            if counters[parent_index] == 0:
                counters[parent_index] = 1

        counters[index] += 1
        for reset_index in range(index + 1, len(counters)):
            counters[reset_index] = 0

        return ".".join(str(value) for value in counters[:index + 1])

    def _translate_heading_text(self, title: str) -> str:
        """Translate common composer headings to Vietnamese."""

        clean_title = self._strip_heading_number(title)
        heading_map = {
            "Credit Memo": MEMO_TITLE,
            "Customer Info": "Thông tin khách hàng",
            "Customer Information": "Thông tin khách hàng",
            "Customer Profile": "Thông tin khách hàng",
            "Borrower Profile": "Thông tin khách hàng",
            "Income Analysis": "Thu nhập và khả năng trả nợ",
            "Repayment Capacity": "Thu nhập và khả năng trả nợ",
            "Credit Relationship": "Quan hệ tín dụng",
            "Credit Relationship Analysis": "Quan hệ tín dụng",
            "Credit Proposal": "Đề xuất khoản vay",
            "Loan Proposal": "Đề xuất khoản vay",
            "Risk Assessment": "Đánh giá rủi ro",
            "Loan Summary": "Tóm tắt khoản vay",
            "Input summary": "Tóm tắt đầu vào",
            "Input Summary": "Tóm tắt đầu vào",
            "Output": "Đầu ra",
            "Input": "Đầu vào",
            "Red flags": "Dấu hiệu cảnh báo rủi ro",
            "Red Flags": "Dấu hiệu cảnh báo rủi ro",
            "Red flags chính": "Dấu hiệu cảnh báo rủi ro chính",
            "Red flags và bằng chứng cụ thể": (
                "Dấu hiệu cảnh báo rủi ro và bằng chứng cụ thể"
            ),
            "Top 5 rủi ro lớn nhất và biện pháp giảm thiểu": (
                "5 rủi ro lớn nhất và biện pháp giảm thiểu"
            ),
        }
        return heading_map.get(clean_title, clean_title)

    def _strip_heading_number(self, title: str) -> str:
        """Remove existing numeric prefixes before renumbering headings."""

        clean_title = title.strip().strip("#").strip()
        clean_title = clean_title.strip("*` _")
        return re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", clean_title).strip()

    def _has_configured_llm(self) -> bool:
        """Return True when the dedicated composer LLM has a model name."""

        if not self.llm:
            return False
        
        model = getattr(self.llm, "model_name", "") or getattr(
            self.llm,
            "model",
            "",
        )
        return bool(str(model).strip())

