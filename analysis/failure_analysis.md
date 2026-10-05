# Failure Analysis — Lab 18: Production RAG

**Họ và tên:** PhamHoangTrong
**MSSV:** 2A202602765
**Khóa:** K4 — Track 3B
**Ngày hoàn thiện:** 05/10/2026

## 1. Dữ liệu và phạm vi đánh giá

Phân tích dùng báo cáo cuối cùng `reports/ragas_report.json` và `reports/naive_baseline_report.json`, gồm 20 câu hỏi. Không ghép điểm tốt nhất của các lượt chạy. Baseline dùng paragraph chunking, dense-only và top 3; production dùng M1 hierarchical → M5 enrichment trên child → M2 hybrid/RRF → M3 rerank → parent context gốc → LLM → M4 RAGAS. Bản cũ được giữ làm tham chiếu sau bản hiện hành, và câu nhiều ý có tìm kiếm bổ sung theo từng vế.

Kho dữ liệu có 28 tài liệu sau OCR hai PDF scan; baseline và production cùng đọc kho này. Baseline không được cố tình làm yếu để tạo chênh lệch. Đáp án chuẩn chỉ đi vào đánh giá, không dùng làm đầu vào sinh câu trả lời.

## 2. Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| `faithfulness` | 0.8042 | 0.8305 | +0.0263 |
| `answer_relevancy` | 0.7091 | 0.8830 | +0.1739 |
| `context_precision` | 0.9250 | 0.9917 | +0.0667 |
| `context_recall` | 0.9250 | 0.9583 | +0.0333 |

Production vượt baseline ở cả bốn chỉ số. Cải thiện lớn nhất là answer relevancy (+0,1739), phù hợp với việc câu hỏi nhiều ý lấy được cả chính sách phép và bảng lương. Cả bốn metric đều trên 0,75; faithfulness 0,8305 vẫn chưa đạt mốc bonus 0,85. Đây là điểm trung bình của bộ 20 câu, không phải phần trăm câu trả lời đúng và không chứng minh độ tin cậy trên một tập dữ liệu lớn hơn.

## 3. Bottom-5 và Diagnostic Error Tree

Bottom-5 được sắp tăng dần theo trung bình cộng bốn metric, đúng cách `failure_analysis()` thực hiện. Với các trường hợp này, metric thấp nhất đều là faithfulness. Chẩn đoán tự động của M4 là “LLM có thể bịa thông tin ngoài tài liệu”; phần kiểm tra thủ công dưới đây xác định khi nào chẩn đoán đó có căn cứ, khi nào cần kiểm tra lại bộ chấm.

Cây kiểm tra chung:

```text
Điểm thấp / output đáng nghi
├─ Output có khớp yêu cầu và phép tính không?
│  ├─ Không → kiểm tra điều kiện, đơn vị, ngoại lệ và prompt
│  └─ Có → không tự động kết luận hallucination
├─ Context có đủ căn cứ gốc cho từng claim không?
│  ├─ Không → kiểm tra retrieval, chunking, phiên bản và dữ liệu nguồn
│  └─ Có → kiểm tra generation, citation và evaluator
└─ Query có nhiều ý / yêu cầu lịch sử / thiếu quy ước không?
   ├─ Nhiều ý → bổ sung truy vấn theo vế, giữ đủ nguồn trong top-k
   ├─ Lịch sử → bản hiện hành trước, bản cũ làm tham chiếu khi phù hợp
   └─ Thiếu quy ước → nêu giả định, không trình bày như quy định
```

### #1 — Câu 17: Tính toán có giả định chưa được tài liệu xác nhận

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày).
- **Got:** Nhân viên tạm ứng 15 triệu VNĐ, chậm thanh toán 5 ngày (20 ngày - 15 ngày).   Phạt = 2% x 15.000.000 VNĐ x 5/30 (tính theo tháng) = 2% x 15.000.000 x 0,1667 ≈ 50.000 VNĐ.  Kết quả: Nhân viên bị phạt khoảng 50.000 VNĐ.
- **Scores:** faithfulness 0.1429; relevancy 0.8252; precision 1.0000; recall 0.6667. **Trung bình:** 0.6587.
- **Worst metric:** `faithfulness`.
- **Error Tree:** Có phần suy luận vượt nguồn → Context chứa chính sách đúng nhưng thiếu quy tắc pro-rata → Query có đủ số tiền và số ngày → Lỗi căn cứ ở generation và khoảng trống tài liệu/đáp án chuẩn.
- **Root cause / nhận định:** Tài liệu tam_ung.md có hạn 15 ngày và phí 2%/tháng; không quy định tháng 30 ngày hoặc chia phí theo số ngày. Đáp án chuẩn cũng dùng giả định pro-rata. Kết quả 50.000 VNĐ gần đáp án chuẩn nhưng câu trả lời khẳng định số tiền phải trả, không tách giả định khỏi quy định. Đây là thiếu căn cứ thật, không phải thiếu tài liệu tạm ứng.
- **Suggested fix:** Trả lời chắc chắn phần có nguồn: quá hạn 5 ngày, mức phí 300.000 VNĐ/tháng. Nếu minh họa pro-rata, ghi rõ điều kiện giả sử tháng 30 ngày và phí chia theo ngày; khi chưa xác nhận quy ước thì không khẳng định phí thực tế. Đề xuất chủ sở hữu chính sách bổ sung quy tắc tính phí; không tự sửa corpus hay đáp án để nâng điểm.

### #2 — Câu 14: Cần kiểm tra sai số của bộ chấm với suy luận số học

- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu?
- **Expected:** Nhân viên phải cam kết làm việc ít nhất 1 năm sau khi hoàn thành khóa học. Nghỉ sau 8 tháng là trước hạn cam kết, phải hoàn trả 100% chi phí tức 25.000.000 VNĐ.
- **Got:** Nhân viên phải hoàn trả **100% chi phí** đào tạo đã được tài trợ, tức là **25.000.000 VNĐ**, vì nhân viên nghỉ việc trước thời hạn cam kết 1 năm.   Nguồn: Chính sách hoàn chi đào tạo.
- **Scores:** faithfulness 0.5000; relevancy 0.7843; precision 1.0000; recall 1.0000. **Trung bình:** 0.8211.
- **Worst metric:** `faithfulness`.
- **Error Tree:** Output khớp expected → Context đủ điều kiện hoàn chi → Query đủ dữ kiện → Kiểm tra cách evaluator phân rã và xác minh suy luận.
- **Root cause / nhận định:** Nguồn hoan_chi_dao_tao.md nêu cam kết 1 năm, nghỉ trước hạn hoàn trả 100%. Câu trả lời 25 triệu đúng phép tính 100% × 25 triệu. Số tiền và 8 tháng lấy từ câu hỏi. Không thấy mâu thuẫn rõ trong phần trả lời đã lưu; faithfulness 0,5 là tín hiệu cần kiểm tra, chưa chứng minh hallucination.
- **Suggested fix:** Tách rõ dữ kiện đề bài, điều kiện từ tài liệu và phép tính. Kiểm tra claim-level judgement của RAGAS bằng một lượt đánh giá chẩn đoán sau này; bổ sung test tính toán xác định và đánh giá thủ công, không sửa điểm đã đo.

### #3 — Câu 18: Câu trả lời đúng số học nhưng faithfulness thấp

- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Lương gross của nhân viên Junior (P1-P2) có mức cao nhất là 20.000.000 VNĐ/tháng. Trong thời gian thử việc, nhân viên nhận 85% lương, nên lương thử việc cao nhất là:  20.000.000 VNĐ * 85% = 17.000.000 VNĐ.  Vậy, lương thử việc cao nhất của nhân viên Junior là 17.000.000 VNĐ. (Nguồn: Bảng lương năm 2024)
- **Scores:** faithfulness 0.5000; relevancy 0.8241; precision 1.0000; recall 1.0000. **Trung bình:** 0.8310.
- **Worst metric:** `faithfulness`.
- **Error Tree:** Output đúng 17 triệu → Context đủ mức trần và 85% → Query rõ Junior và mức cao nhất → Nghi vấn scoring/calculation attribution, chưa có bằng chứng hallucination.
- **Root cause / nhận định:** Bảng lương năm 2024 có Junior tối đa 20 triệu; bảng lương và chính sách thử việc đều nêu 85%. Output tính 20 triệu × 85% = 17 triệu, đúng expected. Context recall bằng 1 cho thấy đáp án chuẩn được ngữ cảnh bao phủ. Cần kiểm tra evaluator trước khi kết luận retrieval lỗi.
- **Suggested fix:** Giữ đơn vị VNĐ/tháng nhất quán, trích hai tiền đề và trình bày một phép tính. Thêm kiểm tra số học ngoài LLM-as-judge; đo lại nhiều lần để phân biệt sai số chấm với lỗi thật.

### #4 — Câu 19: Trích nguồn chưa đủ chính xác và câu trả lời thiếu quy tắc xử lý

- **Question:** Thông tin lương thuộc cấp độ phân loại dữ liệu nào?
- **Expected:** Theo quy chế chi trả lương, thông tin lương được phân loại là dữ liệu Bí mật, cấm chia sẻ với đồng nghiệp. Theo chính sách phân loại dữ liệu, dữ liệu Bí mật (cấp 3) phải mã hóa khi truyền và hạn chế truy cập theo need-to-know.
- **Got:** Thông tin lương thuộc cấp độ phân loại dữ liệu **Bí mật**. Nguồn hỗ trợ: Chính sách phân loại dữ liệu.
- **Scores:** faithfulness 0.5000; relevancy 0.8688; precision 1.0000; recall 1.0000. **Trung bình:** 0.8422.
- **Worst metric:** `faithfulness`.
- **Error Tree:** Output đúng nhãn nhưng thiếu giải thích/nguồn trực tiếp → Context đủ hai chính sách → Query rõ → Lỗi generation và gán nguồn, không phải thiếu retrieval.
- **Root cause / nhận định:** Output kết luận Bí mật đúng, nhưng chỉ ghi nguồn Chính sách phân loại dữ liệu. Tài liệu này mô tả cấp 3 và quy tắc xử lý; quan hệ thông tin lương → Bí mật được quy định rõ ở ky_luong.md. Expected còn yêu cầu cấm chia sẻ, mã hóa khi truyền và need-to-know. Context có đủ nhưng câu trả lời chưa kết nối đầy đủ hai nguồn.
- **Suggested fix:** Dẫn ky_luong.md cho phân loại thông tin lương, phan_loai_du_lieu.md cho cấp 3 và cách xử lý. Trả lời ngắn nhưng đủ quy tắc trực tiếp liên quan, thay vì chỉ nêu nhãn.

### #5 — Câu 20: Câu trả lời đúng phê duyệt nhưng bỏ sót hệ quả liên quan

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của Giám đốc điều hành (CEO). Lưu ý: nghỉ trên 14 ngày không lương, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** Nghỉ phép không lương 20 ngày cần phê duyệt của **Giám đốc điều hành (CEO)**. (Nguồn: Chính sách nghỉ phép không lương, phiên bản 1.2, ngày hiệu lực 01/07/2023)
- **Scores:** faithfulness 0.5000; relevancy 0.9331; precision 1.0000; recall 1.0000. **Trung bình:** 0.8583.
- **Worst metric:** `faithfulness`.
- **Error Tree:** Output đúng CEO nhưng thiếu lưu ý → Context đã có điều kiện bảo hiểm → Query đủ 20 ngày → Lỗi completeness ở generation; cần kiểm tra scoring faithfulness.
- **Root cause / nhận định:** Chính sách nghi_phep_khong_luong.md nêu nghỉ 16–30 ngày cần CEO phê duyệt, đồng thời nghỉ trên 14 ngày phải tự đóng phần bảo hiểm. Output trả lời đúng câu hỏi trực tiếp nhưng bỏ lưu ý mà đáp án chuẩn có. Điểm faithfulness 0,5 không đủ chứng minh câu CEO là sai; đây còn là sự khác biệt về độ đầy đủ giữa output và expected.
- **Suggested fix:** Trả lời CEO phê duyệt, kèm lưu ý nghỉ 20 ngày vượt ngưỡng 14 ngày nên phải tự đóng phần bảo hiểm. Đánh giá riêng correctness và completeness để không gán mọi thiếu ý thành hallucination.

## 4. Case study: câu Senior có 9 năm thâm niên

**Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

Ở lượt trước, top 3 đều là chính sách phép; LLM trả lời được số ngày phép nhưng không có bảng lương. Cây chẩn đoán: output thiếu mức lương → context thiếu nguồn lương → query gồm hai ý → cần sửa retrieval trước khi sửa prompt. Luồng mới tìm thêm từng vế, rerank văn bản gốc và chọn parent khác nhau. Kiểm tra truy xuất ở `reports/retrieval_recall_check.json` cho thấy context chứa chính sách phép 2024, bảng lương 2024 và bản phép 2023 làm tham chiếu. Căn cứ hiện hành cho kết quả là 15 + 9/3 = 18 ngày và Senior 20–35 triệu VNĐ/tháng. Bản cũ không được dùng thay thế quy định hiện hành.

Một thay đổi khác từng tăng precision nhưng giảm recall: loại hoàn toàn bản cũ. Đáp án chuẩn của câu về thâm niên và mật khẩu còn chứa thông tin phiên bản cũ nên ngữ cảnh thiếu phần đối chiếu. Giải pháp là ưu tiên bản hiện hành, giữ bản cũ liên quan trong ngân sách ngữ cảnh; không đưa tất cả phiên bản vào mọi câu hỏi.

## 5. Nếu có thêm một giờ

1. 20 phút: đối chiếu từng claim của năm câu trên với nguồn, đặc biệt quy tắc pro-rata; thống nhất yêu cầu đầy đủ của bộ đáp án với chủ tài liệu.
2. 20 phút: bổ sung test phép tính, citation và câu nhiều ý; đánh giá một tập holdout chưa dùng để debug để tránh tối ưu quá mức cho 20 câu hiện tại.
3. 20 phút: xem báo cáo latency, thử giảm số ứng viên hoặc model ONNX/GPU; so sánh lại precision/recall và latency cùng lúc. Không chỉ tối ưu một metric.

## 6. Giới hạn

RAGAS là bộ chấm tự động, có thể cho điểm khác nhau giữa các lượt. Các câu số học đúng vẫn có faithfulness thấp; phải xem claim-level judgement trước khi quy tất cả thành hallucination. Tài liệu OCR còn cần rà soát dấu tiếng Việt và bảng biểu. Top-k=3 là cấu hình của bài lab, chưa chắc tối ưu cho mọi loại câu hỏi. Không đạt mục tiêu rerank dưới 150 ms trên CPU trong benchmark đã lưu.
