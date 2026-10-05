# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Phạm Hoàng Trọng
**MSSV:** 2A202602765
**Khóa:** K4 — Track 3B
**Ngày hoàn thiện:** 05/10/2026

## Phần 1. Mapping bài giảng vào code

| Lecture concept | Module | Hàm cụ thể | Quan sát và phân tích từ bài làm |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Regex tách câu; all-MiniLM-L6-v2 mã hóa; cosine của câu liền kề dưới 0,85 thì ngắt nhóm. Threshold là tham số cần đánh giá theo corpus, không mặc định coi 0,85 tối ưu cho mọi văn bản tiếng Việt. Chưa đo số semantic chunks trên toàn corpus ở lượt cuối. |
| Hierarchical và structure-aware | M1 | `chunk_hierarchical()`, `chunk_structure_aware()`, `prepare_chunks()` | Cha tối đa 2048, con tối đa 256 ký tự, liên kết bằng parent_id duy nhất theo tài liệu. Kho hiện tại tạo 64 parents và 493 children; baseline có 210 paragraph chunks. Structure-aware xử lý #/##/### và tránh nhầm heading trong code fence; pipeline hiện chọn hierarchical, không khẳng định chạy cả ba chiến lược cùng lúc. |
| BM25 + dense, RRF | M2 | `segment_vietnamese()`, `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | Tokenization áp dụng cùng cách cho corpus/query; theo đề, dấu gạch dưới được đổi thành khoảng trắng. Dense dùng bge-m3 1024 chiều, Qdrant cosine và query_points. RRF cộng 1/(60+rank+1), không cộng trực tiếp hai thang score khác nhau. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker._load_model()`, `.rerank()`, `retrieve_contexts()` | bge-reranker-v2-m3 chấm cặp query/văn bản gốc, batch=4 và max_length=1024. M5 vẫn hỗ trợ retrieval nhưng câu sinh giả định không được dùng như chứng cứ. Chọn tối đa ba parent khác nhau. Benchmark CPU cũ: trung bình 3206,18 ms cho 20 candidates; chưa đạt 150 ms. |
| RAGAS và Diagnostic Tree | M4 | `evaluate_ragas()`, `failure_analysis()`, `save_report()` | Điểm cuối: faithfulness 0,8305; relevancy 0,8830; precision 0,9917; recall 0,9583. Faithfulness thấp nhất; bottom-5 tập trung vào phép tính, giả định và gán nguồn. Metric thấp là tín hiệu điều tra, không phải kết luận nguyên nhân chắc chắn. |
| Contextual prepend, HyQA, combined enrichment | M5 | `_enrich_single_call()`, `enrich_chunks()`, `contextual_prepend()`, `enrich_children()` | Một prompt trả JSON summary/questions/context/metadata; cache theo input để tránh gọi lại. Enrichment trên child trước index; LLM trả lời nhận parent gốc. Có fallback khi thiếu key/lỗi API. Chưa có ablation riêng chứng minh phần tăng điểm nào chỉ do M5. |

Kết quả baseline và production là cùng một lượt đánh giá cuối, không chọn điểm tốt nhất từ nhiều lượt:

| Metric | Baseline | Production | Δ |
|---|---:|---:|---:|
| `faithfulness` | 0.8042 | 0.8305 | +0.0263 |
| `answer_relevancy` | 0.7091 | 0.8830 | +0.1739 |
| `context_precision` | 0.9250 | 0.9917 | +0.0667 |
| `context_recall` | 0.9250 | 0.9583 | +0.0333 |

Bài học quan trọng là cải tiến có thể đánh đổi precision và recall. Loại bản cũ giúp xếp bản mới đúng hơn, nhưng làm thiếu phần lịch sử trong một số đáp án chuẩn. Phương án cuối giữ thứ tự hiện hành trước, tham chiếu cũ sau và bổ sung nguồn cho từng ý của câu hỏi. Không làm yếu baseline để chứng minh production tốt hơn.

## Phần 2. Khó khăn và cách giải quyết

### 2.1. PDF scan không có text layer

Hai PDF BCTC.pdf (2 trang) và Nghi_dinh_so_13-2023_ve_bao_ve_du_lieu_ca_nhan_508ee.pdf (39 trang) trả 0 ký tự khi trích text; so_tay_an_toan.pdf đọc được bình thường. Đây là tình trạng dữ liệu, không phải lỗi Dense Search. Thông báo của loader khi không có OCR là:

```text
Bỏ qua BCTC.pdf: PDF scan ảnh, không có text layer (cần OCR).
```

Cách debug: đếm trang và ký tự trích xuất, kiểm tra ảnh render, OCR rồi đọc lại nội dung. Bản .ocr.md giữ source là tên PDF và tránh nạp trùng PDF/bản OCR. BCTC.pdf thực tế là tờ khai thuế GTGT; phải tin nội dung kiểm tra trực tiếp thay vì suy đoán theo tên file. OCR không đồng nghĩa mọi số liệu và dấu tiếng Việt đều chính xác, cần rà soát trước khi dùng nghiệp vụ.

### 2.2. Lỗi tài nguyên trên máy và chạy dài

Log cũ `reports/main_run.log` ghi đúng lỗi:

```text
memory allocation of 1312 bytes failed
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

Quá trình debug: kiểm tra tiến trình dừng ở giai đoạn nạp/embedding; tránh chạy thêm nhiều mô hình và test nặng đồng thời; giảm batch reranker xuống 4 và giới hạn input 1024 tokens; lưu câu trả lời sau từng câu và cache enrichment để có thể tiếp tục mà không gọi lại toàn bộ API. Log chỉ chứng minh sự cố cấp phát, chưa đủ kết luận chính xác nguyên nhân RAM, pagefile hay thư viện. Lượt chạy cuối do tôi thực hiện đã hoàn thành và terminal báo tổng 645,7 giây; không khẳng định đây là latency cho một câu hỏi.

### 2.3. Sai vị trí bản cũ, thiếu ý trong câu nhiều vế

Triệu chứng chính xác ở các lượt trước: bản mật khẩu cũ đứng trước bản hiện hành; câu Senior chỉ có tài liệu phép mà thiếu lương; sau khi loại bản cũ, recall giảm xuống 0,8583 dù precision lên 0,9750. Đây là lỗi chất lượng, không có exception message.

Cách debug: so báo cáo từng câu, xem raw contexts và ngày hiệu lực, kiểm tra child→parent, tách retrieval lỗi khỏi generation lỗi. Sửa bằng rerank văn bản gốc kèm tiêu đề, chọn parent khác nhau, ưu tiên phiên bản hiện hành nhưng giữ lịch sử liên quan; câu nhiều ý có bổ sung search theo vế. Kiểm tra thật cho câu Senior lấy được bảng lương 20–35 triệu/tháng. Kích thước child/parent vẫn giữ 256/2048; baseline không đổi.

### 2.4. Metric thấp không luôn đồng nghĩa output sai

Câu Junior tính 20 triệu × 85% = 17 triệu đúng nguồn và đáp án, nhưng faithfulness chỉ 0,5. Câu tạm ứng gần đáp án chuẩn nhưng dùng giả định tháng 30 ngày mà chính sách không ghi; faithfulness 0,1429 là cảnh báo có căn cứ. Cần học thêm cách RAGAS phân rã claims, phân biệt correctness/completeness/faithfulness và kiểm tra phép tính bằng test xác định. Không chỉ giảm temperature: bài hiện không đặt temperature=0 khi gọi generation/enrichment theo cấu hình cuối.

## Phần 3. Action plan cho project cá nhân

### Project đề xuất: Trợ lý tra cứu chính sách nội bộ tiếng Việt

Đây là hướng mở rộng từ repo lab, chưa phải một hệ thống production đã triển khai. Hiện có corpus chính sách và PDF, pipeline năm module, Qdrant và bộ đánh giá 20 câu. Bottlenecks là latency CPU, chất lượng OCR, câu cần tính toán/ngoại lệ, quản lý phiên bản và giới hạn của bộ test nhỏ.

### Kế hoạch áp dụng

1. **Chunking:** dùng hierarchical cho truy xuất chính xác và context cha; thử structure-aware cho bảng/danh sách. Lưu source, ngày hiệu lực, trạng thái thay thế, quyền truy cập. Đo ablation trước khi đổi kích thước.
2. **Search:** giữ BM25 + bge-m3 + RRF; bổ sung bộ query nhiều ý và lỗi chính tả. Đánh giá giữ từ ghép tiếng Việt thay vì chỉ tuân theo normalization của lab; corpus và query luôn cùng tokenizer.
3. **Reranking:** so bge-reranker-v2-m3 trên GPU với ONNX/Flashrank trên CPU; đo p50/p95 và độ chính xác top-k. Không mặc định hứa dưới 5 ms hoặc dưới 150 ms khi chưa benchmark trên phần cứng đích.
4. **Evaluation:** tăng lên ít nhất 60 câu, chia dev/holdout; thêm version, negation, numeric, multi-hop, không có đáp án. RAGAS kết hợp review thủ công và test phép tính/citation; mục tiêu holdout faithfulness ≥0,85, precision/recall ≥0,90 qua nhiều lượt.
5. **Enrichment:** combined single-call, cache và incremental update chỉ cho chunk thay đổi; đo chi phí API và ablation bật/tắt M5. Chứng cứ trả lời luôn là tài liệu gốc.

### Timeline đề xuất

| Thời gian | Công việc | Kết quả kiểm chứng |
|---|---|---|
| Tuần 1 (05–11/10/2026) | Rà OCR, chuẩn hóa version/source; bổ sung ít nhất 40 câu, chia dev/holdout | Corpus có nguồn và phiên bản; bộ 60 câu có đáp án được đối chiếu tài liệu |
| Tuần 2 (12–18/10/2026) | Ablation chunking/hybrid/M5, đánh giá multi-intent; benchmark CPU/GPU/ONNX | Bảng metric + p50/p95 + chi phí, chọn cấu hình dựa trên holdout |
| Tuần 3 (19–25/10/2026) | Citation theo claim, kiểm tra phép tính, xử lý không có đáp án và quyền truy cập | Test hồi quy và kiểm tra tài liệu ngoài quyền không bị trả ra |
| Tuần 4 (26/10–01/11/2026) | Pilot nhỏ, quan sát failure log, incremental indexing, kiểm tra tải | Báo cáo pilot, rollback index và kế hoạch vận hành |

### Tiêu chí hoàn thành

Một thay đổi chỉ được giữ khi test hồi quy pass và có kết quả trên holdout. Cần công bố cả trường hợp chưa đạt, gồm latency và câu có quy ước chưa được chính sách xác nhận. Benchmark mới chỉ đo retrieval được trình bày riêng tại `analysis/latency_breakdown.md`; không đánh đồng thời gian cả lab với độ trễ phục vụ một người dùng.

### Kiểm tra cuối bài

`python check_lab.py` chạy bằng môi trường `.venv` đã kiểm tra đủ source/report/reflection và báo **110/110 tests passed (100%)**, **0 TODO**. Log nằm ở `reports/submission_check.log`. Chỉ tăng timeout chờ test từ 120 lên 300 giây vì bộ CPU test đã từng mất 125 giây; không đổi ngưỡng pass hay bộ test để làm đẹp kết quả.
