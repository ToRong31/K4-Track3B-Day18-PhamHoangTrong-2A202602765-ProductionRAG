# Latency — Lab 18

**Học viên:** PhamHoangTrong — **MSSV:** 2A202602765

## Phạm vi bằng chứng

Lượt `main.py` cuối cùng do học viên chạy báo tổng **645,7 giây**, gồm baseline, production, indexing, generation và RAGAS. Đây không phải độ trễ trả lời một câu hỏi. Lượt này chưa lưu timer riêng cho từng bước, nên không chia tổng thời gian bằng ước lượng.

Benchmark riêng đã lưu tại `reports/reranker_benchmark.json`: `BAAI/bge-reranker-v2-m3`, CPU, 20 candidates, 3 lượt sau warmup. Kết quả này là benchmark trước đó; không phải phép đo toàn pipeline của lượt cuối.

| Phần | Thời gian có bằng chứng | Phạm vi / trạng thái |
|---|---:|---|
| Tổng lượt lab cuối | 645,7 s | Terminal học viên cung cấp; bao gồm cả baseline và đánh giá |
| M3 rerank trung bình | 3206,18 ms | Benchmark CPU riêng, 20 candidates, 3 runs sau warmup |
| M3 rerank nhanh nhất | 3017,84 ms | Cùng benchmark |
| M3 rerank chậm nhất | 3353,29 ms | Cùng benchmark |
| M1 load/chunk | Chưa đo riêng | Không suy ra từ tổng thời gian |
| M5 enrichment/cache | Chưa đo riêng | Cần tách cache hit và cache miss |
| M2 embedding/index | Chưa đo riêng ở lượt cuối | Cần tách indexing khỏi query-time dense search |
| M2 BM25/dense/RRF khi truy vấn | Chưa đo riêng | Không báo một số giả định |
| LLM generation | Chưa đo riêng | Cần timer từng câu và số token |
| M4 RAGAS | Chưa đo riêng ở lượt cuối | Không lấy thời gian của lượt cũ làm thời gian mới |

Benchmark rerank vượt mục tiêu 150 ms khoảng **21,37 lần**. Không có bằng chứng Flashrank đạt dưới 5 ms trên máy này; lớp tùy chọn đã được hoàn thiện và kiểm tra bằng mock, chưa benchmark mô hình ONNX thật.

## Kiểm tra bổ sung và giới hạn

Đã thử đo retrieval trên index có sẵn, kiểm tra index khớp trước khi chạy. Phép đo dừng trước khi chạy mô hình vì không truy cập được Qdrant và client chuyển sang in-memory trống. Kiểm tra trực tiếp ghi `qdrant_client.http.exceptions.ResponseHandlingException: timed out`; Docker báo không tìm thấy pipe `dockerDesktopLinuxEngine`. Vì vậy không có báo cáo mới về latency retrieval thành công. Việc dừng này không thay đổi báo cáo RAGAS cuối đã lưu.

## Kế hoạch đo tiếp

1. Khi Qdrant sẵn sàng, xác minh index khớp corpus; warmup encoder và reranker trước query benchmark.
2. Dùng `time.perf_counter()` quanh BM25, dense query, RRF, rerank, context assembly và generation; ghi JSON theo từng câu.
3. Báo p50/p95 trên ít nhất 30 query, tách câu một ý và nhiều ý. Không cộng các số của hai lượt khác nhau.
4. Đo cold start và indexing riêng. So CPU/GPU/ONNX cùng chất lượng retrieval; chỉ giữ cấu hình đạt yêu cầu cả latency và accuracy.

**Đối chiếu rubric:** có số đo rerank thật và bảng phạm vi từng bước, nhưng chưa có đầy đủ thời gian từng bước ở lượt cuối. Không tự khẳng định đạt trọn bonus latency breakdown.
