# QbH Final V5: train một đợt, đánh giá test một lần

## Phạm vi

Đây là pipeline cuối được chốt cho **Kaggle GPU T4**, không phải cam kết model đạt độ chính xác hoàn hảo. Hai notebook độc lập chứa toàn bộ code, bao gồm backbone CoverHunter. Không cần tải các file Python của dự án lên Kaggle.

- Train Kaggle: [conformer_bytehum_qbh_final_train_kaggle.ipynb](../notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb).
- Notebook nguồn: [conformer_bytehum_qbh_final_train.ipynb](../notebooks/templates/conformer_bytehum_qbh_final_train.ipynb).
- Test: [conformer_bytehum_qbh_final_evaluation.ipynb](../notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb).
- Bản cũ, dữ liệu và checkpoint v4 được giữ nguyên. V5 bắt đầu từ SHS100K Stage 1, không tiếp tục các checkpoint v4 đã qua nhiều thử nghiệm.

## Cấu hình đã chốt

| Thành phần | Cấu hình |
| --- | --- |
| Backbone | CoverHunter Conformer pretrained SHS100K, embedding 128-D |
| Freeze | Frontend và 3 block đầu; khóa CMVN và chế độ train của phần đã freeze |
| Huấn luyện | 8 bài x 2 query/microbatch, tích lũy 4, FP16 AMP, tối đa 30 epoch |
| Optimizer | AdamW; backbone 1e-4, auxiliary heads 3e-4, weight decay 0.01 |
| Loss | Global triplet + LAT + LAC + domain BCE/GRL; prototype diversity 0.05 |
| Mining | Semi-hard rồi tăng tỷ trọng hard; top-5, memory FIFO 4096 vector global, loại cùng song ID |
| Positive | Đúng fragment; vocal ratio 0.80 giữ từ cấu hình cũ; chọn chunk có fused similarity cao nhất trong toàn bộ reference fragment, sau đó forward lại có gradient |
| Augmentation | Nonlinear time-grid warp 0.7–1.3; pitch ±12 bin; time/frequency mask điền mức dB nhỏ nhất |
| Audio/CQT | Toàn bộ file; 16 kHz, cửa sổ 10 s/hop 3 s; thêm cửa sổ cuối nếu còn mẫu chưa phủ; 96 x 83, floor -80 dB |
| Cache | Disk theo SHA256 nội dung audio + code + cấu hình + runtime; RAM LRU 512 MB |
| Scoring | 0.5 x query-directed local MaxMean + 0.5 x max cosine global, chấm toàn gallery theo block |
| Chọn checkpoint | Micro fused MRR validation; strict improvement; early stopping patience 5 |
| Validation | Toàn bộ 773 query/45 bài val; gallery 255 full recordings thuộc train+val |
| Test | Toàn bộ 774 query/45 bài test; gallery 300 full recordings |

128-D và scoring trên được giữ để tránh đưa thêm projection/index chưa được chứng minh vào một đợt train duy nhất. Không dùng FAISS candidate gating vì gate có thể loại đáp án trước reranking. Không tự chọn trọng số DTW, threshold hoặc augmentation bằng kết quả test.

Chọn positive tốt nhất là multiple-instance selection **không có gradient ở bước chọn**; gradient đi qua forward lại của chunk đã chọn. Không phải soft-DTW hoặc mô hình căn chỉnh thứ tự. Local MaxMean vẫn có giới hạn về thứ tự thời gian. Thay đổi chọn positive này cần được đo bằng kết quả V5, chưa có bằng chứng rằng nó nâng MRR.

## Chạy trên Kaggle

1. Upload notebook train. Bật **GPU T4** và Internet nếu kernel cần cài `typeguard==2.13.3`. Attach dataset có `dataset_split/{train,val,test}` và `2_music_database`, cùng `coverhunter_shs100k_pretrained.pth` từ nguồn tin cậy. Dataset có thể đã giải nén hoặc là một ZIP chứa cả hai thư mục (hỗ trợ ZIP đổi đuôi `.bin`/`.dat`); notebook kiểm tra đường dẫn archive an toàn và dung lượng đĩa trước khi giải nén. Nếu có nhiều bản dữ liệu/checkpoint trong Inputs, đặt đường dẫn chính xác ở cell cấu hình đầu, gồm `DATASET_ARCHIVE_OVERRIDE` nếu cần; notebook sẽ không tự chọn một bản ngẫu nhiên.
2. Run All notebook train. GPU preflight chạy trước cache toàn bộ audio: train mode, loss thực tế, AMP accumulation, backward, một bước AdamW và identity scoring; sau đó rollback weights/buffers/RNG. Optimizer/scaler/scheduler thật được giữ nguyên. Tiếp theo cache CQT toàn bộ audio train/val cần dùng để phát hiện file hỏng/silence. Kết thúc khi đạt 30 epoch hoặc early stopping. Lưu Kaggle output, đặc biệt thư mục `qbh_final_v5/train/` gồm `qbh_final_v5_best.pth`, `qbh_final_v5_last.pth`, `final_manifest.json`, `run_contract.json`, `history.json`, biểu đồ và best-validation metrics. **Notebook train không decode/sử dụng audio humming test để huấn luyện hoặc chọn checkpoint**; nếu dùng ZIP, việc giải nén đương nhiên bao gồm copy các file test.
3. Upload notebook evaluation. Attach đúng output train và cùng dataset, cùng phiên bản PyTorch/audio libraries. Set `CHECKPOINT_OVERRIDE` nếu discovery báo có nhiều best checkpoint. Run All một lần; giữ toàn bộ thư mục output `qbh_final_v5/test/`.

Không chỉnh `RUN_CONFIG` giữa hai notebook. Phiên bản code, preprocessing, content hash train/val và runtime phải khớp. Nếu cần sửa nghiên cứu, tạo run mới; đó không còn là cùng một đợt train/test đã chốt.

Nếu Kaggle báo không có CUDA, hãy kiểm tra Accelerator trước khi chạy; pipeline không tự đổi sang CPU. T4 x2 vẫn chỉ dùng GPU 0, không có DDP. Chọn positive nhiều chunk tăng thời gian inference trong training; thời gian tổng cần đo trên T4. Không tự giảm batch sau OOM vì sẽ thay đổi mining/protocol của run.

## Khi bị ngắt

- Train resume ở ranh giới epoch: attach **LAST schema 2 từ cùng phiên bản code**, đặt `RESUME_LAST_OVERRIDE`, giữ `RESUME=True`. LAST chứa cả snapshot BEST, khôi phục optimizer, scheduler, scaler, toàn bộ RNG, DataLoader generator, history và patience. Ghi LAST atomically trước BEST/reports; có thể dựng lại BEST bị thiếu/cũ sau khi ngắt. Epoch đang dang dở chạy lại; BEST riêng lẻ bị từ chối resume. Để `RESUME_BEST_OVERRIDE=None`.
- Test resume: attach output evaluation cũ. Notebook tự tìm thư mục chứa `test_lock.json` của V5 và khôi phục state vào output writable; nếu có nhiều bản, đặt `EVAL_RESUME_DIR` tới đúng thư mục. `test_progress.json` là prefix query đã hoàn thành, được lưu atomically sau mỗi query. Với cùng checkpoint/dataset/protocol, chỉ chạy phần chưa hoàn thành.
- Nếu `test_metrics.json` đã tồn tại và đúng signature, notebook hiển thị báo cáo cũ, không chạy model lại. Cơ chế khóa chỉ có hiệu lực khi giữ lại output; một session Kaggle mới với thư mục trống không thể tự biết test đã chạy ở session trước. Không xóa lock/report để chỉnh model theo test.

Checkpoint sử dụng `torch.load(weights_only=False)` để khôi phục RNG/optimizer: chỉ attach checkpoint của chính bạn hoặc nguồn tin cậy, không mở pickle không rõ nguồn gốc.

Bản reliability thay đổi code digest/schema: checkpoint V5 cũ phải dùng notebook cũ.
Chạy bản sửa từ Stage 1 trong `OUTPUT_ROOT` mới và dùng evaluator cùng bản phát hành.
Snapshot BEST nhúng trong LAST làm tăng dung lượng LAST khoảng một bộ trọng số model.
Reference feature được tái sử dụng cho đường dẫn trùng trong cùng batch; không cache
feature model qua optimizer update. Loss, mining, vocal ratio và scoring không đổi.
Xem [V5_CODE_IMPROVEMENTS.md](V5_CODE_IMPROVEMENTS.md).

## Báo cáo cuối

`test_metrics.json` và `test_queries.csv` lưu checkpoint hash, best epoch, số query/gallery, MRR và Recall@1/5/10/20 của local/global/fused, fused song-macro MRR, khoảng tin cậy 95% bootstrap theo bài hát, thời gian query p50/p95. Mọi query phải có ground truth trong gallery và score hữu hạn; lỗi không được bỏ qua để làm đẹp mẫu số.

Tie convention: điểm gần nhau theo rtol=1e-6/atol=1e-7 dùng average rank; MRR là nghịch đảo average rank, không phải expected reciprocal rank của random tie-breaking. Recall@K dùng expected hit theo tie group. Thời gian query bao gồm cache lookup/CQT nếu cache miss, encoding và exact scoring; không bao gồm xây gallery. `wall_seconds_this_session` chỉ là thời gian session hiện tại nếu có resume.

**Giới hạn cần nêu trong báo cáo khoa học:** test split hiện tại đã được đo trong các thí nghiệm trước, nên chạy V5 một lần không biến nó thành một tập blind test mới. Muốn kết luận tổng quát độc lập cần một tập mới hoặc benchmark ngoài dự án. Pipeline này là closed-set; chưa có open-set calibration, MIR-QBSH/IOACAS/MTG-QBH hay bằng chứng vượt baseline. Không công bố các phần đó như kết quả đã hoàn thành.

## Kiểm chứng trước khi bàn giao

Các kiểm tra trong `tests/test_final_solution.py` bao phủ compile/JSON structure của hai notebook, code tự chứa và common code khớp, cắt audio không bỏ đuôi/trùng offset, tie handling, micro/macro denominator, atomic JSON và khóa/resume, split/gallery thực tế. Có test PyTorch riêng cho scorer theo block và gradient triplet; chúng bị skip nếu runtime kiểm tra không có PyTorch. Kiểm tra tĩnh/NumPy không thay thế việc chạy GPU preflight, huấn luyện thật và benchmark thật trên Kaggle.
