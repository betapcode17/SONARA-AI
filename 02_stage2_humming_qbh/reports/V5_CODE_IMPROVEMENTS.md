# V5: sua do tin cay cua training

## Thay doi

1. **Checkpoint nhat quan khi ngat.** LAST schema 2 chua training state va snapshot
   BEST (weights, epoch, validation metrics). LAST la commit atomically duy nhat;
   BEST/history/report duoc xuat sau. Khi resume, doi chieu snapshot voi history va
   contract, dung lai BEST neu thieu/cu. Khong dung BEST rieng le de bat dau lai
   training ngam. Xuat lai BEST giong nhau khong ghi lai bytes, giu checkpoint hash.
2. **Preflight truoc full CQT cache.** Chay train mode voi mot nhom accumulation,
   cac loss/curriculum thuc te, AMP backward, clip va mot buoc AdamW. Dung optimizer,
   scaler, memory bank thu rieng; rollback weights, BN buffers, GRL lambda, che do
   model va RNG ke ca DataLoader generator trong finally. Kiem tra scorer identity.
   Day la smoke test; khong thay the train/validation tren GPU that.
3. **Chon positive it forward lap.** Nhom reference trung duong dan trong mot batch,
   encode moi bag mot lan theo block, cham rieng tung anchor va giu thu tu query.
   Dong diem van lay chunk dau tien. Khong giu feature qua lan goi/optimizer update.
   Chunk da chon van duoc forward lai co gradient nhu truoc.

## Khong thay doi

- Backbone, so block freeze, augmentation, vocal ratio 0.80, loss weights/mining,
  batch/accumulation, schedule, patience va cong thuc fused scoring.
- Validation 773 query voi gallery 255 bai train+val; test 774 query/gallery 300 bai.
- Khong train them, khong doc/test checkpoint de do accuracy trong lan sua code nay.
- Khong sua dataset, checkpoint, runs va output notebook lich su.

## Cach su dung

Upload `notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb`,
gan dataset va Stage-1 checkpoint nhu truoc, GPU T4/Internet, Run All.
Dung `notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb`
duoc sinh cung ban code nay khi train xong.

Code digest da doi: khong resume LAST cu vao code moi. Neu da co output cu,
chon OUTPUT_ROOT trong va train tu Stage 1; khong xoa run cu. Voi LAST schema 2
moi, chi can RESUME_LAST_OVERRIDE, de RESUME_BEST_OVERRIDE=None.
Epoch dang do chay lai tu dau sau epoch da commit cuoi cung.

## Kiem chung

`tests/test_final_training_reliability.py` thuc thi helper nhung trong notebook,
gia lap loi truoc/sau commit LAST, BEST cu/thieu, history/contract sai, xuat lap
khong doi bytes va so sanh chon positive voi cong thuc naive bang NumPy.
Kiem tra AST preflight va test Torch CPU rollback rieng (skip neu thieu Torch).
Fake checkpoint IO/NumPy khong kiem chung serialization Torch hay CUDA.
Bo test archive `test_notebook_research.py` cung chua chay duoc trong runtime nay:
import dung voi `ModuleNotFoundError: No module named 'torch'`.

So test, hash notebook va cac test skip duoc ghi vao
`final_solution_validation.json` va `kaggle_ready_validation.json`.
Chua co ket qua train GPU moi hay bang chung MRR/Recall tang sau ban sua nay.
Ban goc truoc sua duoc luu trong `backups/pre_training_reliability_*.zip`.
