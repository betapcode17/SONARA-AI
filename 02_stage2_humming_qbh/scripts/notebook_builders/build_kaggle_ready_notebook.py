"""Wrap the authoritative V5 train notebook for Kaggle; preserve its protocol."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
SOURCE = ROOT / 'templates' / 'conformer_bytehum_qbh_final_train.ipynb'
TARGET = ROOT / 'kaggle' / 'train' / 'conformer_bytehum_qbh_final_train_kaggle.ipynb'

INTRO = """# Conformer-ByteHum Final V5: Train trên Kaggle T4

Tải **notebook này** lên Kaggle. Toàn bộ model, tiền xử lý, loss, mining, train
và validation nằm trong notebook; không cần các file .py của dự án.
Thuật toán và cấu hình V5 được giữ nguyên, không thêm ablation mới.

## Trước khi Run All

1. Settings: Accelerator **GPU T4**, Internet **On**. T4 x2 chỉ sử dụng GPU 0.
2. Inputs: gắn **datasets.zip** (có dataset_split và 2_music_database) cùng
   **coverhunter_shs100k_pretrained.pth**. Dataset được giải nén tự động nếu là
   ZIP/.bin/.dat; có thể gắn trực tiếp thư mục dataset đã giải nén.
3. Giữ các đường dẫn override là None nếu mỗi input chỉ có một bản. Nếu có nhiều
   bản, điền đúng đường dẫn /kaggle/input ở cell cấu hình bên dưới.
4. Run All. GPU preflight kiểm tra một lần cập nhật AdamW với AMP trước khi tạo
   toàn bộ CQT cache bằng CPU. Trạng thái thử nghiệm được khôi phục sau kiểm tra.

Output: /kaggle/working/qbh_final_v5/train/ và một ZIP chỉ chứa training outputs
tại /kaggle/working. Lưu phiên bản Kaggle cùng outputs. Best checkpoint được chọn
bằng validation; không dùng humming test để chọn model.
Sau đó dùng conformer_bytehum_qbh_final_evaluation.ipynb cho đánh giá test đã khóa.

## Khi cần resume

Gắn qbh_final_v5_last.pth của chính phiên bản code này (checkpoint schema 2), đặt
RESUME_LAST_OVERRIDE tới đúng đường dẫn input; BEST được khôi phục từ LAST.
giữ RESUME=True. Epoch đang dang dở chạy lại; optimizer, scaler, scheduler,
RNG, patience và history được khôi phục. Không đổi cấu hình giữa chừng.
Checkpoint từ notebook cũ phải dùng code cũ; với bản này hãy tạo run mới từ Stage 1.
Luôn dùng notebook evaluation được phát hành cùng bản train này.

Notebook đã được kiểm tra cấu trúc và regression có phạm vi. Khi tạo file này,
chưa có bằng chứng một đợt train thực tế trên Kaggle GPU đã hoàn thành.
"""

BOOTSTRAP = '''def prepare_kaggle_environment(input_root, output_root):
    from pathlib import Path
    import os
    import sys
    import importlib.util
    import subprocess

    if not Path(input_root).is_dir():
        raise RuntimeError('Upload this notebook to Kaggle and attach dataset/model Inputs first.')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if importlib.util.find_spec('torch') is None:
        raise RuntimeError('PyTorch is missing. Select the standard Kaggle GPU image; do not install CPU-only torch.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Kaggle Settings > Accelerator > GPU T4, then restart the session.')
    if not hasattr(torch, 'amp') or not hasattr(torch.amp, 'GradScaler'):
        raise RuntimeError('This notebook requires a Kaggle PyTorch image with torch.amp.GradScaler.')

    dependencies = {'numpy': 'numpy', 'pandas': 'pandas', 'matplotlib': 'matplotlib',
                    'tqdm': 'tqdm', 'IPython': 'ipython', 'scipy': 'scipy',
                    'soundfile': 'soundfile', 'librosa': 'librosa'}
    missing = [package for module, package in dependencies.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print('Install missing dependencies:', missing)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
        except subprocess.CalledProcessError as error:
            raise RuntimeError('Dependency installation failed. Enable Internet in Kaggle Settings and inspect pip output.') from error

    for env, folder in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib')]:
        directory = Path(output_root) / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[env] = str(directory)
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
    print('Input root:', input_root)
    print('Training output:', str(Path(output_root) / 'train'))
    print('typeguard compatibility is checked in the next cell; Internet may be needed for typeguard==2.13.3.')

prepare_kaggle_environment(INPUT_ROOT, OUTPUT_ROOT)
'''

EXPORT = '''def package_training_outputs(training_dir, working_dir):
    from pathlib import Path
    import hashlib
    import json
    import os
    import zipfile

    training_dir, working_dir = Path(training_dir), Path(working_dir)
    required = ['qbh_final_v5_best.pth', 'qbh_final_v5_last.pth', 'final_manifest.json']
    for name in required:
        if not (training_dir / name).is_file():
            raise FileNotFoundError('Training did not finish or output is missing: ' + name)
    receipt = json.loads((training_dir / 'final_manifest.json').read_text(encoding='utf-8'))
    h = hashlib.sha256()
    with (training_dir / 'qbh_final_v5_best.pth').open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    if receipt.get('checkpoint_sha256') != h.hexdigest():
        raise ValueError('Best checkpoint does not match the completed-training manifest.')
    names = required + ['run_contract.json', 'history.json', 'best_validation_metrics.json', 'training_curves.png']
    archive = working_dir / 'qbh_final_v5_train_outputs.zip'
    temp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for name in names:
            source = training_dir / name
            if source.is_file():
                zipped.write(source, arcname='train/' + name)
    os.replace(temp, archive)
    return archive

training_archive = package_training_outputs(TRAIN_OUTPUT, Path(OUTPUT_ROOT).parent)
print('Training finished. Download checkpoints from Kaggle Output:')
print(TRAIN_OUTPUT / BEST_NAME)
print(TRAIN_OUTPUT / LAST_NAME)
print('Checkpoint-only archive (dataset and CQT cache are excluded):', training_archive)
'''


def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=source.splitlines(True))
    if kind == 'code':
        item.update(execution_count=None, outputs=[])
    return item


def build():
    source_bytes = SOURCE.read_bytes()
    original = json.loads(source_bytes.decode('utf-8'))
    codes = [copy.deepcopy(c) for c in original['cells'] if c['cell_type'] == 'code']
    if not codes or 'SOURCE_CODE_DIGEST' not in ''.join(codes[0]['source']):
        raise ValueError('Expected the V5 configuration as the first code cell.')
    cells = [cell('markdown', INTRO), codes[0],
             cell('markdown', '## Kaggle environment check\n'), cell('code', BOOTSTRAP)]
    cells.extend(codes[1:])
    cells.extend([cell('markdown', '## Download training outputs\n'), cell('code', EXPORT)])
    for index, item in enumerate(cells):
        item['id'] = f'kaggle-ready-{index:03d}'
        if item['cell_type'] == 'code':
            item.update(execution_count=None, outputs=[])
            compile(''.join(item['source']), f'{TARGET.name}:cell{index + 1}', 'exec')
    document = copy.deepcopy(original)
    document['cells'] = cells
    document.setdefault('metadata', {})['qbh_kaggle_ready'] = dict(
        source_notebook=SOURCE.name, source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        profile='Kaggle T4; one GPU', gpu_training_verified_at_creation=False,
        original_v5_code_preserved=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(TARGET)


if __name__ == '__main__':
    build()
