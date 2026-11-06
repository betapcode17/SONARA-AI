#!/usr/bin/env python3
"""Create a sanitized QBH V6 dataset archive and training notebook from V5 inputs.

The original archive is read-only. Outputs are written beside it with a v6 suffix.
Missing source query IDs are retained in an audit field and excluded from active
training counts; they are never fabricated.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DEFAULT_ARCHIVE = DIST / "datasets_qbh_clean_mirqbsh_70_15_15.zip"
DEFAULT_NOTEBOOK = ROOT / "notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb"
OUT_ARCHIVE = DIST / "datasets_qbh_clean_mirqbsh_70_15_15_v6.zip"
OUT_CONTRACT = DIST / "qbh_dataset_contract_v6.json"
OUT_HASHES = DIST / "qbh_dataset_audio_sha256_v6.json"
OUT_NOTEBOOK = ROOT / "notebooks/kaggle/train/conformer_bytehum_qbh_final_v6_kaggle.ipynb"
SOURCE_CONTRACT = ROOT / "backups/qbh_dataset_contract_before_v6.json"
if not SOURCE_CONTRACT.exists():
    SOURCE_CONTRACT = DIST / "qbh_dataset_contract.json"
MISSING_TARGETS = {
    "12777492f4c9adb9", "1b474bae1557a0a6", "1f79db7550ca5772", "21f08f46c5dec2fa",
    "2adad3180e590a3a", "2ea10a2837697f48", "2fab930015f8d575", "342f50db0e5ebed5",
    "3bab87dd00f0facb", "451b846328464d65", "462b38e882e8cd78", "49a20d60e3e08806",
    "4b48293d9ca92cbd", "4e73f1ae5a45f82e", "548e61c546a1c145", "59b516c09115b3cf",
    "606bacde410ba841", "609716be5d522942", "6c376f8dfa4c0223", "6f73450e8574c99f",
    "76071fd87e6868ac", "7aadb74954cbd26f", "8455a49d7c5619e3", "84819d82da591aeb",
    "8cb95eb3955ac409", "9287ecbd7717673c", "a37811befd2456a1", "a9bbacd43cab1652",
    "ac0e51fd30446060", "c1c966ef839ade42", "c8f211bf97d38ddb", "ce96551498969d69",
    "dbe9902d72fb658b", "e4b143c7119adf13", "e4cdb670e61fa685", "e7da3058a6501def",
    "e8a16fee7f810f17", "e9e41f8e936d07fd", "f7ac30225541952c",
}
RENAMED_TITLE = "Eduard Artemyev - At Home Among Strangers.mp3"
RUSSIAN_TITLE = "Музыка из фильма. Свой среди чужих, чужой среди своих"


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def source_for(path: Path) -> str:
    return "original" if not path.startswith("datasets/mir_qbsh_extension/") else "MIR-QBSH"


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def make_notebook(source: Path, target: Path) -> None:
    nb = json.loads(source.read_text(encoding="utf-8"))
    cells = nb["cells"]
    for cell in cells:
        s = "".join(cell.get("source", []))
        s = s.replace("Conformer-ByteHum Final V5", "Conformer-ByteHum Final V6")
        s = s.replace("qbh_final_v5", "qbh_final_v6")
        s = s.replace("qbh_final_v5_best.pth", "qbh_final_v6_best.pth")
        s = s.replace("qbh_final_v5_last.pth", "qbh_final_v6_last.pth")
        s = s.replace("qbh_final_v5_train_outputs.zip", "qbh_final_v6_train_outputs.zip")
        cell["source"] = s.splitlines(keepends=True)

    runtime_imports = "".join(cells[4]["source"])
    if "import scipy.signal" not in runtime_imports:
        runtime_imports = runtime_imports.replace("import scipy.sparse\n", "import scipy.sparse\nimport scipy.signal\n")
    cells[4]["source"] = runtime_imports.splitlines(keepends=True)

    # Training controls and checkpoint identity.
    s = "".join(cells[5]["source"])
    s = s.replace("pipeline='qbh_final_v6', seed=42, epochs=30, patience=5,", "pipeline='qbh_final_v6', seed=42, epochs=40, patience=8,")
    s = s.replace("freeze_blocks=3, embedding_dim=128, vocal_ratio=0.80,", "freeze_blocks=6, gradual_unfreeze_epoch=6, embedding_dim=128, vocal_ratio=0.80,")
    s = s.replace("global_triplet=1.0, lac=1.0, lat=1.0, grl=0.2,", "global_triplet=1.0, lac=0.25, lat=1.0, grl=0.2,")
    s = s.replace("weight_decay=0.01, clip=1.0, amp='fp16',", "weight_decay=0.01, clip=1.0, amp='fp16',\n                  low_band_degradation_probability=0.30, degradation_sample_rate=8000,\n                  lr_unfreeze_lower=1e-5, lr_unfreeze_upper=5e-5,")
    s = s.replace("selection='fused_micro_mrr'", "selection='fused_micro_mrr'")
    cells[5]["source"] = s.splitlines(keepends=True)

    # Every training song contributes K samples exactly once per epoch.
    s = """class SongBalancedBatchSampler:
    \"\"\"Each epoch visits every song once and draws K queries for that song.\"\"\"
    def __init__(self, dataset, songs_per_batch=8, samples_per_song=2):
        self.dataset = dataset
        self.songs_per_batch = int(songs_per_batch)
        self.samples_per_song = int(samples_per_song)
        self.batch_size = self.songs_per_batch * self.samples_per_song
        self.song_to_indices = {}
        for index, (song, _, _) in enumerate(dataset.humming_samples):
            self.song_to_indices.setdefault(song, []).append(index)
        self.songs = sorted(self.song_to_indices)
        if len(self.songs) < self.songs_per_batch:
            raise ValueError(f\"Need at least {self.songs_per_batch} songs, found {len(self.songs)}\")
        self.num_batches = math.ceil(len(self.songs) / self.songs_per_batch)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        song_order = self.songs.copy()
        random.shuffle(song_order)
        for start in range(0, len(song_order), self.songs_per_batch):
            selected_songs = song_order[start:start + self.songs_per_batch]
            batch = []
            for song in selected_songs:
                indices = self.song_to_indices[song]
                if len(indices) >= self.samples_per_song:
                    batch.extend(random.sample(indices, self.samples_per_song))
                else:
                    batch.extend(random.choices(indices, k=self.samples_per_song))
            random.shuffle(batch)
            yield batch
"""
    cells[12]["source"] = s.splitlines(keepends=True)

    # Add low-band degradation before CQT extraction, cached separately.
    s = "".join(cells[14]["source"])
    s = s.replace("def raw_cqts(path):\n", "def raw_cqts(path, degrade_to_8khz=False):\n")
    s = s.replace("key = hashlib.md5(f\"{str(path)}_{stat.st_size}_{stat.st_mtime_ns}_{PREPROCESS_CONFIG}\".encode()).hexdigest()",
                  "key = hashlib.md5(f\"{str(path)}_{stat.st_size}_{stat.st_mtime_ns}_{PREPROCESS_CONFIG}_low8k={degrade_to_8khz}\".encode()).hexdigest()")
    s = s.replace("        y = prep.load_audio(str(path))\n        if np.max(np.abs(y)) <= 1e-6:",
                  "        y = prep.load_audio(str(path))\n        if degrade_to_8khz:\n            # Simulate an 8 kHz capture channel, then restore the model's 16 kHz input rate.\n            y = scipy.signal.resample_poly(y, 1, 2).astype(np.float32)\n            y = scipy.signal.resample_poly(y, 2, 1).astype(np.float32)\n        if np.max(np.abs(y)) <= 1e-6:")
    s = s.replace("        self.pitch = CQTPitchShiftAugmentation(12, 0.8)\n", "        self.pitch = CQTPitchShiftAugmentation(12, 0.8)\n")
    s = s.replace("        a = random.choice(raw_cqts(row['path'])).clone()\n", "        degrade = (not row['song'].startswith('MIR-QBSH-') and\n                   random.random() < RUN_CONFIG['low_band_degradation_probability'])\n        a = random.choice(raw_cqts(row['path'], degrade_to_8khz=degrade)).clone()\n")
    s = s.replace("                for block in list(encoder.encoders)[:RUN_CONFIG['freeze_blocks']]:\n", "                freeze_count = RUN_CONFIG.get('freeze_blocks_current', RUN_CONFIG['freeze_blocks'])\n                for block in list(encoder.encoders)[:freeze_count]:\n")
    cells[14]["source"] = s.splitlines(keepends=True)

    # Explicit optimizer groups permit unfreezing all six Conformer blocks at epoch 6.
    s = "".join(cells[15]["source"])
    old = """model.freeze_lower_layers(RUN_CONFIG['freeze_blocks'], freeze_frontend=True)
model.to(device)

optimizer = torch.optim.AdamW([
    dict(params=[p for name, p in model.named_parameters() if p.requires_grad and not any(k in name for k in ('lac', 'domain_classifier'))], lr=RUN_CONFIG['lr_backbone']),
    dict(params=[p for name, p in model.named_parameters() if p.requires_grad and any(k in name for k in ('lac', 'domain_classifier'))], lr=RUN_CONFIG['lr_heads'])
], weight_decay=RUN_CONFIG['weight_decay'])

updates_per_epoch = math.ceil(len(train_loader) / RUN_CONFIG['accumulation'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    max_lr=[RUN_CONFIG['lr_backbone'], RUN_CONFIG['lr_heads']],
    total_steps=updates_per_epoch * RUN_CONFIG['epochs'], pct_start=0.1)"""
    new = """model.freeze_lower_layers(RUN_CONFIG['freeze_blocks'], freeze_frontend=True)
model.to(device)
encoder_blocks = list(model.feature_extractor._encoder.encoders)
if len(encoder_blocks) != 6:
    raise ValueError(f\"V6 gradual-unfreeze expects 6 Conformer blocks, found {len(encoder_blocks)}\")
lower_block_ids = {id(p) for block in encoder_blocks[:3] for p in block.parameters()}
upper_block_ids = {id(p) for block in encoder_blocks[3:] for p in block.parameters()}
head_params = [p for name, p in model.named_parameters() if any(k in name for k in ('lac', 'domain_classifier'))]
encoder_param_ids = lower_block_ids | upper_block_ids
head_param_ids = {id(p) for p in head_params}
other_backbone = [p for p in model.parameters() if id(p) not in encoder_param_ids and id(p) not in head_param_ids and p.requires_grad]
optimizer = torch.optim.AdamW([
    dict(params=[p for block in encoder_blocks[:3] for p in block.parameters()], lr=RUN_CONFIG['lr_unfreeze_lower']),
    dict(params=[p for block in encoder_blocks[3:] for p in block.parameters()], lr=RUN_CONFIG['lr_unfreeze_upper']),
    dict(params=other_backbone, lr=RUN_CONFIG['lr_backbone']),
    dict(params=head_params, lr=RUN_CONFIG['lr_heads'])
], weight_decay=RUN_CONFIG['weight_decay'])

updates_per_epoch = math.ceil(len(train_loader) / RUN_CONFIG['accumulation'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    max_lr=[RUN_CONFIG['lr_unfreeze_lower'], RUN_CONFIG['lr_unfreeze_upper'], RUN_CONFIG['lr_backbone'], RUN_CONFIG['lr_heads']],
    total_steps=updates_per_epoch * RUN_CONFIG['epochs'], pct_start=0.20, anneal_strategy='cos')"""
    if old not in s:
        raise RuntimeError("Could not locate V5 optimizer block for safe V6 transformation")
    s = s.replace(old, new)
    s = s.replace("        model.train()\n        memory.clear()", "        if epoch >= RUN_CONFIG['gradual_unfreeze_epoch']:\n            RUN_CONFIG['freeze_blocks_current'] = 0\n            for block in encoder_blocks:\n                for parameter in block.parameters():\n                    parameter.requires_grad_(True)\n            print('Gradual unfreezing: all six Conformer blocks enabled; lower/upper peak LR =',\n                  RUN_CONFIG['lr_unfreeze_lower'], RUN_CONFIG['lr_unfreeze_upper'])\n        model.train()\n        memory.clear()")
    cells[15]["source"] = s.splitlines(keepends=True)

    # The V5 Kaggle export omitted helper modules present in the repository's
    # canonical template. Embed those definitions so V6 can instantiate the model.
    template_path = ROOT / "notebooks/templates/conformer_bytehum_qbh_final_train.ipynb"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    required_helpers = {
        "CenterLoss", "Conv2dSubsampling8", "FocalLoss", "HardTripletLoss",
        "LinearNoSubsampling", "NoPositionalEncoding",
    }
    snippets = []
    for template_cell in template["cells"]:
        if template_cell["cell_type"] != "code":
            continue
        code = "".join(template_cell["source"])
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in required_helpers:
                snippets.append((node.lineno, ast.get_source_segment(code, node)))
                required_helpers.remove(node.name)
    if required_helpers:
        raise RuntimeError(f"Canonical notebook template lacks helper classes: {sorted(required_helpers)}")
    helper_code = "\n\n".join(code for _, code in sorted(snippets)) + "\n\n"
    cells[10]["source"] = (helper_code + "".join(cells[10]["source"])).splitlines(keepends=True)

    # Update output packaging names in final packaging cell.
    nb["metadata"]["qbh_v6"] = {"generated_from": source.name, "pipeline": "qbh_final_v6"}
    target.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def build_archive(input_zip: Path, contract_path: Path, out_zip: Path, out_contract: Path, out_hashes: Path) -> None:
    old_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if set(old_contract.get("missing_source_query_files", [])) != MISSING_TARGETS:
        raise RuntimeError("Missing source-query list differs from the audited 39 IDs")
    missing_by_split = {"train": 29, "val": 3, "test": 7}
    seen_audio: set[str] = set()
    checksums: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    mapping = None
    combined_manifest = None

    with zipfile.ZipFile(input_zip, "r") as zin, zipfile.ZipFile(
        out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as zout:
        for info in zin.infolist():
            name = info.filename
            dest = name
            if name == "datasets/2_music_database/,.mp3":
                dest = "datasets/2_music_database/" + RENAMED_TITLE
            if name.endswith("song_name_mapping.json") and name == "datasets/song_name_mapping.json":
                mapping = json.loads(zin.read(info).decode("utf-8"))
                mapping[RUSSIAN_TITLE] = RENAMED_TITLE.removesuffix(".mp3")
                payload = json_bytes(mapping)
                zout.writestr(dest, payload, compress_type=zipfile.ZIP_DEFLATED)
                continue
            if name == "datasets/combined_split_manifest.json":
                combined_manifest = json.loads(zin.read(info).decode("utf-8"))
                combined_manifest["actual_original_queries"] = dict(old_contract["query_files_present_and_sha_verified"])
                combined_manifest["missing_original_queries"] = len(MISSING_TARGETS)
                combined_manifest["missing_original_query_ids_excluded"] = sorted(MISSING_TARGETS)
                payload = json_bytes(combined_manifest)
                zout.writestr(dest, payload, compress_type=zipfile.ZIP_DEFLATED)
                continue

            if info.is_dir():
                zout.writestr(dest, b"")
                continue

            # Hash every audio member as bytes pass through, and count only active split queries.
            is_audio = Path(name).suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
            active_split = None
            normalized = name.replace("\\", "/")
            parts = normalized.split("/")
            if len(parts) >= 5 and parts[0:2] == ["datasets", "dataset_split"] and parts[2] in split_counts:
                if is_audio and Path(name).name != "reference.wav" and not Path(name).name.startswith("reference_vocal"):
                    active_split = parts[2]
                    if Path(name).stem in MISSING_TARGETS:
                        raise RuntimeError(f"Missing source query unexpectedly present in active split: {name}")
                    split_counts[active_split] += 1
            digest = hashlib.sha256() if is_audio else None
            with zin.open(info, "r") as src, zout.open(dest, "w") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
                    if digest:
                        digest.update(chunk)
            if digest:
                checksums[dest] = digest.hexdigest()
                seen_audio.add(dest)

        if mapping is None or combined_manifest is None:
            raise RuntimeError("Required mapping or combined manifest missing from source archive")

        expected = old_contract["query_files_present_and_sha_verified"]
        for split, count in expected.items():
            if split_counts[split] != count:
                raise RuntimeError(f"Active {split} query count mismatch: archive={split_counts[split]}, contract={count}")
        query_hash_manifest = {"algorithm": "sha256", "scope": "all audio files in archive", "files": checksums}
        zout.writestr("datasets/qbh_dataset_audio_sha256.json", json_bytes(query_hash_manifest), compress_type=zipfile.ZIP_DEFLATED)

    contract = {
        "schema_version": 2,
        "dataset_archive": out_zip.name,
        "source_split_counts_songs": old_contract["source_split_counts_songs"],
        "source_query_counts": dict(old_contract["query_files_present_and_sha_verified"]),
        "query_files_present_and_sha_verified": dict(old_contract["query_files_present_and_sha_verified"]),
        "missing_source_query_files": [],
        "missing_source_query_counts_by_split": {"train": 0, "val": 0, "test": 0},
        "excluded_missing_source_query_files": sorted(MISSING_TARGETS),
        "excluded_missing_source_query_counts_by_split": missing_by_split,
        "audio_files_sha256_verified": len(checksums),
        "audio_sha256_manifest": out_hashes.name,
        "final_training_ready": True,
        "locked_final_test": old_contract["locked_final_test"],
        "source_split_mapping_sha256": old_contract["source_split_mapping_sha256"],
        "source_humming_manifest_sha256": old_contract["source_humming_manifest_sha256"],
        "locked_manifest_sha256": old_contract["locked_manifest_sha256"],
        "note": "The 39 unavailable source query IDs are explicitly excluded from the active train/val/test scan and retained here for audit. No query audio was fabricated. All audio members in the V6 archive have SHA-256 entries.",
    }
    out_contract.write_bytes(json_bytes(contract))
    out_hashes.write_bytes(json_bytes({"algorithm": "sha256", "files": checksums}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--contract", type=Path, default=SOURCE_CONTRACT)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--archive-only", action="store_true")
    parser.add_argument("--notebook-only", action="store_true")
    args = parser.parse_args()
    backups = ROOT / "backups"
    backups.mkdir(exist_ok=True)
    for src in (args.notebook, args.contract):
        dest = backups / f"{src.stem}_before_v6{src.suffix}"
        if not dest.exists():
            shutil.copy2(src, dest)
    if not args.archive_only:
        make_notebook(args.notebook, OUT_NOTEBOOK)
    if not args.notebook_only:
        build_archive(args.archive, args.contract, OUT_ARCHIVE, OUT_CONTRACT, OUT_HASHES)
        shutil.copy2(OUT_CONTRACT, DIST / "qbh_dataset_contract.json")
        embedded_contract = json.loads(OUT_CONTRACT.read_text(encoding="utf-8"))
        embedded_contract["audio_sha256_manifest"] = "datasets/qbh_dataset_audio_sha256.json"
        with zipfile.ZipFile(OUT_ARCHIVE, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
            zipped.writestr("datasets/qbh_dataset_contract.json", json_bytes(embedded_contract))
    print(f"V6 notebook: {OUT_NOTEBOOK if not args.archive_only else 'skipped'}")
    print(f"V6 dataset:  {OUT_ARCHIVE}")
    print(f"Contract:    {OUT_CONTRACT}")
    print(f"SHA-256:     {OUT_HASHES}")


if __name__ == "__main__":
    main()
