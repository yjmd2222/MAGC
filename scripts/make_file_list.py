import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath("."))

from utils.file import list_image_files


def gather_images(folders: Sequence[str], follow_links: bool) -> List[str]:
    files: List[str] = []
    for folder in folders:
        folder_files = list_image_files(
            folder,
            exts=(".jpg", ".png", ".jpeg"),
            follow_links=follow_links,
            log_progress=True,
            log_every_n_files=10000,
        )
        print(f"find {len(folder_files)} images in {folder}")
        files.extend(folder_files)
    return sorted(files)


def pair_by_basename(hr_files: Sequence[str], ref_files: Sequence[str]) -> List[Tuple[str, str]]:
    if len(hr_files) != len(ref_files):
        raise ValueError(f"hr/ref file counts differ: {len(hr_files)} vs {len(ref_files)}")

    pairs: List[Tuple[str, str]] = []
    for hr_path, ref_path in zip(hr_files, ref_files):
        if Path(hr_path).name != Path(ref_path).name:
            raise ValueError(
                "hr/ref ordering mismatch:\n"
                f"  hr:  {hr_path}\n"
                f"  ref: {ref_path}"
            )
        pairs.append((hr_path, ref_path))
    return pairs


def write_list(file_path: str, items: Iterable[str]) -> None:
    with open(file_path, "w") as fp:
        for item in items:
            fp.write(f"{item}\n")


parser = ArgumentParser()
parser.add_argument("--hr", nargs="+", required=True, help="One or more HR image folders.")
parser.add_argument("--ref", nargs="+", required=True, help="One or more reference image folders.")
parser.add_argument(
    "--val-size",
    type=int,
    default=0,
    help="Number of paired samples to place in the validation split. Defaults to 0.",
)
parser.add_argument(
    "--save-folder",
    type=str,
    default="filelists",
    help="Output directory for generated manifest files. Defaults to filelists.",
)
parser.add_argument("--follow-links", action="store_true")
args = parser.parse_args()

hr_files = gather_images(args.hr, args.follow_links)
ref_files = gather_images(args.ref, args.follow_links)
pairs = pair_by_basename(hr_files, ref_files)

if args.val_size < 0:
    raise ValueError(f"val_size must be >= 0, got {args.val_size}")
if args.val_size >= len(pairs):
    raise ValueError(f"val_size must be smaller than total paired samples {len(pairs)}")

val_pairs = pairs[:args.val_size]
train_pairs = pairs[args.val_size:]

os.makedirs(args.save_folder, exist_ok=True)

write_list(os.path.join(args.save_folder, "train_hr.txt"), (hr for hr, _ in train_pairs))
write_list(os.path.join(args.save_folder, "train_ref.txt"), (ref for _, ref in train_pairs))
write_list(os.path.join(args.save_folder, "val_hr.txt"), (hr for hr, _ in val_pairs))
write_list(os.path.join(args.save_folder, "val_ref.txt"), (ref for _, ref in val_pairs))

print(f"find {len(pairs)} paired samples")
print(f"train split: {len(train_pairs)}")
print(f"val split: {len(val_pairs)}")
print(f"saved manifests to {os.path.abspath(args.save_folder)}")
