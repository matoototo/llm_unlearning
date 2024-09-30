import re
import argparse
import subprocess

from tqdm import tqdm
from pathlib import Path

def run_lm_eval(checkpoint_path, output_file):
    cmd = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={checkpoint_path}",
        "--tasks", "wmdp,mmlu",
        "--batch_size=auto"
    ]

    table_start_regex = re.compile(r'^\|.*\|$')
    table_separator_regex = re.compile(r'^\|[-:|]+\|$')

    capturing = False
    table_lines = []

    with open(output_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        for line in process.stdout:
            line = line.strip()
            if table_start_regex.match(line):
                if not capturing:
                    capturing = True
                table_lines.append(line)
            elif capturing and table_separator_regex.match(line):
                table_lines.append(line)
            elif capturing:
                if table_start_regex.match(line):
                    table_lines.append(line)
                else:
                    capturing = False
                    if table_lines:
                        f.write('\n'.join(table_lines) + '\n\n')
                        table_lines = []
        if table_lines:
            f.write('\n'.join(table_lines) + '\n\n')

        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)

def process_checkpoints(root_folder):
    checkpoints = list(Path(root_folder).rglob('checkpoint-*'))
    with tqdm(total=len(checkpoints), desc="Processing Checkpoints") as pbar:
        for checkpoint_path in checkpoints:
            if not checkpoint_path.is_dir():
                pbar.update(1)
                continue

            results_folder = checkpoint_path.parent / "results"
            results_folder.mkdir(exist_ok=True)

            checkpoint_number = checkpoint_path.name.split('-')[-1]
            output_file = results_folder / f"lm_eval_checkpoint_{checkpoint_number}.md"

            if output_file.exists():
                print(f"Skipping {checkpoint_path} (already evaluated)")
                pbar.update(1)
                continue

            print(f"Evaluating {checkpoint_path}")
            try:
                run_lm_eval(str(checkpoint_path), str(output_file))
            except subprocess.CalledProcessError as e:
                print(f"Error evaluating {checkpoint_path}: {e}")
            pbar.update(1)

def main():
    parser = argparse.ArgumentParser(description="lm-eval on checkpoints")
    parser.add_argument("root_folder", help="Root folder to search for checkpoints")
    args = parser.parse_args()

    process_checkpoints(args.root_folder)

if __name__ == "__main__":
    main()
