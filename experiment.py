"""Resume-safe experiment helper for the tabular competition loop.

The agent still owns experiment ideas and edits, but this CLI owns the
repeatable bookkeeping: running committed experiments, parsing metrics,
logging outcomes, and safely discarding worse commits.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.tsv"
RUN_LOG_PATH = ROOT / "run.log"
STATE_PATH = ROOT / ".experiment_state.json"
RESULTS_HEADER = ["commit", "val_auc", "elapsed_sec", "status", "description"]
STATE_VERSION = 1


@dataclass
class ResultRow:
    commit: str
    val_auc: float
    elapsed_sec: float
    status: str
    description: str


@dataclass
class Metrics:
    val_auc: float | None
    elapsed_sec: float | None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_cmd(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise SystemExit(f"Command failed: {' '.join(args)}\n{detail}")
    return result


def git(args: list[str], *, check: bool = True) -> str:
    return run_cmd(["git", *args], check=check).stdout.strip()


def current_branch() -> str:
    return git(["branch", "--show-current"])


def current_commit(short: bool = False) -> str:
    args = ["rev-parse", "--short", "HEAD"] if short else ["rev-parse", "HEAD"]
    return git(args)


def tracked_dirty() -> bool:
    return run_cmd(["git", "diff-index", "--quiet", "HEAD", "--"], check=False).returncode != 0


def dirty_tracked_files() -> list[str]:
    output = git(["status", "--short", "--untracked-files=no"])
    return [line for line in output.splitlines() if line.strip()]


def experiment_branches() -> list[str]:
    output = git(["branch", "--list", "exp/*", "--format=%(refname:short)"])
    return [line for line in output.splitlines() if line.strip()]


def commit_exists(rev: str) -> bool:
    return run_cmd(["git", "rev-parse", "--verify", "--quiet", rev], check=False).returncode == 0


def ensure_clean_tracked(action: str) -> None:
    if tracked_dirty():
        files = "\n".join(dirty_tracked_files())
        raise SystemExit(f"Refusing to {action}: tracked files are dirty.\n{files}")


def read_results() -> list[ResultRow]:
    if not RESULTS_PATH.exists():
        return []

    rows: list[ResultRow] = []
    with RESULTS_PATH.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for raw in reader:
            if not raw or not raw.get("commit"):
                continue
            try:
                val_auc = float(raw.get("val_auc") or 0.0)
                elapsed_sec = float(raw.get("elapsed_sec") or 0.0)
            except ValueError:
                continue
            rows.append(
                ResultRow(
                    commit=raw["commit"],
                    val_auc=val_auc,
                    elapsed_sec=elapsed_sec,
                    status=raw.get("status", ""),
                    description=raw.get("description", ""),
                )
            )
    return rows


def commit_logged(commit: str, rows: Iterable[ResultRow] | None = None) -> bool:
    rows = read_results() if rows is None else rows
    prefixes = {commit, commit[:7]}
    return any(row.commit in prefixes or commit.startswith(row.commit) for row in rows)


def best_keep(rows: Iterable[ResultRow] | None = None) -> ResultRow | None:
    rows = read_results() if rows is None else list(rows)
    keep_rows = [row for row in rows if row.status == "keep"]
    if not keep_rows:
        return None
    return max(keep_rows, key=lambda row: row.val_auc)


def ensure_results_file() -> None:
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size > 0:
        return
    with RESULTS_PATH.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(RESULTS_HEADER)


def append_result(row: ResultRow) -> None:
    ensure_results_file()
    rows = read_results()
    if commit_logged(row.commit, rows):
        raise SystemExit(f"Refusing to append duplicate result for commit {row.commit}.")
    with RESULTS_PATH.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                row.commit,
                f"{row.val_auc:.6f}",
                f"{row.elapsed_sec:.1f}",
                row.status,
                row.description,
            ]
        )


def parse_metrics(log_text: str) -> Metrics:
    val_match = re.search(r"^val_auc:\s*([0-9]*\.?[0-9]+)", log_text, re.MULTILINE)
    elapsed_match = re.search(r"^elapsed_seconds:\s*([0-9]*\.?[0-9]+)", log_text, re.MULTILINE)
    val_auc = float(val_match.group(1)) if val_match else None
    elapsed_sec = float(elapsed_match.group(1)) if elapsed_match else None
    return Metrics(val_auc=val_auc, elapsed_sec=elapsed_sec)


def read_run_log_metrics() -> Metrics | None:
    if not RUN_LOG_PATH.exists() or RUN_LOG_PATH.stat().st_size == 0:
        return None
    return parse_metrics(RUN_LOG_PATH.read_text())


def state_payload(
    *,
    status: str,
    description: str,
    active_commit: str | None = None,
    best: ResultRow | None = None,
) -> dict[str, object]:
    active_commit = active_commit or current_commit()
    best = best if best is not None else best_keep()
    return {
        "version": STATE_VERSION,
        "branch": current_branch(),
        "active_commit": active_commit,
        "description": description,
        "status": status,
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "best_commit": best.commit if best else None,
        "best_val_auc": best.val_auc if best else None,
    }


def write_state(payload: dict[str, object]) -> None:
    STATE_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def update_state(**kwargs: object) -> None:
    existing: dict[str, object] = {}
    if STATE_PATH.exists():
        try:
            existing = json.loads(STATE_PATH.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(kwargs)
    existing.setdefault("version", STATE_VERSION)
    existing["updated_at"] = now_iso()
    write_state(existing)


def decide_status(metrics: Metrics, rows: list[ResultRow]) -> str:
    if metrics.val_auc is None:
        return "crash"
    best = best_keep(rows)
    if best is None or metrics.val_auc > best.val_auc:
        return "keep"
    return "discard"


def reset_discarded_commit(full_commit: str, status: str) -> None:
    if status not in {"discard", "crash"}:
        return
    if current_commit() != full_commit:
        raise SystemExit("Refusing to reset: HEAD changed after result logging.")
    ensure_clean_tracked("reset discarded experiment")
    git(["reset", "--hard", "HEAD~1"])


def run_experiment(description: str) -> None:
    ensure_clean_tracked("run experiment")
    full_commit = current_commit()
    short_commit = current_commit(short=True)
    rows = read_results()
    if commit_logged(full_commit, rows):
        raise SystemExit(f"Refusing to run: commit {short_commit} is already in results.tsv.")

    write_state(
        state_payload(
            status="running",
            description=description,
            active_commit=full_commit,
            best=best_keep(rows),
        )
    )

    started = time.time()
    with RUN_LOG_PATH.open("w") as log_file:
        try:
            completed = subprocess.run(
                ["uv", "run", "train.py"],
                cwd=ROOT,
                check=False,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            returncode = completed.returncode
        except FileNotFoundError as exc:
            log_file.write(f"Failed to execute uv: {exc}\n")
            returncode = 127

    metrics = read_run_log_metrics() or Metrics(val_auc=None, elapsed_sec=None)
    if metrics.elapsed_sec is None and metrics.val_auc is not None:
        metrics.elapsed_sec = time.time() - started
    status = "crash" if returncode != 0 and metrics.val_auc is None else decide_status(metrics, rows)
    val_auc = metrics.val_auc if metrics.val_auc is not None else 0.0
    elapsed_sec = metrics.elapsed_sec if metrics.elapsed_sec is not None and status != "crash" else 0.0
    result = ResultRow(short_commit, val_auc, elapsed_sec, status, description)

    append_result(result)
    if status == "keep":
        write_state(
            state_payload(
                status="keep",
                description=description,
                active_commit=full_commit,
                best=result,
            )
        )
        print(f"keep {short_commit}: val_auc={val_auc:.6f} elapsed={elapsed_sec:.1f}s")
        return

    update_state(
        status=status,
        branch=current_branch(),
        active_commit=full_commit,
        description=description,
        best_commit=(best_keep() or result).commit,
        best_val_auc=(best_keep() or result).val_auc,
    )
    reset_discarded_commit(full_commit, status)
    print(f"{status} {short_commit}: val_auc={val_auc:.6f}; reset to {current_commit(short=True)}")


def record_last(description: str) -> None:
    ensure_clean_tracked("record last run")
    full_commit = current_commit()
    short_commit = current_commit(short=True)
    rows = read_results()
    if commit_logged(full_commit, rows):
        raise SystemExit(f"Refusing to record: commit {short_commit} is already in results.tsv.")
    metrics = read_run_log_metrics()
    if metrics is None or metrics.val_auc is None:
        raise SystemExit("Refusing to record: run.log is missing, empty, or has no val_auc.")

    status = decide_status(metrics, rows)
    elapsed_sec = metrics.elapsed_sec if metrics.elapsed_sec is not None else 0.0
    result = ResultRow(short_commit, metrics.val_auc, elapsed_sec, status, description)
    append_result(result)
    best = best_keep()
    write_state(
        state_payload(
            status=f"recorded-{status}",
            description=description,
            active_commit=full_commit,
            best=best,
        )
    )
    print(f"recorded {status} {short_commit}: val_auc={metrics.val_auc:.6f} elapsed={elapsed_sec:.1f}s")


def status_lines() -> list[str]:
    rows = read_results()
    full_commit = current_commit()
    short_commit = current_commit(short=True)
    best = best_keep(rows)
    metrics = read_run_log_metrics()
    logged = commit_logged(full_commit, rows)
    dirty = dirty_tracked_files()
    branches = experiment_branches()

    lines = [
        f"branch: {current_branch()}",
        f"head: {short_commit} ({full_commit})",
        f"experiment_branches: {', '.join(branches) if branches else '(none)'}",
        f"tracked_dirty: {'yes' if dirty else 'no'}",
    ]
    if dirty:
        lines.append("dirty_tracked_files:")
        lines.extend(f"  {line}" for line in dirty)
    if best:
        lines.append(f"best_keep: {best.commit} val_auc={best.val_auc:.6f} desc={best.description}")
    else:
        lines.append("best_keep: (none)")
    if rows:
        lines.append("recent_results:")
        for row in rows[-5:]:
            lines.append(
                f"  {row.commit} {row.status} val_auc={row.val_auc:.6f} "
                f"elapsed={row.elapsed_sec:.1f}s desc={row.description}"
            )
    else:
        lines.append("recent_results: (none)")
    if metrics:
        val = f"{metrics.val_auc:.6f}" if metrics.val_auc is not None else "(missing)"
        elapsed = f"{metrics.elapsed_sec:.1f}s" if metrics.elapsed_sec is not None else "(missing)"
        lines.append(f"run_log: val_auc={val} elapsed={elapsed}")
    else:
        lines.append("run_log: (missing or empty)")
    lines.append(f"head_logged: {'yes' if logged else 'no'}")

    if dirty:
        recommendation = "commit or revert tracked changes before running/resuming"
    elif not current_branch().startswith("exp/") and branches:
        recommendation = f"resume with: uv run python experiment.py resume --branch {branches[-1]}"
    elif not logged and metrics and metrics.val_auc is not None:
        recommendation = "record completed run with: uv run python experiment.py record-last --description \"...\""
    elif not logged:
        recommendation = "run current committed experiment with: uv run python experiment.py run --description \"...\""
    else:
        recommendation = "edit train.py, commit the experiment, then run experiment.py run --description \"...\""
    lines.append(f"recommended_action: {recommendation}")
    return lines


def print_status() -> None:
    print("\n".join(status_lines()))


def resume(branch: str) -> None:
    ensure_clean_tracked("resume branch")
    if not branch.startswith("exp/"):
        raise SystemExit("Refusing to resume: branch must start with exp/.")
    if not commit_exists(branch):
        raise SystemExit(f"Refusing to resume: branch {branch} does not exist.")
    git(["checkout", branch])
    write_state(
        state_payload(
            status="resumed",
            description=f"resumed {branch}",
            active_commit=current_commit(),
        )
    )
    print_status()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show resumable experiment status.")

    run_parser = subparsers.add_parser("run", help="Run the committed experiment and log the result.")
    run_parser.add_argument("--description", required=True, help="Short experiment description.")

    record_parser = subparsers.add_parser("record-last", help="Record existing run.log for current HEAD.")
    record_parser.add_argument("--description", required=True, help="Short experiment description.")

    resume_parser = subparsers.add_parser("resume", help="Checkout an experiment branch safely.")
    resume_parser.add_argument("--branch", required=True, help="Existing experiment branch, e.g. exp/<tag>.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print_status()
    elif args.command == "run":
        run_experiment(args.description)
    elif args.command == "record-last":
        record_last(args.description)
    elif args.command == "resume":
        resume(args.branch)
    else:
        raise SystemExit(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
