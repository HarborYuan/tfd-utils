"""Stress-test the index-build lock on a real HDFS mount.

Spawns N concurrent worker processes that all open `TFRecordRandomAccess` on
the same shard set under HDFS_DIR.  All workers race to grab the build lock
at the same instant.  We want to observe:

  - exactly one worker enters `_build_index` (others wait, then load).
  - every worker ends up with the full expected key set.
  - no worker hangs past the timeout (livelock / deadlock).
  - heartbeat keeps the lock alive across the build.

Run multiple rounds; each round wipes the index/lock files and re-races so
you can hunt for low-probability races on weak-consistency filesystems.

Usage:
    uv run python scripts/stress_hdfs_lock.py            # 3 rounds, 16 workers
    uv run python scripts/stress_hdfs_lock.py 10 32      # 10 rounds, 32 workers
"""

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tfd_utils.pb2 import BytesList, Example, Feature, Features
from tfd_utils.random_access import TFRecordRandomAccess
from tfd_utils.writer import TFRecordWriter


# Use a dedicated subdirectory so we never clobber unrelated files.
HDFS_DIR = "/mnt/hdfs/storage/sunyueyi/tes_env/lock_stress"
NUM_SHARDS = 4
RECORDS_PER_SHARD = 500          # enough work that build is observable (~seconds)
WORKER_TIMEOUT_S = 600           # hard cap per worker per round
LAUNCH_DELAY_S = 3.0             # sync window so workers race for the lock
BUILD_LOG = os.path.join(HDFS_DIR, "_build_log.jsonl")


def _expected_keys() -> set:
    return {
        f"shard{s:02d}_rec{r:04d}"
        for s in range(NUM_SHARDS)
        for r in range(RECORDS_PER_SHARD)
    }


def _shard_paths() -> list:
    last = NUM_SHARDS - 1
    return [
        os.path.join(HDFS_DIR, f"{s:05d}_of_{last:05d}.tfrecord")
        for s in range(NUM_SHARDS)
    ]


def ensure_test_data() -> list:
    """Create NUM_SHARDS TFRecord files under HDFS_DIR if not already present."""
    Path(HDFS_DIR).mkdir(parents=True, exist_ok=True)
    paths = _shard_paths()
    if all(os.path.exists(p) for p in paths):
        print(f"[setup] reusing existing shards under {HDFS_DIR}")
        return paths

    print(f"[setup] writing {NUM_SHARDS} shards × {RECORDS_PER_SHARD} records to {HDFS_DIR}")
    for s, path in enumerate(paths):
        with TFRecordWriter(path) as w:
            for r in range(RECORDS_PER_SHARD):
                key = f"shard{s:02d}_rec{r:04d}"
                ex = Example(features=Features(feature={
                    'key': Feature(bytes_list=BytesList(value=[key.encode()])),
                }))
                w.write(ex.SerializeToString())
    print("[setup] done")
    return paths


def reset_round() -> None:
    """Remove index/lock/log files from the previous round (keep shards)."""
    for fn in os.listdir(HDFS_DIR):
        if (
            fn.endswith('.index')
            or fn.endswith('.tmp')
            or '.lock' in fn
            or fn == os.path.basename(BUILD_LOG)
        ):
            try:
                os.remove(os.path.join(HDFS_DIR, fn))
            except OSError:
                pass


class TrackedReader(TFRecordRandomAccess):
    """Subclass that logs each entry/exit of `_build_index` to BUILD_LOG.

    Used to detect whether the lock actually serialised builds — if more than
    one builder's [start, end] interval overlaps, the lock failed.
    """

    def _build_index(self):
        _append_log({'event': 'build_start', 'pid': os.getpid(), 'ts': time.time()})
        try:
            return super()._build_index()
        finally:
            _append_log({'event': 'build_end', 'pid': os.getpid(), 'ts': time.time()})


def _append_log(record: dict) -> None:
    line = json.dumps(record) + '\n'
    # Append-mode opens are concurrent-safe enough on POSIX; on HDFS this is
    # best-effort. Worst case: a torn record we can spot during analysis.
    with open(BUILD_LOG, 'a') as f:
        f.write(line)


def worker(worker_id: int, shard_paths: list, expected_keys: set,
           launch_at: float, result_queue) -> None:
    """One worker: wait for the launch barrier, build/load index, report."""
    while time.time() < launch_at:
        time.sleep(0.01)

    t0 = time.time()
    try:
        reader = TrackedReader(shard_paths)
        keys = set(reader.get_keys())
        result_queue.put({
            'worker_id': worker_id,
            'pid': os.getpid(),
            'elapsed': time.time() - t0,
            'ok': keys == expected_keys,
            'num_keys': len(keys),
            'index_file': reader.index_file,
            'error': None,
        })
    except Exception as e:
        result_queue.put({
            'worker_id': worker_id,
            'pid': os.getpid(),
            'elapsed': time.time() - t0,
            'ok': False,
            'num_keys': 0,
            'index_file': None,
            'error': repr(e),
        })


def analyze_build_log() -> dict:
    """Read BUILD_LOG and check that no two builders overlapped."""
    if not os.path.exists(BUILD_LOG):
        return {'starts': 0, 'ends': 0, 'overlap': False, 'intervals': []}

    starts_by_pid = {}
    intervals = []
    with open(BUILD_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get('pid')
            if rec.get('event') == 'build_start':
                starts_by_pid[pid] = rec['ts']
            elif rec.get('event') == 'build_end' and pid in starts_by_pid:
                intervals.append((starts_by_pid.pop(pid), rec['ts'], pid))

    intervals.sort()
    overlap = False
    for i in range(1, len(intervals)):
        prev_start, prev_end, prev_pid = intervals[i - 1]
        cur_start, cur_end, cur_pid = intervals[i]
        if cur_start < prev_end:
            overlap = True
            break

    return {
        'starts': len(intervals) + len(starts_by_pid),
        'ends': len(intervals),
        'overlap': overlap,
        'intervals': intervals,
        'unfinished_pids': list(starts_by_pid.keys()),
    }


def run_round(round_idx: int, num_workers: int,
              shard_paths: list, expected_keys: set) -> bool:
    print(f"\n{'=' * 70}")
    print(f"[round {round_idx}] resetting + launching {num_workers} workers")
    print('=' * 70)
    reset_round()

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    launch_at = time.time() + LAUNCH_DELAY_S
    procs = []
    for i in range(num_workers):
        p = ctx.Process(
            target=worker,
            args=(i, shard_paths, expected_keys, launch_at, queue),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=WORKER_TIMEOUT_S)
        if p.is_alive():
            print(f"[round {round_idx}] worker pid={p.pid} hung past "
                  f"{WORKER_TIMEOUT_S}s — killing")
            p.terminate()
            p.join()

    results = []
    while not queue.empty():
        results.append(queue.get())
    results.sort(key=lambda r: r['elapsed'])

    n_ok = sum(1 for r in results if r['ok'])
    n_err = sum(1 for r in results if r['error'])

    print(f"\n[round {round_idx}] worker results "
          f"({len(results)}/{num_workers} returned):")
    for r in results:
        tag = 'OK ' if r['ok'] else 'BAD'
        err = f"  err={r['error']}" if r['error'] else ''
        print(f"  {tag}  worker={r['worker_id']:3d}  pid={r['pid']:7d}  "
              f"elapsed={r['elapsed']:7.2f}s  keys={r['num_keys']}{err}")

    log = analyze_build_log()
    print(f"\n[round {round_idx}] build-log analysis:")
    print(f"  builders entered : {log['starts']}")
    print(f"  builders exited  : {log['ends']}")
    print(f"  intervals overlap: {log['overlap']}")
    if log['unfinished_pids']:
        print(f"  unfinished pids  : {log['unfinished_pids']}")
    for start, end, pid in log['intervals']:
        print(f"    pid={pid:7d}  start={start:.3f}  end={end:.3f}  "
              f"dur={end - start:.2f}s")

    expected_builders = 1
    healthy = (
        n_ok == num_workers
        and n_err == 0
        and log['starts'] == expected_builders
        and not log['overlap']
    )
    verdict = 'PASS' if healthy else 'FAIL'
    print(f"\n[round {round_idx}] verdict: {verdict}  "
          f"(ok={n_ok}/{num_workers}, errors={n_err}, "
          f"builders={log['starts']}, overlap={log['overlap']})")
    return healthy


def main() -> int:
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    num_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    print(f"target dir : {HDFS_DIR}")
    print(f"shards     : {NUM_SHARDS} × {RECORDS_PER_SHARD} records each")
    print(f"runs       : {num_runs}")
    print(f"workers    : {num_workers}\n")

    shard_paths = ensure_test_data()
    expected_keys = _expected_keys()

    failures = 0
    for i in range(1, num_runs + 1):
        if not run_round(i, num_workers, shard_paths, expected_keys):
            failures += 1

    print(f"\n{'=' * 70}")
    print(f"summary: {num_runs - failures}/{num_runs} rounds PASS, "
          f"{failures} FAIL")
    print('=' * 70)
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
