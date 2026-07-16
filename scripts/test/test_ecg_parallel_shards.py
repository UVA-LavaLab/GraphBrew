import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_parallel_shards_are_isolated(tmp_path):
    shards = tmp_path / "shards.tsv"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/slurm/make_slurm_shards.py",
            "--profile", "ecg_smoke",
            "--run-tag", "local_parallel_test",
            "--out", str(shards),
        ],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert len(shards.read_text().splitlines()) == 7

    run_root = tmp_path / "runs"
    launched = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/flows/run_local_shards.py",
            "--shards", str(shards),
            "--run-root", str(run_root),
            "--jobs", "4",
            "--cache-sim-jobs", "4",
            "--dry-run",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert launched.returncode == 0, launched.stdout + launched.stderr
    assert launched.stdout.count("[dry-run]") == 7
    assert launched.stdout.count("--lock-path") == 7
    assert launched.stdout.count("--graph-dir") == 7
    assert launched.stdout.count("--no-build") == 7
    assert "caps={'cache-sim': 4, 'gem5': 1, 'sniper': 1}" in launched.stdout


def test_parallel_shard_reader_rejects_duplicates(tmp_path):
    module = load_module(
        "run_local_shards_test",
        ROOT / "scripts/experiments/ecg/flows/run_local_shards.py")
    shard_file = tmp_path / "duplicate.tsv"
    row = "ecg_smoke\t01_ecg_cache_sim_smoke\tsynthetic_g12\tpr\tLRU\ttag\n"
    shard_file.write_text(row + row)
    suites = {"01_ecg_cache_sim_smoke": "cache-sim"}
    try:
        module.read_shards(shard_file, suites)
    except SystemExit as error:
        assert "duplicate shard row" in str(error)
    else:
        raise AssertionError("duplicate shards were accepted")


def test_full_3sim_smoke_expands_to_120_shards(tmp_path):
    shards = tmp_path / "three_sim.tsv"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/slurm/make_slurm_shards.py",
            "--profile", "ecg_3sim_allalg_smoke",
            "--run-tag", "three_sim_smoke",
            "--out", str(shards),
        ],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    rows = [line.split("\t") for line in shards.read_text().splitlines()]
    assert len(rows) == 120
    assert {row[3] for row in rows} == {"pr", "bfs", "sssp", "bc", "cc"}
    assert len({row[4] for row in rows}) == 8


def test_full_3sim_realgraph_expands_to_360_shards(tmp_path):
    shards = tmp_path / "three_sim_realgraph.tsv"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/slurm/make_slurm_shards.py",
            "--profile", "ecg_3sim_realgraph_allalg",
            "--run-tag", "three_sim_realgraph",
            "--out", str(shards),
            "--allow-missing-graphs",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    rows = [line.split("\t") for line in shards.read_text().splitlines()]
    assert len(rows) == 360
    assert {row[2] for row in rows} == {
        "web-Google", "soc-pokec", "cit-Patents"}
    assert {row[3] for row in rows} == {"pr", "bfs", "sssp", "bc", "cc"}
    assert len({row[4] for row in rows}) == 8


def test_capped_3sim_realgraph_expands_to_360_shards(tmp_path):
    shards = tmp_path / "three_sim_realgraph_1b.tsv"
    generated = subprocess.run(
        [
            sys.executable,
            "scripts/experiments/ecg/slurm/make_slurm_shards.py",
            "--profile", "ecg_3sim_realgraph_allalg_1b",
            "--run-tag", "three_sim_realgraph_1b",
            "--out", str(shards),
            "--allow-missing-graphs",
        ],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert generated.returncode == 0, generated.stdout + generated.stderr
    rows = [line.split("\t") for line in shards.read_text().splitlines()]
    assert len(rows) == 360
    assert {row[1].split("_", 1)[0] for row in rows} == {"22", "25", "26"}


def test_slurm_shards_use_per_run_lock():
    source = (
        ROOT / "scripts/experiments/ecg/slurm/slurm_final_shard.sbatch"
    ).read_text()
    assert '--lock-path "$run_dir/.paper_run.lock"' in source
    assert '--graph-dir "${GRAPHBREW_GRAPH_DIR:-results/graphs}"' in source
    local = (
        ROOT / "scripts/experiments/ecg/flows/run_local_shards.py"
    ).read_text()
    assert '".local_shard.lock"' in local
    assert "fcntl.LOCK_NB" in local
