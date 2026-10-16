from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import threading

MEMORY_LIMIT = 4 << 30
CPU_SECONDS = 20
FILE_LIMIT = 32 << 20
OUTPUT_LIMIT = 64 * 1024
ENVIRONMENT = ("PATH", "HOME", "LANG")
LIMITS = (
    "import resource, runpy, sys\n"
    f"resource.setrlimit(resource.RLIMIT_AS, ({MEMORY_LIMIT}, {MEMORY_LIMIT}))\n"
    f"resource.setrlimit(resource.RLIMIT_CPU, ({CPU_SECONDS}, {CPU_SECONDS}))\n"
    f"resource.setrlimit(resource.RLIMIT_FSIZE, ({FILE_LIMIT}, {FILE_LIMIT}))\n"
    "runpy.run_path(sys.argv[1], run_name='__main__')\n"
)


def drain(stream, sink: list) -> None:
    size = 0
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        if size < OUTPUT_LIMIT:
            sink.append(chunk[: OUTPUT_LIMIT - size])
        size += len(chunk)


def run_program(program: str, timeout: float) -> dict:
    with tempfile.TemporaryDirectory(prefix="roleflow_") as directory:
        path = os.path.join(directory, "program.py")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(program)
        environment = {key: os.environ[key] for key in ENVIRONMENT if key in os.environ}
        environment["PYTHONHASHSEED"] = "0"
        process = subprocess.Popen(
            [sys.executable, "-I", "-c", LIMITS, path],
            cwd=directory,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout, stderr = [], []
        readers = [threading.Thread(target=drain, args=(process.stdout, stdout), daemon=True), threading.Thread(target=drain, args=(process.stderr, stderr), daemon=True)]
        for reader in readers:
            reader.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            for reader in readers:
                reader.join(timeout=1.0)
        return {
            "passed": process.returncode == 0 and not timed_out,
            "returncode": None if timed_out else process.returncode,
            "timeout": timed_out,
            "stdout": b"".join(stdout).decode("utf-8", errors="replace"),
            "stderr": b"".join(stderr).decode("utf-8", errors="replace"),
        }
