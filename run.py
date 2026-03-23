import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4
import zipfile


class gem5Run:
    """
    Holds all info required to run a single gem5 SE-mode simulation and
    track its lifecycle (spawning, polling, timeout, result saving).
    """

    _id: UUID
    hash: str
    type: str
    name: str
    gem5_binary: Path
    run_script: Path
    params: Tuple[str, ...]
    timeout: int

    gem5_name: str
    script_name: str
    string: str
    outdir: Path
    command: List[str]

    running: bool
    enqueue_time: float
    start_time: float
    end_time: float
    return_code: int
    kill_reason: str
    status: str
    pid: int
    task_id: Any

    @classmethod
    def _create(cls,
                name: str,
                gem5_binary: Path,
                run_script: Path,
                outdir: Path,
                params: Tuple[str, ...],
                timeout: int) -> 'gem5Run':
        run = cls()
        run.name = name
        run.gem5_binary = gem5_binary
        run.run_script = run_script
        run.params = params
        run.timeout = timeout

        run._id = uuid4()
        run.outdir = outdir.resolve()

        # Assumes **/<gem5_name>/gem5.<anything>
        run.gem5_name = run.gem5_binary.parent.name
        # Assumes **/<script_name>.py
        run.script_name = run.run_script.stem

        run.running = False
        run.enqueue_time = time.time()
        run.start_time = 0.0
        run.end_time = 0.0
        run.return_code = 0
        run.kill_reason = ''
        run.status = "Created"
        run.pid = 0
        run.task_id = None
        run.results = None

        return run

    @classmethod
    def createSERun(cls,
                    name: str,
                    gem5_binary: str,
                    run_script: str,
                    outdir: str,
                    *params: str,
                    timeout: int = 60 * 15) -> 'gem5Run':
        """
        Create a gem5 SE-mode run.

        name        — human-readable label (e.g. "qsort_multi_branch")
        gem5_binary — path to compiled gem5 binary
        run_script  — path to gem5 Python config script
        outdir      — directory where stats/logs will be written
        *params     — extra arguments forwarded to the run script
        timeout     — seconds before the subprocess is killed
        """
        run = cls._create(name, Path(gem5_binary), Path(run_script),
                          Path(outdir), params, timeout)

        run.string = f"{run.gem5_name} {run.script_name} "
        run.string += ' '.join(run.params)

        run.command = [
            str(run.gem5_binary),
            '-re', f'--outdir={run.outdir}',
            str(run.run_script),
        ]
        run.command += list(params)

        run.hash = run._getHash()
        run.type = 'gem5 run'

        os.makedirs(run.outdir, exist_ok=True)
        run.dumpJson('info.json')

        return run

    @classmethod
    def loadJson(cls, filename: str) -> 'gem5Run':
        with open(filename) as f:
            d = json.load(f)
            d['_id'] = UUID(d['_id'])
        try:
            return cls.loadFromDict(d)
        except KeyError:
            print(f"Incompatible json file: {filename}!")
            raise

    @classmethod
    def loadFromDict(cls, d: Dict[str, Union[str, UUID]]) -> 'gem5Run':
        run = cls()
        for k, v in d.items():
            setattr(run, k, v)
        return run

    def __repr__(self) -> str:
        return str(self._getSerializable())

    def _getSerializable(self) -> Dict[str, Union[str, UUID]]:
        d = vars(self).copy()
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def _getHash(self) -> str:
        to_hash = [
            str(self.run_script).encode(),
            ' '.join(self.params).encode(),
        ]
        return hashlib.md5(b''.join(to_hash)).hexdigest()

    @classmethod
    def _convertForJson(cls, d: Dict[str, Any]) -> Dict[str, str]:
        for k, v in d.items():
            if isinstance(v, UUID):
                d[k] = str(v)
        return d

    def dumpJson(self, filename: str) -> None:
        d = self._convertForJson(self._getSerializable())
        with open(self.outdir / filename, 'w') as f:
            json.dump(d, f)

    def dumpsJson(self) -> str:
        d = self._convertForJson(self._getSerializable())
        return json.dumps(d)

    def run(self, task: Any = None, cwd: str = '.') -> None:
        """
        Fork a subprocess running the gem5 command, poll it every 5 seconds,
        and save results when it finishes or times out.
        """
        self.status = "Spawning"
        self.start_time = time.time()
        self.dumpJson('info.json')

        proc = subprocess.Popen(self.command, cwd=cwd)

        def handler(signum, frame):
            proc.kill()
            self.kill_reason = 'sigterm'
            self.dumpJson('info.json')

        signal.signal(signal.SIGTERM, handler)

        while proc.poll() is None:
            self.status = "Running"
            self.current_time = time.time()
            self.pid = proc.pid
            self.running = True

            if self.current_time - self.start_time > self.timeout:
                proc.kill()
                self.kill_reason = 'timeout'

            self.dumpJson('info.json')
            time.sleep(5)

        print(f"Done running {' '.join(self.command)}")

        self.running = False
        self.end_time = time.time()
        self.return_code = proc.returncode
        self.status = "Finished" if self.return_code == 0 else "Failed"

        self.dumpJson('info.json')
        self.saveResults()

        print(f"Done storing results of {' '.join(self.command)}")

    def saveResults(self) -> None:
        """Zip the output directory for archiving."""
        with zipfile.ZipFile(self.outdir / 'results.zip', 'w',
                             zipfile.ZIP_DEFLATED) as zipf:
            for path in self.outdir.glob("**/*"):
                if path.name == 'results.zip':
                    continue
                zipf.write(path, path.relative_to(self.outdir.parent))

    def __str__(self) -> str:
        return self.string + ' -> ' + self.status


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch a single gem5 SE run")

    parser.add_argument("--gem5-bin", required=True,
                        help="Path to gem5 binary (e.g. build/X86/gem5.opt)")
    parser.add_argument("--run-script", required=True,
                        help="Path to gem5 config script (e.g. gem5-config/run_micro.py)")
    parser.add_argument("--outdir", required=True,
                        help="Output directory for results")
    parser.add_argument("--name", default="gem5-run",
                        help="Human-readable name for this run")
    parser.add_argument("--timeout", type=int, default=60 * 15,
                        help="Timeout in seconds (default: 900)")
    parser.add_argument(
        "--bp-type",
        default="multi_branch",
        choices=["none", "local", "bimode", "tage_base", "multi_branch"],
        help="Branch predictor type to pass to the run script"
    )
    parser.add_argument("extra_args", nargs=argparse.REMAINDER,
                        help="Additional arguments forwarded to the run script")

    args = parser.parse_args()

    params = [f"--bp_type={args.bp_type}"] + args.extra_args

    run = gem5Run.createSERun(
        name=args.name,
        gem5_binary=args.gem5_bin,
        run_script=args.run_script,
        outdir=args.outdir,
        *params,
        timeout=args.timeout,
    )

    run.run()
