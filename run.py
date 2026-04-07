import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any, Dict, List, Tuple, Union
from uuid import UUID, uuid4
import zipfile


class gem5Run:
    """
    Holds all info required to run a single gem5 SE-mode simulation and
    track its lifecycle (spawning, polling, timeout, result saving).
    """

    @classmethod
    def _create(cls, name, gem5_binary, run_script, outdir, params, timeout):
        run = cls()
        run.name        = name
        run.gem5_binary = gem5_binary
        run.run_script  = run_script
        run.params      = params
        run.timeout     = timeout
        run._id         = uuid4()
        run.outdir      = outdir.resolve()
        run.gem5_name   = run.gem5_binary.parent.name
        run.script_name = run.run_script.stem
        run.running     = False
        run.enqueue_time = time.time()
        run.start_time  = 0.0
        run.end_time    = 0.0
        run.return_code = 0
        run.kill_reason = ''
        run.status      = "Created"
        run.pid         = 0
        run.task_id     = None
        run.results     = None
        return run

    @classmethod
    def createSERun(cls, name, gem5_binary, run_script, outdir,
                    *params, timeout=60*15):
        run = cls._create(name, Path(gem5_binary), Path(run_script),
                          Path(outdir), params, timeout)
        run.string  = f"{run.gem5_name} {run.script_name} "
        run.string += ' '.join(run.params)
        run.command = [
            str(run.gem5_binary),
            '-re', f'--outdir={run.outdir}',
            str(run.run_script),
        ] + list(params)
        run.hash = run._getHash()
        run.type = 'gem5 run'
        os.makedirs(run.outdir, exist_ok=True)
        run.dumpJson('info.json')
        return run

    @classmethod
    def loadFromDict(cls, d):
        run = cls()
        for k, v in d.items():
            setattr(run, k, v)
        return run

    def __repr__(self):
        return str(self._getSerializable())

    def _getSerializable(self):
        d = vars(self).copy()
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d

    def _getHash(self):
        to_hash = [str(self.run_script).encode(),
                   ' '.join(self.params).encode()]
        return hashlib.md5(b''.join(to_hash)).hexdigest()

    @classmethod
    def _convertForJson(cls, d):
        for k, v in d.items():
            if isinstance(v, UUID):
                d[k] = str(v)
        return d

    def dumpJson(self, filename):
        d = self._convertForJson(self._getSerializable())
        with open(self.outdir / filename, 'w') as f:
            json.dump(d, f)

    def dumpsJson(self):
        d = self._convertForJson(self._getSerializable())
        return json.dumps(d)

    def run(self, task=None, cwd='.'):
        self.status     = "Spawning"
        self.start_time = time.time()
        self.dumpJson('info.json')

        proc = subprocess.Popen(self.command, cwd=cwd)

        def handler(signum, frame):
            proc.kill()
            self.kill_reason = 'sigterm'
            self.dumpJson('info.json')

        signal.signal(signal.SIGTERM, handler)

        while proc.poll() is None:
            self.status       = "Running"
            self.current_time = time.time()
            self.pid          = proc.pid
            self.running      = True
            if self.current_time - self.start_time > self.timeout:
                proc.kill()
                self.kill_reason = 'timeout'
            self.dumpJson('info.json')
            time.sleep(5)

        print(f"Done running {' '.join(self.command)}")
        self.running     = False
        self.end_time    = time.time()
        self.return_code = proc.returncode
        self.status      = "Finished" if self.return_code == 0 else "Failed"
        self.dumpJson('info.json')
        self.saveResults()
        print(f"Done storing results of {' '.join(self.command)}")

    def saveResults(self):
        with zipfile.ZipFile(self.outdir / 'results.zip', 'w',
                             zipfile.ZIP_DEFLATED) as zipf:
            for path in self.outdir.glob("**/*"):
                if path.name == 'results.zip':
                    continue
                zipf.write(path, path.relative_to(self.outdir.parent))

    def __str__(self):
        return self.string + ' -> ' + self.status
