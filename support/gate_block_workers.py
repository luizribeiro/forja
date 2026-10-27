"""Process initialization is sampled independently inside each four-leg block."""
import random
import time
from gate_records import save
from gate_worker import close_workers, worker_pair, worker_policy


class BlockWorkers:
    def __init__(self, row, directory, guard, warmups, seed, timeout):
        self.row, self.directory, self.guard = row, directory, guard
        self.warmups, self.seed, self.timeout = warmups, seed, timeout
        self.active, self.key, self.next_leg = {}, None, 0
        self.info = {"execution": "resident-block", "cache_state": "fresh-request", "blocks": []}

    def persist(self):
        entry = self.info["blocks"][-1]
        save(self.directory / "workers" / f"{entry['phase']}-{entry['index']:02}" / "workers.json", entry)
        save(self.directory / "workers.json", self.info)

    def begin(self, phase, index):
        directory = self.directory / "workers" / f"{phase}-{index:02}"
        directory.mkdir(parents=True)
        seed = f"{self.seed}:worker-start:{phase}:{index}"
        order = "AB"
        if not random.Random(seed).getrandbits(1):
            order = order[::-1]
        entry = {"phase": phase, "index": index, "started": time.time(), "startup_seed": seed}
        self.info["blocks"].append(entry)
        try:
            self.active, info = worker_pair(self.row, directory, self.guard, 1, self.warmups,
                                           timeout=self.timeout, order=order)
            if not self.active:
                raise ValueError("native worker eligibility changed")
            entry.update(info)
            self.key, self.next_leg = (phase, index), 0
        except BaseException as error:
            entry.update(error=str(error), ended=time.time())
            self.persist()
            raise
        self.persist()

    def sample(self, side, output, log, phase, index, leg):
        if self.key != (phase, index):
            if self.active or leg != 0:
                raise ValueError("cannot replace an incomplete worker block")
            self.begin(phase, index)
        if leg != self.next_leg:
            raise ValueError("worker legs must be requested in order")
        event = self.active[side].sample(output, log)
        event["worker_block"] = {"phase": phase, "index": index}
        self.next_leg += 1
        if leg == 3:
            self.close()
        return event

    def close(self):
        if self.active:
            close_workers(self.active)
            entry = self.info["blocks"][-1]
            entry.update(ended=time.time(), legs_completed=self.next_leg, completed=self.next_leg == 4,
                         returncodes={side: worker.process.returncode
                         for side, worker in self.active.items()})
            entry["wall_seconds"] = entry["ended"] - entry["started"]
            self.persist()
        self.active, self.key = {}, None


def block_workers(row, directory, guard, warmups, seed, enabled=True, timeout=3600):
    reason = worker_policy(row, enabled)
    if reason:
        return {}, {"execution": "process-per-leg", "reason": reason}
    workers = BlockWorkers(row, directory, guard, warmups, seed, timeout)
    return workers, workers.info
