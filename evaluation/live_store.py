"""Single-writer, fsync-backed journal. In-flight attempts are never auto-replayed."""
import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from app.strategies.live_base import BudgetStop, utc_now


def redact(value):
    if isinstance(value, dict): return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    if isinstance(value, str):
        for key in ['OPENAI_API_KEY', 'TYPESAFE_API_KEY', 'OPENROUTER_API_KEY']:
            secret = os.getenv(key)
            if secret: value = value.replace(secret, '[REDACTED]')
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(redact(value), handle, indent=2, allow_nan=False)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def run_lock(directory):
    with (directory / '.run.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another process owns this live run') from None
        try: yield
        finally: fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class LiveJournal:
    def __init__(self, directory, max_calls, max_spend=None):
        self.path = directory / 'observations.jsonl'
        self.events = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                try: self.events.append(json.loads(line))
                except json.JSONDecodeError:
                    raise ValueError('Truncated live journal: refusing automatic replay; inspect the run first') from None
        self.max_calls, self.max_spend = max_calls, max_spend

    def append(self, event):
        event = redact({**event, 'timestamp': utc_now()})
        with self.path.open('a') as handle:
            handle.write(json.dumps(event, allow_nan=False) + '\n')
            handle.flush(); os.fsync(handle.fileno())
        self.events.append(event)

    @property
    def attempts(self): return [e for e in self.events if e['event'] == 'attempt_start']

    @property
    def attempt_results(self): return [e for e in self.events if e['event'] == 'attempt_result']

    @property
    def latest(self):
        rows = {}
        for event in self.events:
            if event['event'] == 'observation': rows[event['key']] = event['row']
        return rows

    def revision(self, key):
        return 1 + max([e.get('revision', 0) for e in self.events if e.get('key') == key], default=0)

    def pending(self, key):
        started_revision = max([e['revision'] for e in self.attempts if e['key'] == key], default=0)
        return started_revision > self.latest.get(key, {}).get('revision', 0)

    def was_started(self, key): return any(e['key'] == key for e in self.attempts)

    def reserve(self, key, revision):
        if len(self.attempts) >= self.max_calls:
            raise BudgetStop()
        if self.max_spend is not None:
            completed = self.attempt_results
            if len(completed) != len(self.attempts) or any(e['data']['routing_cost_usd'] is None for e in completed):
                raise BudgetStop()
            if sum(e['data']['routing_cost_usd'] for e in completed) >= self.max_spend:
                raise BudgetStop()
        self.append({'event': 'attempt_start', 'key': key, 'revision': revision})
