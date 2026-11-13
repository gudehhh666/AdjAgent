"""Deterministic lease clock/fencing regressions; no provider calls."""

import json

import pytest

from adj_partner.contracts import Budget, SessionConfig
from adj_partner.runtime.journal import LeaseLost, RunBusy, RunJournal, _system_boot_id
from adj_partner.storage import Store


class Clocks:
    def __init__(self, *, wall=1000.0, monotonic=10.0, boot="boot-aaaa"):
        self.wall_value = wall
        self.monotonic_value = monotonic
        self.boot_value = boot

    def wall(self):
        return self.wall_value

    def monotonic(self):
        return self.monotonic_value

    def boot_id(self):
        return self.boot_value


def make(tmp_path, clocks):
    store = Store(tmp_path / "state")
    sid = store.create_session(SessionConfig().to_dict())["id"]
    journal = RunJournal(store, wall_clock=clocks.wall,
                         monotonic_clock=clocks.monotonic,
                         boot_id_provider=clocks.boot_id)
    rid = journal.start(sid, "lease test", {}, Budget().to_dict())["id"]
    return store, journal, rid


def test_wall_clock_jump_does_not_expire_same_boot_lease(tmp_path):
    clocks = Clocks()
    _, journal, rid = make(tmp_path, clocks)
    owner = journal.acquire(rid, ttl=15)

    clocks.wall_value += 1200  # jump beyond any wall-based lease deadline
    clocks.monotonic_value += 4

    journal.renew(rid, owner, ttl=15)
    with pytest.raises(RunBusy):
        journal.acquire(rid)
    journal.checkpoint(rid, owner, state={"phase": "ready"})


def test_real_host_boot_id_uses_same_boot_monotonic_lease_when_available(tmp_path):
    boot_id = _system_boot_id()
    if boot_id is None:
        pytest.skip("host does not expose a stable boot identity in this process")
    clocks = Clocks(boot=boot_id)
    store, journal, rid = make(tmp_path, clocks)
    owner = journal.acquire(rid, ttl=15)
    with store.transaction() as conn:
        row = dict(conn.execute("SELECT * FROM runtime_leases WHERE run_id=?", (rid,)).fetchone())
    assert row["clock_id"] == boot_id and row["expires_monotonic"] == clocks.monotonic_value + 15

    clocks.wall_value += 1200
    clocks.monotonic_value += 3
    journal.renew(rid, owner, ttl=15)
    with pytest.raises(RunBusy):
        journal.acquire(rid)


def test_monotonic_expiry_allows_takeover_but_fences_old_owner(tmp_path):
    clocks = Clocks()
    _, journal, rid = make(tmp_path, clocks)
    old_owner = journal.acquire(rid, ttl=5)
    clocks.monotonic_value += 5.01

    new_owner = journal.acquire(rid, ttl=5)
    assert new_owner != old_owner
    with pytest.raises(LeaseLost):
        journal.renew(rid, old_owner)
    with pytest.raises(LeaseLost):
        journal.checkpoint(rid, old_owner, state={"phase": "stale"})
    journal.checkpoint(rid, new_owner, state={"phase": "ready"})


def test_new_boot_can_take_over_monotonic_lease_and_old_owner_stays_fenced(tmp_path):
    clocks = Clocks()
    _, journal, rid = make(tmp_path, clocks)
    old_owner = journal.acquire(rid, ttl=100)
    clocks.boot_value = "boot-bbbb"

    new_owner = journal.acquire(rid, ttl=100)
    assert new_owner != old_owner
    with pytest.raises(LeaseLost):
        journal.checkpoint(rid, old_owner, state={"phase": "stale"})
    journal.checkpoint(rid, new_owner, state={"phase": "ready"})


def test_missing_boot_identity_fails_closed_for_existing_monotonic_lease(tmp_path):
    clocks = Clocks()
    _, journal, rid = make(tmp_path, clocks)
    old_owner = journal.acquire(rid, ttl=100)
    clocks.boot_value = None

    with pytest.raises(LeaseLost, match="clock identity"):
        journal.renew(rid, old_owner)
    with pytest.raises(LeaseLost, match="clock identity"):
        journal.checkpoint(rid, old_owner, state={"phase": "must_not_commit"})
    with pytest.raises(LeaseLost, match="clock identity"):
        journal.acquire(rid)


def test_new_lease_explicitly_uses_wall_fallback_when_boot_id_unavailable(tmp_path):
    clocks = Clocks(boot=None)
    store, journal, rid = make(tmp_path, clocks)
    owner = journal.acquire(rid, ttl=5)
    with store.transaction() as conn:
        row = dict(conn.execute("SELECT * FROM runtime_leases WHERE run_id=?", (rid,)).fetchone())
    assert row["clock_id"] == "wall"
    clocks.wall_value += 4
    clocks.boot_value = "boot-aaaa"  # availability recovery does not implicitly change domains
    journal.renew(rid, owner, ttl=5)
    with store.transaction() as conn:
        renewed = dict(conn.execute("SELECT * FROM runtime_leases WHERE run_id=?", (rid,)).fetchone())
    assert renewed["clock_id"] == "wall" and renewed["expires_monotonic"] is None
    clocks.wall_value += 5.01
    with pytest.raises(LeaseLost):
        journal.checkpoint(rid, owner, state={"phase": "stale"})


def test_renewal_does_not_downgrade_existing_monotonic_row_on_intermittent_boot_probe(tmp_path):
    clocks = Clocks()
    store = Store(tmp_path / "state")
    sid = store.create_session(SessionConfig().to_dict())["id"]
    journal = RunJournal(store, wall_clock=clocks.wall, monotonic_clock=clocks.monotonic,
                         boot_id_provider=clocks.boot_id)
    rid = journal.start(sid, "intermittent probe", {}, Budget().to_dict())["id"]
    owner = journal.acquire(rid, ttl=10)
    calls = 0

    def intermittent_boot_id():
        nonlocal calls
        calls += 1
        return clocks.boot_value if calls <= 1 else None

    journal._boot_id_provider = intermittent_boot_id
    journal.renew(rid, owner, ttl=10)
    with store.transaction() as conn:
        row = dict(conn.execute("SELECT * FROM runtime_leases WHERE run_id=?", (rid,)).fetchone())
    assert calls == 1  # fencing reads identity once; renewal keeps that established domain
    assert row["clock_id"] == "boot-aaaa" and row["expires_monotonic"] == clocks.monotonic_value + 10


def test_unavailable_boot_identity_does_not_change_existing_lease_row(tmp_path):
    clocks = Clocks()
    store, journal, rid = make(tmp_path, clocks)
    owner = journal.acquire(rid, ttl=10)
    with store.transaction() as conn:
        before = dict(conn.execute("SELECT * FROM runtime_leases WHERE run_id=?", (rid,)).fetchone())
    clocks.boot_value = None

    with pytest.raises(LeaseLost, match="clock identity"):
        journal.renew(rid, owner)
    with store.transaction() as conn:
        after = dict(conn.execute("SELECT * FROM runtime_leases WHERE run_id=?", (rid,)).fetchone())
    assert after == before


def test_pre_migration_lease_rows_keep_wall_expiry_semantics(tmp_path):
    clocks = Clocks(boot=None)
    store, journal, rid = make(tmp_path, clocks)
    old_owner = journal.acquire(rid, ttl=5)
    with store.transaction(write=True) as conn:
        conn.execute("UPDATE runtime_leases SET clock_id=NULL,expires_monotonic=NULL WHERE run_id=?", (rid,))

    clocks.wall_value += 4
    with pytest.raises(RunBusy):
        journal.acquire(rid)
    clocks.wall_value += 1.01
    new_owner = journal.acquire(rid)
    assert new_owner != old_owner
    with pytest.raises(LeaseLost):
        journal.checkpoint(rid, old_owner, state={"phase": "stale"})


def test_heartbeat_diagnostic_is_sanitized_and_keeps_dual_clock_reading(tmp_path):
    clocks = Clocks()
    store, journal, rid = make(tmp_path, clocks)
    owner = journal.acquire(rid)
    secret = "/private/provider-token-sentinel"

    error = journal.record_heartbeat_failure(rid, owner, OSError(secret))

    assert isinstance(error, LeaseLost)
    assert "OSError" in str(error)
    assert secret not in str(error)
    with store.transaction() as conn:
        rows = [dict(row) for row in conn.execute(
            "SELECT * FROM runtime_lease_diagnostics WHERE run_id=?", (rid,))]
    assert len(rows) == 1
    diagnostic = rows[0]
    assert diagnostic["category"] == "heartbeat_renewal_error"
    assert diagnostic["exception_type"] == "OSError"
    assert diagnostic["wall_time"] == 1000.0
    assert diagnostic["monotonic_time"] == 10.0
    assert diagnostic["clock_id"] == "boot-aaaa"
    assert secret not in json.dumps(diagnostic)
    assert journal.diagnostics(rid)[0]["category"] == "heartbeat_renewal_error"
