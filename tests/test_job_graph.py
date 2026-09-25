"""Batch job-graph submission (POST /jobs/graph).

Covers: ref->id assignment in array order, forward and cross-tenant
in-batch references, mixing existing-job ids with in-batch refs,
per-tenant weight consistency, whole-graph cycle detection, atomic
rollback of every failure mode (tenant weights, queue and jobs untouched),
and the new jobs immediately following preview/batch dependency judgement.
"""
from __future__ import annotations

import pytest

from app import db, scheduler
from tests.conftest import at, iso  # noqa: F401


def _snapshot():
    """Raw tenants/jobs/dependency rows for rollback assertions."""
    conn = db.get_conn()
    tenants = [tuple(r) for r in conn.execute(
        "SELECT id, weight, deficit, max_gpus, budget_gpu_min, "
        "aging_settled_at, created_at, updated_at FROM tenants ORDER BY id"
    )]
    jobs = [tuple(r) for r in conn.execute("SELECT * FROM jobs ORDER BY id")]
    deps = [tuple(r) for r in conn.execute(
        "SELECT * FROM job_dependencies ORDER BY job_id, depends_on"
    )]
    return tenants, jobs, deps


# ---------------------------------------------------------------------------
# Service level
# ---------------------------------------------------------------------------

def test_graph_happy_path_resolves_refs_and_ids(conn):
    pre = scheduler.submit_job("base", 1, 5)  # J000001, external prerequisite

    res = scheduler.submit_job_graph([
        {"ref": "extract", "tenant_id": "team-a", "gpus": 2,
         "duration_min": 30, "weight": 2.0, "depends_on": [pre["id"]]},
        {"ref": "train", "tenant_id": "team-b", "gpus": 4,
         "duration_min": 60, "weight": 1.5, "depends_on_refs": ["extract"]},
        {"ref": "evaluate", "tenant_id": "team-a", "gpus": 1,
         "duration_min": 10, "weight": 2.0, "depends_on_refs": ["train"]},
    ])

    # Ids are sequential in array order; refs map to them.
    assert res["refs"] == {
        "extract": "J000002", "train": "J000003", "evaluate": "J000004",
    }
    jobs = res["jobs"]
    assert [j["id"] for j in jobs] == ["J000002", "J000003", "J000004"]
    assert [j["status"] for j in jobs] == ["QUEUED"] * 3
    # In-batch refs resolved to the freshly assigned ids.
    assert jobs[0]["depends_on"] == [pre["id"]]
    assert jobs[1]["depends_on"] == ["J000002"]
    assert jobs[2]["depends_on"] == ["J000003"]
    # blocked_by reflects the new graph immediately.
    assert jobs[1]["blocked_by"] == [{"id": "J000002", "status": "QUEUED"}]
    assert jobs[2]["blocked_by"] == [{"id": "J000003", "status": "QUEUED"}]
    # Persisted: plain GET sees the same objects.
    assert scheduler.get_job("J000003")["depends_on"] == ["J000002"]
    # Tenants auto-registered with the declared weights.
    assert scheduler.get_tenant("team-a")["weight"] == 2.0
    assert scheduler.get_tenant("team-b")["weight"] == 1.5


def test_graph_forward_reference_allowed(conn):
    # "second" is declared first but depends on "first", declared later.
    res = scheduler.submit_job_graph([
        {"ref": "second", "tenant_id": "t", "gpus": 1, "duration_min": 5,
         "depends_on_refs": ["first"]},
        {"ref": "first", "tenant_id": "t", "gpus": 1, "duration_min": 5},
    ])
    assert res["refs"] == {"second": "J000001", "first": "J000002"}
    assert res["jobs"][0]["depends_on"] == ["J000002"]
    assert res["jobs"][0]["blocked_by"] == [
        {"id": "J000002", "status": "QUEUED"}
    ]
    assert res["jobs"][1]["depends_on"] == []


def test_graph_combined_prerequisites_deduped_and_sorted(conn):
    p1 = scheduler.submit_job("base", 1, 5)  # J000001
    p2 = scheduler.submit_job("base", 1, 5)  # J000002

    res = scheduler.submit_job_graph([
        {"ref": "a", "tenant_id": "t1", "gpus": 1, "duration_min": 5,
         "depends_on": [p2["id"], p1["id"], p2["id"]]},
        {"ref": "b", "tenant_id": "t2", "gpus": 1, "duration_min": 5,
         "depends_on": [p1["id"]], "depends_on_refs": ["a", "a"]},
    ])
    assert res["jobs"][0]["depends_on"] == [p1["id"], p2["id"]]
    # Existing ids and resolved refs merge into one deduplicated sorted set.
    assert res["jobs"][1]["depends_on"] == [p1["id"], res["refs"]["a"]]


def test_graph_ids_continue_existing_sequence(conn):
    scheduler.submit_job("base", 1, 5)  # J000001
    scheduler.submit_job("base", 1, 5)  # J000002
    res = scheduler.submit_job_graph([
        {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5},
        {"ref": "b", "tenant_id": "t", "gpus": 1, "duration_min": 5},
    ])
    assert res["refs"] == {"a": "J000003", "b": "J000004"}
    # A subsequent single submission continues right after the graph.
    assert scheduler.submit_job("t", 1, 5)["id"] == "J000005"


def test_graph_tenant_upsert_semantics(conn):
    scheduler.upsert_tenant("existing", 1.0, max_gpus=2, budget_gpu_min=100)
    res = scheduler.submit_job_graph([
        {"ref": "a", "tenant_id": "existing", "gpus": 1, "duration_min": 5,
         "weight": 4.0},
        {"ref": "b", "tenant_id": "fresh", "gpus": 1, "duration_min": 5},
    ])
    assert res["refs"] == {"a": "J000001", "b": "J000002"}
    # Existing tenant: weight updated, cap and budget preserved.
    t = scheduler.get_tenant("existing")
    assert t["weight"] == 4.0
    assert t["max_gpus"] == 2
    assert t["budget_gpu_min"] == 100
    # New tenant: auto-registered with the default weight, uncapped.
    t2 = scheduler.get_tenant("fresh")
    assert t2["weight"] == 1.0
    assert t2["max_gpus"] is None
    assert t2["budget_gpu_min"] is None


def test_graph_empty_array_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([])
    assert "non-empty" in str(exc.value)
    with pytest.raises(scheduler.SchedulerError):
        scheduler.submit_job_graph(None)


def test_graph_item_shape_validation(conn):
    def item(**overrides):
        base = {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5}
        base.update(overrides)
        return base

    bad_payloads = [
        ["not-an-object"],
        [item(ref="")],
        [item(ref="  ")],
        [item(ref=123)],
        [item(tenant_id="")],
        [item(gpus=0)],
        [item(gpus=-1)],
        [item(gpus=True)],
        [item(gpus="2")],
        [item(duration_min=0)],
        [item(duration_min=False)],
        [item(weight=0)],
        [item(weight=-1.5)],
        [item(weight=True)],
        [item(weight="heavy")],
        [item(depends_on="J000001")],
        [item(depends_on=[""])],
        [item(depends_on=[123])],
        [item(depends_on_refs="a")],
        [item(depends_on_refs=["  "])],
    ]
    for payload in bad_payloads:
        with pytest.raises(scheduler.SchedulerError):
            scheduler.submit_job_graph(payload)
    # Nothing was ever written.
    assert scheduler.list_jobs() == []
    assert scheduler.list_tenants() == []


def test_graph_duplicate_ref_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5},
            {"ref": "a", "tenant_id": "t", "gpus": 2, "duration_min": 5},
        ])
    assert "duplicate ref 'a'" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_conflicting_weight_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "weight": 1.0},
            {"ref": "b", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "weight": 2.0},
        ])
    assert "conflicting weights" in str(exc.value)
    assert scheduler.list_jobs() == []
    # The tenant was never registered.
    assert scheduler.list_tenants() == []


def test_graph_unknown_external_id_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on": ["J000999"]},
        ])
    assert "unknown depends_on job id(s): J000999" in str(exc.value)
    # A not-yet-created id (the batch's own first id) is not an "existing"
    # job either: in-batch prerequisites must go through refs.
    with pytest.raises(scheduler.SchedulerError) as exc2:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on": ["J000001"]},
        ])
    assert "unknown depends_on" in str(exc2.value)
    assert scheduler.list_jobs() == []


def test_graph_unknown_in_batch_ref_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on_refs": ["ghost"]},
        ])
    assert "unknown depends_on_refs ref(s): ghost" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_cycle_within_batch_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on_refs": ["c"]},
            {"ref": "b", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on_refs": ["a"]},
            {"ref": "c", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on_refs": ["b"]},
        ])
    assert "dependency cycle" in str(exc.value)
    # Cycle nodes are reported by their batch refs.
    assert "a" in str(exc.value) and "c" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_self_reference_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5,
             "depends_on_refs": ["a"]},
        ])
    assert "dependency cycle" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_capacity_rejected(conn):
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.submit_job_graph([
            {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5},
            {"ref": "big", "tenant_id": "t", "gpus": 99, "duration_min": 5},
        ])
    assert "exceeds cluster capacity" in str(exc.value)
    assert "'big'" in str(exc.value)
    assert scheduler.list_jobs() == []


def test_graph_any_failure_rolls_back_everything(conn):
    scheduler.upsert_tenant("team-a", 3.0)
    scheduler.submit_job("team-a", 1, 10, 3.0)  # J000001 queued
    before = _snapshot()

    def item(ref, **overrides):
        base = {"ref": ref, "tenant_id": "team-a", "gpus": 1,
                "duration_min": 5, "weight": 3.0}
        base.update(overrides)
        return base

    failing_payloads = [
        # duplicate ref
        [item("x"), item("x")],
        # conflicting weights for one tenant
        [item("x"), item("y", weight=9.0)],
        # unknown in-batch ref
        [item("x", depends_on_refs=["ghost"])],
        # unknown existing job id
        [item("x", depends_on=["J000999"])],
        # cycle through in-batch refs (weight change included, must not stick)
        [item("x", weight=9.0, depends_on_refs=["y"]),
         item("y", tenant_id="newbie", weight=1.0, depends_on_refs=["x"])],
        # capacity overflow
        [item("x", gpus=99)],
    ]
    for payload in failing_payloads:
        with pytest.raises(scheduler.SchedulerError):
            scheduler.submit_job_graph(payload)
        # Tenant weights/rows, jobs and dependency edges are all untouched.
        assert _snapshot() == before
        assert scheduler.get_tenant("team-a")["weight"] == 3.0
        assert scheduler.list_jobs()[0]["id"] == "J000001"

    # A failed batch consumes no job ids: the next valid graph continues
    # the sequence exactly where the successful submissions left it.
    res = scheduler.submit_job_graph([item("ok")])
    assert res["refs"] == {"ok": "J000002"}


def test_graph_jobs_follow_preview_and_batch_dependency_rules(conn):
    res = scheduler.submit_job_graph([
        {"ref": "extract", "tenant_id": "t1", "gpus": 2, "duration_min": 10},
        {"ref": "train", "tenant_id": "t2", "gpus": 2, "duration_min": 10,
         "depends_on_refs": ["extract"]},
    ])
    a, b = res["refs"]["extract"], res["refs"]["train"]

    # The dependent is queued behind its in-batch prerequisite immediately:
    # preview and committed batch dispatch only the prerequisite.
    assert scheduler.preview_batch(now=at(1))["selection_order"] == [a]
    batch = scheduler.run_batch(now=at(1))
    assert batch["selection_order"] == [a]
    # The dependency-blocked tenant earns nothing while waiting.
    assert all(e["tenant_id"] != "t2" for e in batch["events"])
    assert scheduler.get_job(b)["blocked_by"] == [
        {"id": a, "status": "RUNNING"}
    ]
    assert scheduler.preview_batch(now=at(2))["selection_order"] == []

    # Completing the prerequisite releases the dependent at once.
    scheduler.complete_job(a)
    assert scheduler.preview_batch(now=at(2))["selection_order"] == [b]
    assert scheduler.run_batch(now=at(2))["selection_order"] == [b]
    assert scheduler.get_job(b)["status"] == "RUNNING"


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def test_http_graph_happy_path(client):
    pre = client.post("/jobs", json={
        "tenant_id": "base", "gpus": 1, "duration_min": 5}).json()["id"]

    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "extract", "tenant_id": "team-a", "gpus": 2,
         "duration_min": 30, "weight": 2.0, "depends_on": [pre]},
        {"ref": "train", "tenant_id": "team-b", "gpus": 4,
         "duration_min": 60, "weight": 1.5, "depends_on_refs": ["extract"]},
        {"ref": "evaluate", "tenant_id": "team-a", "gpus": 1,
         "duration_min": 10, "weight": 2.0, "depends_on_refs": ["train"]},
    ]})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["refs"] == {
        "extract": "J000002", "train": "J000003", "evaluate": "J000004"}
    jobs = body["jobs"]
    assert [j["id"] for j in jobs] == ["J000002", "J000003", "J000004"]
    assert jobs[0]["depends_on"] == [pre]
    assert jobs[1]["depends_on"] == ["J000002"]
    assert jobs[2]["depends_on"] == ["J000003"]
    assert jobs[1]["blocked_by"] == [{"id": "J000002", "status": "QUEUED"}]

    # GET reflects the same objects; tenants carry the declared weights.
    assert client.get("/jobs/J000003").json()["depends_on"] == ["J000002"]
    assert client.get("/tenants/team-a").json()["weight"] == 2.0
    assert client.get("/tenants/team-b").json()["weight"] == 1.5


def test_http_graph_validation_422(client):
    url = "/jobs/graph"
    item = {"ref": "a", "tenant_id": "t", "gpus": 1, "duration_min": 5}
    assert client.post(url, json={}).status_code == 422
    assert client.post(url, json={"jobs": []}).status_code == 422
    assert client.post(url, json={"jobs": "x"}).status_code == 422
    # Missing required item fields.
    assert client.post(url, json={"jobs": [
        {"tenant_id": "t", "gpus": 1, "duration_min": 5}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {"ref": "a", "tenant_id": "t", "duration_min": 5}]}).status_code == 422
    # Bad field shapes.
    assert client.post(url, json={"jobs": [
        {**item, "gpus": 0}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {**item, "gpus": True}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {**item, "duration_min": False}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {**item, "weight": 0}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {**item, "depends_on": [""]}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {**item, "depends_on": "J000001"}]}).status_code == 422
    assert client.post(url, json={"jobs": [
        {**item, "depends_on_refs": ["  "]}]}).status_code == 422
    # The failed validations changed nothing.
    assert client.get("/jobs").json() == []
    assert client.get("/tenants").json() == []


def test_http_graph_errors_400_leave_state_untouched(client):
    client.post("/tenants", json={"tenant_id": "team-a", "weight": 3.0})
    client.post("/jobs", json={
        "tenant_id": "team-a", "gpus": 1, "duration_min": 10, "weight": 3.0})
    jobs_before = client.get("/jobs").json()

    def stable_tenants():
        # budget_window_* timestamps are evaluated live at each request, so
        # compare only the persisted fields.
        return [
            {k: v for k, v in t.items()
             if not k.startswith("budget_window_")}
            for t in client.get("/tenants").json()
        ]

    tenants_before = stable_tenants()

    def item(ref, **overrides):
        base = {"ref": ref, "tenant_id": "team-a", "gpus": 1,
                "duration_min": 5, "weight": 3.0}
        base.update(overrides)
        return base

    cases = [
        ([item("x"), item("x")], "duplicate ref"),
        ([item("x"), item("y", weight=9.0)], "conflicting weights"),
        ([item("x", depends_on_refs=["ghost"])], "unknown depends_on_refs"),
        ([item("x", depends_on=["J000999"])], "unknown depends_on job id"),
        ([item("x", depends_on_refs=["y"]),
          item("y", depends_on_refs=["x"])], "dependency cycle"),
        ([item("x", gpus=100)], "exceeds cluster capacity"),
    ]
    for payload, needle in cases:
        r = client.post("/jobs/graph", json={"jobs": payload})
        assert r.status_code == 400, (needle, r.text)
        assert needle in r.json()["error"]
        # Whole batch rolled back: weights, tenants and jobs unchanged.
        assert stable_tenants() == tenants_before
        assert client.get("/jobs").json() == jobs_before


def test_http_graph_then_preview_and_batch(client):
    r = client.post("/jobs/graph", json={"jobs": [
        {"ref": "a", "tenant_id": "t1", "gpus": 2, "duration_min": 10},
        {"ref": "b", "tenant_id": "t2", "gpus": 2, "duration_min": 10,
         "depends_on_refs": ["a"]},
    ]})
    assert r.status_code == 201, r.text
    a, b = r.json()["refs"]["a"], r.json()["refs"]["b"]

    # The new graph immediately follows preview/batch dependency judgement.
    assert client.post("/scheduler/preview").json()["selection_order"] == [a]
    assert client.post("/scheduler/batch").json()["selection_order"] == [a]
    assert client.get(f"/jobs/{b}").json()["status"] == "QUEUED"

    assert client.post(f"/jobs/{a}/complete").status_code == 200
    assert client.post("/scheduler/batch").json()["selection_order"] == [b]
