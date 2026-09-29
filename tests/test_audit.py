import json
from concurrent.futures import ThreadPoolExecutor


def test_original_replay_and_no_prompt(router, request_factory):
    request = request_factory(prompt="PRIVATE BUSINESS INPUT")
    original = router.route(request)
    router.config["version"] = "changed-later"
    router.models[0].available = False
    record = router.audit.replay(request.request_id)[0]
    assert record["response"] == original.model_dump(mode="json")
    assert record["routing_config_version"] != "changed-later"
    assert "PRIVATE BUSINESS INPUT" not in json.dumps(record)
    assert record["prompt_hash"]
    assert record["candidate_snapshot"][0]["available"] is True


def test_concurrent_audit(router, request_factory):
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(lambda _: router.route(request_factory()), range(8)))
    assert len(router.audit.replay("TEST-001")) == 8
