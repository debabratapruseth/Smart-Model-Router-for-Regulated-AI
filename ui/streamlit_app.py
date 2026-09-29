"""Run from the project root: streamlit run ui/streamlit_app.py."""
import json
import sys
from pathlib import Path
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError
from app.core.config_loader import ROOT, load_config
from app.core.router import ModelRouter
from app.models.request import RoutingRequest
from evaluation.business_case import calculate
from evaluation.research_design import BENCHMARK_NAMES, AGREEMENT_NOTE, METHODOLOGY_NOTE, final_test_data
from evaluation.reporting import report_directories, load_report, display_summary

load_dotenv(ROOT / ".env")
st.set_page_config(page_title="Smart Model Router for Banking", page_icon="🏦", layout="wide")


def developer_view(router):
    st.subheader("Developer / Architect")
    st.caption("Inspect hard policy filtering before model ranking. No models are executed in this view.")
    examples = json.loads((ROOT / "data/sample_requests.json").read_text())
    chosen = st.selectbox("Example request", range(len(examples)), format_func=lambda i: examples[i]["prompt"][:90])
    example = examples[chosen]
    with st.form("route_form"):
        prompt = st.text_area("Business prompt", value=example["prompt"], height=110)
        metadata_text = st.text_area("Metadata (JSON)", value=json.dumps(example["metadata"], indent=2), height=280)
        strategy = st.selectbox("Routing strategy", list(router.strategies), index=1)
        submitted = st.form_submit_button("Route request", type="primary")
    if submitted:
        try:
            request = RoutingRequest(prompt=prompt, metadata=json.loads(metadata_text), request_id=f"UI-{uuid4()}")
            result = router.route(request, strategy)
            st.session_state["decision"] = result.model_dump(mode="json")
            st.session_state["request"] = request.model_dump(mode="json")
        except (ValueError, ValidationError) as exc:
            st.error(str(exc))
    if "decision" in st.session_state:
        decision = st.session_state["decision"]
        cols = st.columns(4)
        cols[0].metric("Status", decision["status"])
        cols[1].metric("Selected model", (decision.get("decision") or {}).get("selected_model", "—"))
        cols[2].metric("Routing latency", f'{decision["estimated"]["routing_latency_ms"]:.2f} ms')
        cost = decision["estimated"]["cost_usd"]
        cols[3].metric("Synthetic estimated cost", f"${cost:.6f}" if cost is not None else "—")
        st.write("Inferred features", decision["features"])
        st.write("Eligible models", decision["eligible_models"])
        st.write("Controlled reasons", decision["reason_codes"])
        st.dataframe(pd.DataFrame(decision["rejected_models"]), hide_index=True)
        st.write("Score breakdown", decision["score_breakdown"])
        st.dataframe(pd.DataFrame(decision["alternatives"]), hide_index=True)
        with st.expander("Complete request and routing response"):
            st.json(st.session_state["request"])
            st.json(decision)
    with st.expander("Historical audit replay"):
        request_id = st.text_input("Request ID")
        if st.button("Replay original records"):
            st.json(router.audit.replay(request_id))


def executive_view(router):
    st.subheader("Executive / Research")
    st.info("SYNTHETIC DEMO VALUES. Quality proxies, prices, and model latencies are illustrative. Routing latency is measured locally.")
    benchmark_type = st.selectbox("Benchmark", list(BENCHMARK_NAMES), format_func=BENCHMARK_NAMES.get)
    st.caption(METHODOLOGY_NOTE)
    st.subheader("Synthetic Reference-Route Agreement")
    st.caption(AGREEMENT_NOTE)
    st.caption("Research Eligible means evidence about the stated local implementation under this synthetic benchmark. Jev mock is not evidence of real Jev performance. LIVE on an unavailable adapter is a reserved mode, not an API execution.")
    if st.button("Run local 100-request comparison"):
        from evaluation.benchmark import run_benchmark, train_ml
        with st.spinner("Training ML, checking splits, and comparing local routers..."):
            rows, labels = final_test_data(benchmark_type, 100)
            train_ml(router, benchmark_type, rows, labels)
            run_benchmark(rows, labels, router, ["frontier_only", "cheapest_only", "rules", "weighted", "ml", "jev"],
                          benchmark_type=benchmark_type)
    directories = report_directories(benchmark_type)
    if directories:
        directory = st.selectbox("Report run", directories, format_func=lambda p: str(p.relative_to(ROOT / "reports")) if p.is_relative_to(ROOT / "reports") else p.name)
        summary, details, manifest = load_report(directory)
        if manifest.get("legacy_report"):
            st.warning(manifest["provenance_note"])
        st.json({"Benchmark Type": manifest["benchmark_type"], "Evidence Type": manifest["evidence_type"],
                 "Dataset Size": manifest.get("test_request_count", int(summary.total_requests.max())),
                 "Dataset Hash": manifest.get("test_dataset_hash", manifest.get("dataset_hash")),
                 "Training Templates": manifest.get("training_template_ids", "Not recorded"),
                 "Held-Out Templates": manifest.get("test_template_ids", []) if benchmark_type == "TEMPLATE_HELD_OUT" else [],
                 "Training Seed": manifest.get("training_seed", "Not recorded"),
                 "Validation Seed": manifest.get("validation_seed", "Not recorded"),
                 "Test Seed": manifest.get("test_seed", manifest.get("random_seed")),
                 "ML Training Count (generated)": manifest.get("training_request_count_generated", "Not recorded"),
                 "ML Training Count (fitted)": manifest.get("training_request_count_fitted", "Not recorded"),
                 "ML Validation Count": manifest.get("validation_request_count", "Not recorded"),
                 "Final-Test Count": manifest.get("test_request_count", int(summary.total_requests.max()))})
        st.dataframe(display_summary(summary), hide_index=True)
        name = st.selectbox("Strategy for KPIs", summary.strategy)
        row = summary[summary.strategy == name].iloc[0]
        metrics = [("Total requests", row.total_requests), ("Policy violation rate", row.policy_violation_rate),
                   ("Average estimated cost", row.average_estimated_model_cost_usd), ("Average model latency (ms)", row.average_model_latency_ms),
                   ("Acceptable Reference-Route Agreement — Including Correct Abstentions", row.acceptable_reference_route_agreement_including_correct_abstentions), ("No-route rate", row.no_route_rate),
                   ("Fallback rate", row.fallback_rate), ("Frontier usage", row.frontier_model_usage), ("Small-model usage", row.small_model_usage)]
        columns = st.columns(3)
        for index, (label, value) in enumerate(metrics):
            columns[index % 3].metric(label, "N/A" if pd.isna(value) else f"{value:.5g}")
        chart_data = summary.set_index("strategy")
        for label, column in [("Synthetic inference cost", "average_estimated_model_cost_usd"),
                              ("Synthetic model p95 latency", "average_model_latency_ms"),
                              ("Acceptable Reference-Route Agreement — Including Correct Abstentions", "acceptable_reference_route_agreement_including_correct_abstentions"),
                              ("Synthetic quality proxy", "synthetic_quality_proxy")]:
            st.write(label)
            st.bar_chart(chart_data[[column]])
        if summary.downstream_quality.notna().any():
            st.write("Measured downstream quality (methods may differ)")
            st.bar_chart(chart_data[["downstream_quality"]])
        st.write("Model distribution")
        st.bar_chart(details.groupby(["selected_model", "strategy"]).size().unstack(fill_value=0))
        stability_path = directory / "stability_results.csv"
        if stability_path.exists():
            st.write("Paraphrase route stability")
            st.dataframe(pd.read_csv(stability_path).groupby("strategy")[["stable", "both_routed"]].mean())
    else:
        st.write("Run the benchmark to populate reports.")
    st.subheader("ILLUSTRATIVE SCENARIO ONLY")
    defaults = load_config("evaluation")["business_case"]
    annual = st.number_input("Annual requests", min_value=0, value=defaults["annual_requests"])
    current = st.number_input("Current average cost per request (USD)", min_value=0.0, value=defaults["current_average_cost_per_request"], format="%.5f")
    routed = st.number_input("Router average cost per request (USD)", min_value=0.0, value=defaults["router_average_cost_per_request"], format="%.5f")
    st.json(calculate(annual, current, routed))
    st.caption("Excludes routing API spend, integration costs and operating costs. This is not a real banking savings estimate.")


def live_view():
    st.subheader("Live / Research")
    st.info("Empirical live routing execution evaluated against a synthetic reference-routing objective.")
    st.caption("The reference labels remain synthetic even when the routing API execution is live. No selected downstream model is executed.")
    st.caption("BASELINE_REPLAY preserves frozen Rules, Weighted and ML decisions and their original latency. It is not a newly measured execution. MOCK is never evidence about real Jev performance.")
    from evaluation.live_reporting import live_report_manifests
    runs = live_report_manifests()
    if not runs:
        st.write("No live runs recorded. Start a pilot explicitly in the terminal with --live-pilot --allow-live-api.")
        return
    path = st.selectbox("Live run", runs, format_func=lambda p: p.parent.name)
    manifest = json.loads(path.read_text())
    st.json({key: manifest.get(key) for key in ["run_id", "status", "planned_request_counts", "planned_no_route_counts",
        "api_call_count", "failed_call_count", "retry_count", "routing_cost_usd", "known_routing_cost_usd",
        "unknown_cost_attempts", "router_models", "execution_mode_per_strategy", "research_eligible_per_strategy"]})
    summary_path = path.parent / "benchmark_summary.csv"
    if summary_path.exists():
        frame = pd.read_csv(summary_path)
        st.subheader("Synthetic Reference-Route Agreement")
        st.caption(AGREEMENT_NOTE)
        labels = {"strategy":"Strategy", "benchmark_type":"Benchmark Type", "execution_mode":"Execution Mode",
            "research_eligible":"Research Eligible", "evidence_type":"Execution Evidence", "total_requests":"Requests",
            "preferred_reference_route_agreement_including_correct_abstentions":"Preferred Agreement — Including Correct Abstentions",
            "acceptable_reference_route_agreement_including_correct_abstentions":"Acceptable Agreement — Including Correct Abstentions",
            "preferred_reference_route_agreement_routed_only":"Preferred Agreement — Routed Only",
            "acceptable_reference_route_agreement_routed_only":"Acceptable Agreement — Routed Only",
            "p50_routing_latency_ms":"p50 Routing Latency (ms)", "p95_routing_latency_ms":"p95 Routing Latency (ms)",
            "latency_measurement_source":"Latency Source", "routing_cost_usd":"Routing API Cost (USD)",
            "cost_status":"Cost Status", "pricing_version":"Pricing Version",
            "total_input_tokens":"Input Tokens", "total_cached_input_tokens":"Cached Input Tokens",
            "total_output_tokens":"Output Tokens", "unknown_cost_decisions":"Decisions With Unknown Cost",
            "failure_rate":"Failure Rate", "ineligible_selection_attempts":"Blocked Ineligible Selections"}
        st.dataframe(frame[[k for k in labels if k in frame]].rename(columns=labels), hide_index=True)
        st.caption("Unknown API charges remain blank. The manifest cost ledger includes every attempt, including retries and force-rerun history; this table shows latest observations.")


def main():
    st.title("Smart Model Router for Banking")
    st.caption("Policy first · Transparent ranking · Historical audit · Reproducible experiments")
    # A session owns its router; configuration is reloaded on each rerun.
    router = ModelRouter()
    view = st.sidebar.radio("View", ["Developer / Architect", "Executive / Research", "Live / Research"])
    if view == "Developer / Architect":
        developer_view(router)
    elif view == "Executive / Research":
        executive_view(router)
    else:
        live_view()


if __name__ == "__main__":
    main()
