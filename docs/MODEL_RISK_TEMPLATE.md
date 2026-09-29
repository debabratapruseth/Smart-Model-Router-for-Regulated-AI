# Educational model/system risk assessment

This template is generic and is not a regulatory determination.

| Item | Owner's assessment / evidence |
|---|---|
| System name and version | Smart Model Router for Banking / ... |
| Accountable system owner | ... |
| Business and technical owners | ... |
| Independent reviewer | ... |
| Business purpose and intended users | ... |
| Methodology | Hard policy filter, candidate ranking, independent decision validation |
| Inputs and trust boundaries | Classification, region and PII provenance; prompt and attachment handling |
| Outputs and downstream use | Model choice, abstention, review, execution permissions |
| Training / labels | Provenance, seeds, split strategy, leakage assessment |
| Assumptions | Catalog accuracy, trusted metadata, representative workload |
| Limitations | Synthetic quality/price/latency; keyword features; no certified controls |
| Risk assessment | Data egress, misrouting, operational failure, invalid input, bias |
| Validation plan | Unit/adversarial tests; independent review; real held-out tasks |
| Performance thresholds | Zero observed policy violations; acceptable coverage; approved quality noninferiority margin |
| Policy constraints | Approved providers, models, classification, domain, modality, region |
| Change management | Approver, version/hash, review date, rollback artifacts |
| Monitoring | Drift, no-route rate, fallback, confidence, cost, latency, telemetry freshness |
| Fallback | Approved local route only; no relaxation of hard constraints |
| Override | No automated policy override; manual change must be independently approved and versioned |
| Human review | Ownership, trigger thresholds, evidence and turnaround |
| Audit trail | Access, retention, original snapshots, immutable archive requirements |
| Incident response | Escalation owner, containment, replay and remediation |
| Approval / review expiry | ... |

Evidence checklist: recorded dataset/config hashes, environment lock, test report,
paired benchmark analysis, failure examples, model/endpoint approval evidence,
and documented independent sign-off. A zero observed rate is not proof of zero future risk.
