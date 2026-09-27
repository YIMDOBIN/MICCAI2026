"""CPU-only audit of the author's ToothFairy3 text reports and RDF labels."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from pyshacl import validate
from rdflib import Graph, Namespace, RDF, URIRef


DENTRO = Namespace("http://example.org/dentro#")
REPORT_RE = re.compile(r"^(ToothFairy3[FPS]_\d+)_\d+\.txt$")


def audit(upstream: Path) -> dict:
    reports_dir = upstream / "data/ToothFairy3/reportsTr"
    labels_dir = upstream / "data/ttl_reports"
    shapes_file = upstream / "ontology/shapes.ttl"
    ontology_file = upstream / "ontology/ontology.ttl"
    for path in (reports_dir, labels_dir, shapes_file, ontology_file):
        if not path.exists():
            raise FileNotFoundError(f"Required upstream path missing: {path}")

    groups: dict[str, list[str]] = defaultdict(list)
    unrecognized = []
    for path in sorted(reports_dir.glob("*.txt")):
        match = REPORT_RE.fullmatch(path.name)
        if match:
            groups[match.group(1)].append(path.name)
        else:
            unrecognized.append(path.name)

    label_paths = sorted(labels_dir.glob("*.ttl"))
    label_ids = {p.stem for p in label_paths}
    counts = Counter()
    problems = []
    ontology = Graph().parse(ontology_file, format="turtle")
    shapes = Graph().parse(shapes_file, format="turtle")
    for path in label_paths:
        try:
            graph = Graph().parse(path, format="turtle")
            findings = list(graph.subjects(RDF.type, DENTRO.Finding))
            for finding in findings:
                types = list(graph.objects(finding, DENTRO.findingType))
                for value in types:
                    counts[str(value).removeprefix(str(DENTRO))] += 1
            conforms, _, results_text = validate(
                graph, shacl_graph=shapes, ont_graph=ontology,
                inference="none", abort_on_first=False,
                allow_warnings=True, allow_infos=True,
            )
            # pySHACL can report conformance with warnings when allowed;
            # preserve its text so the reviewer can inspect all issues.
            if not conforms or "Validation Result" in results_text:
                problems.append({"case_id": path.stem, "kind": "shacl", "conforms": bool(conforms),
                                 "detail": results_text[:12000]})
        except Exception as exc:
            problems.append({"case_id": path.stem, "kind": "parse", "detail": str(exc)})

    missing_labels = sorted(set(groups) - label_ids)
    missing_reports = sorted(label_ids - set(groups))
    return {
        "upstream": str(upstream.resolve()),
        "report_files": sum(map(len, groups.values())),
        "report_patients": len(groups),
        "ttl_files": len(label_paths),
        "reports_per_patient": dict(sorted(Counter(map(len, groups.values())).items())),
        "finding_types": dict(sorted(counts.items())),
        "unrecognized_report_names": unrecognized,
        "patients_without_ttl": missing_labels,
        "ttl_without_reports": missing_reports,
        "label_problems": problems,
        "note": "Text-label audit only; CBCT images and model predictions were not evaluated.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/audit.json"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = audit(args.upstream)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary = {key: result[key] for key in ("report_files", "report_patients", "ttl_files", "reports_per_patient")}
    summary["label_problems"] = len(result["label_problems"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Full audit: {args.output}")
    if args.strict and (result["patients_without_ttl"] or result["ttl_without_reports"]
                        or result["unrecognized_report_names"] or any(
                            p["kind"] == "parse" or not p.get("conforms", False)
                            for p in result["label_problems"]
                        )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
