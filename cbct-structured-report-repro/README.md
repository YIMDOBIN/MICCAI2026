# ToothFairy3 structured CBCT report reproduction

This project audits the published labels for *Ontology-Grounded Structured Prediction for Dental CBCT Reporting* (MICCAI 2026) and prepares a reproducible training handoff. It does not train a model, generate a clinical report, or contain clinical data.

## Sources and scope

- [Author implementation](https://github.com/AImageLab-zip/CBCT-Report), pinned to commit `e809ec3b45a82ea6975d42fcfdcac85d865dcf11`.
- The author repository includes 893 Italian text reports and 529 patient-level Turtle labels. The paper describes 529 labelled CBCT volumes; **the imaging volumes themselves are separate downloads**.
- `ontology/ontology.ttl` defines the vocabulary; `ontology/shapes.ttl` defines SHACL constraints; `data/ttl_reports/*.ttl` are the already extracted labels. The author extraction code calls a paid API, so do not rerun it to reproduce the published label audit.
- ToothFairy4's [dataset page](https://ditto.ing.unimore.it/toothfairy4/) requires an account for image downloads. Respect its terms. Do not upload the clinical reports or volumes to this repository.

## Label audit (CPU only)

```bash
git clone https://github.com/AImageLab-zip/CBCT-Report.git upstream
git -C upstream checkout e809ec3b45a82ea6975d42fcfdcac85d865dcf11
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-audit.txt
python audit_labels.py --upstream upstream --output outputs/audit.json
```

The audit checks the report-to-patient mapping, Turtle parsing, ontology types, SHACL violations and warnings, and the number of findings by type. `outputs/` is ignored by Git. Add `--strict` to return nonzero if any patient is missing, reports fail to parse, or a SHACL violation occurs. Review the JSON before treating the labels as training ground truth; SHACL conformance does not establish correspondence to CBCT images.

At the pinned upstream commit, our audit found **893 reports, 529 patients and 529 TTL files**, with no missing patient-to-label matches. **40 TTL files fail SHACL validation** and another **14 conform with warnings**. The audit emits each diagnostic to `outputs/audit.json`; running with `--strict` is expected to exit 1 for this snapshot. This is a label/schema consistency result, not an image-based clinical accuracy result. Keep the original labels intact and resolve each violation with clinical review before using a corrected label set for model training.

## GPU training handoff

After independently obtaining authorized CBCT volumes, place them in the original author's expected `upstream/data/ToothFairy3/` layout. Verify the case IDs and volume orientation first. On a GPU system follow the author's README for VoxTell, volume preprocessing and training. Their `requirements.txt` includes a pinned VoxTell dependency via GitHub SSH and CUDA-specific PyTorch wheels; install these in a suitable environment. The author `ontology/extract.py` has an absolute developer-specific path; bypass it when using the included TTL labels. The training dataset only pairs cases with both preprocessed `.pt` tokens and `.ttl` labels; inspect that intersection before training. The current author `parse_ttl_to_findings` explicitly skips `ToothAbsence`, so clarify whether to preserve that exclusion before interpreting class-wise metrics.

```bash
cd upstream
python main.py preprocess --dataset-root data/ToothFairy3 --output-dir data/preprocessed_tokens
python -m src.scripts.histogram_ttl
python -m src.train --use_kfold false --max_epochs 5 --no_wandb
```

These commands are a handoff, not a claim of successful training or paper-metric replication. Record dataset version, scanner/site split, preprocessing settings, dependencies, seeds and exact upstream commit for the eventual run. Use patient-level separation and an external-center holdout when evaluating generalization.

## First research extension

Compare the original structured predictor with a lesion/tooth ROI model on BrightCT data, checking finding detection, tooth number, qualifiers, and unsupported findings separately. Document FOV and MAR/metal conditions. Clinicians should review outputs before use in any patient workflow.
