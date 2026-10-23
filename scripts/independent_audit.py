#!/usr/bin/env python3
"""Independent audit: recompute all metrics from raw run records, verify cohort integrity,
compare against stored tables and summary, and produce machine-readable audit results.

This script does NOT import any CropXAI analysis code (scripts/analyze_results.py, etc.)
to ensure truly independent verification. It only imports the core metric/provenance
modules that the runs themselves used.
"""
import sys, json, math, sqlite3, os
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results' / 'raw'
DB = ROOT / 'results' / 'experiment_db.sqlite'
PROCESSED = ROOT / 'results' / 'processed'
TABLES = ROOT / 'results' / 'tables'

PILLARS = ('performance', 'explanation', 'calibration', 'robustness', 'uncertainty', 'representation')

def load_raw_records():
    """Load all raw JSON run records."""
    records = []
    for p in sorted(RAW.glob('*.json')):
        r = json.loads(p.read_text())
        records.append(r)
    return records

def mean_ci(values, alpha=0.05):
    """Student-t CI for the mean of independent observations."""
    x = np.array(values, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return {'mean': m, 'std': None, 'ci95': None, 'n': n}
    sd = float(x.std(ddof=1))
    half = float(tdist.ppf(1 - alpha/2, n-1) * sd / math.sqrt(n))
    return {'mean': m, 'std': sd, 'ci95': [m - half, m + half], 'n': n}

def composite_score(pillars, weights=None):
    """Compute composite trust score from pillar values."""
    if weights is None:
        weights = {k: 1/6 for k in PILLARS}
    return sum(pillars[k] * weights[k] for k in PILLARS)

def audit_cohort(records, label):
    """Audit a set of records for cohort integrity."""
    issues = []
    models = defaultdict(list)
    for r in records:
        arch = r['model'].get('architecture', 'UNKNOWN')
        models[arch].append(r)

    audit_rows = []
    for arch, runs in sorted(models.items()):
        seeds_seen = set()
        for r in runs:
            rid = r['run_id']
            seed = r['seed']
            status = r['status']
            epochs_cfg = r['config'].get('epochs', '?')
            bands = r['config'].get('bands', [])
            n_channels = len(bands)
            input_rep = r['config'].get('input_representation', '?')
            purpose = r['config'].get('purpose', '?')

            # Check history epochs
            art_dir = ROOT / 'artifacts' / 'revision' / rid
            hist = art_dir / 'history.json'
            hist_epochs = None
            if hist.exists():
                hist_epochs = len(json.loads(hist.read_text()))

            # Check checkpoint
            ckpt = art_dir / 'checkpoint.pt'
            ckpt_exists = ckpt.exists()

            # Check predictions
            pred = art_dir / 'predictions.npz'
            pred_exists = pred.exists()

            # Metric availability
            metrics = r.get('metrics', {})
            has_accuracy = 'accuracy' in metrics and metrics['accuracy'] is not None
            has_pillars = 'pillars' in metrics and all(k in metrics.get('pillars', {}) for k in PILLARS)
            has_composite = 'composite_trust' in metrics and metrics['composite_trust'] is not None
            has_stability = 'stability_ssim' in metrics and metrics['stability_ssim'] is not None
            has_deletion = 'deletion_auc' in metrics and metrics['deletion_auc'] is not None

            # Check for duplicate seeds
            if seed in seeds_seen:
                issues.append(f"DUPLICATE SEED {seed} for {arch} in cohort {label}")
            seeds_seen.add(seed)

            # Check epoch consistency
            if hist_epochs is not None and hist_epochs != epochs_cfg:
                issues.append(f"EPOCH MISMATCH run {rid}: config={epochs_cfg}, history={hist_epochs}")

            # Check status
            if status != 'completed':
                issues.append(f"INCOMPLETE run {rid}: status={status}")

            audit_rows.append({
                'cohort': label,
                'model': arch,
                'run_id': rid,
                'seed': seed,
                'epochs_config': epochs_cfg,
                'epochs_actual': hist_epochs,
                'bands': bands,
                'n_channels': n_channels,
                'input_representation': input_rep,
                'purpose': purpose,
                'status': status,
                'checkpoint_exists': ckpt_exists,
                'predictions_exist': pred_exists,
                'has_accuracy': has_accuracy,
                'has_all_pillars': has_pillars,
                'has_composite': has_composite,
                'has_stability': has_stability,
                'has_deletion': has_deletion,
                'source_revision': r.get('source_revision', '?')[:16],
                'checkpoint_sha256': r['model'].get('checkpoint_sha256', 'NONE')[:16] if r['model'].get('checkpoint_sha256') else 'NONE',
            })

        n_seeds = len(seeds_seen)
        if n_seeds != 5:
            issues.append(f"SEED COUNT for {arch} in {label}: expected 5, found {n_seeds} ({sorted(seeds_seen)})")

    return audit_rows, issues

def recompute_metrics(records):
    """Recompute aggregated metrics from raw records."""
    # Group by (model, purpose, source_revision, cohort)
    groups = defaultdict(list)
    for r in records:
        if r['status'] != 'completed':
            continue
        key = (r['model']['architecture'],
               r['config'].get('purpose', ''),
               r['config'].get('cohort', ''),
               r.get('source_revision', ''))
        groups[key].append(r)

    results = {}
    for key, runs in sorted(groups.items()):
        model, purpose, cohort, src_rev = key
        if not purpose.endswith('benchmark'):
            continue

        metrics_agg = {}
        # Collect seed-level values
        seed_values = defaultdict(list)
        for r in runs:
            m = r['metrics']
            seed_values['accuracy'].append(m['accuracy'])
            seed_values['macro_f1'].append(m['macro_f1'])
            seed_values['ece'].append(m['ece'])
            seed_values['nll'].append(m['nll'])
            seed_values['brier'].append(m['brier'])
            seed_values['stability_ssim'].append(m['stability_ssim'])
            seed_values['deletion_auc'].append(m['deletion_auc'])
            seed_values['silhouette'].append(m['silhouette'])
            seed_values['composite_trust'].append(m['composite_trust'])

            p = m.get('pillars', {})
            for pillar in PILLARS:
                seed_values[f'pillar_{pillar}'].append(p.get(pillar))

            # Corruption accuracies
            for cr in m.get('corruptions', []):
                k = f"corruption_{cr['kind']}_{cr['severity']:g}"
                seed_values[k].append(cr['accuracy'])

        for metric_name, vals in seed_values.items():
            metrics_agg[metric_name] = mean_ci(vals)

        results[(model, cohort)] = {
            'model': model,
            'cohort': cohort,
            'purpose': purpose,
            'n_runs': len(runs),
            'seeds': sorted([r['seed'] for r in runs]),
            'run_ids': sorted([r['run_id'] for r in runs]),
            'metrics': metrics_agg,
        }

    return results

def compare_with_stored_table(recomputed, table_path):
    """Compare recomputed metrics with the stored table1.json."""
    if not table_path.exists():
        return [{'check': 'table_exists', 'passed': False, 'detail': 'table1.json not found'}]

    stored = json.loads(table_path.read_text())
    checks = []

    for row in stored:
        model = row['model']
        metric = row['metric']
        cohort = row.get('cohort', '')
        key = (model, cohort)

        if key not in recomputed:
            checks.append({'model': model, 'metric': metric, 'passed': False,
                           'detail': f'Model/cohort {key} not found in recomputed results'})
            continue

        rc = recomputed[key]['metrics']

        # Map table metric names to our internal names
        metric_map = {p: f'pillar_{p}' for p in PILLARS}
        metric_map['composite_trust'] = 'composite_trust'
        internal = metric_map.get(metric, metric)

        if internal not in rc:
            checks.append({'model': model, 'metric': metric, 'passed': False,
                           'detail': f'Metric {internal} not found in recomputed results'})
            continue

        r = rc[internal]
        mean_match = abs(r['mean'] - row['mean']) < 1e-10
        std_match = (r['std'] is None and row.get('std') is None) or \
                    (r['std'] is not None and row.get('std') is not None and abs(r['std'] - row['std']) < 1e-10)
        ci_match = True
        if r['ci95'] is not None and row.get('ci95') is not None:
            ci_match = abs(r['ci95'][0] - row['ci95'][0]) < 1e-10 and abs(r['ci95'][1] - row['ci95'][1]) < 1e-10
        n_match = r['n'] == row['n']

        passed = mean_match and std_match and ci_match and n_match
        detail = []
        if not mean_match:
            detail.append(f"mean: recomputed={r['mean']:.12f}, stored={row['mean']:.12f}, diff={abs(r['mean']-row['mean']):.2e}")
        if not std_match:
            detail.append(f"std: recomputed={r['std']}, stored={row.get('std')}")
        if not ci_match:
            detail.append(f"ci95: recomputed={r['ci95']}, stored={row.get('ci95')}")
        if not n_match:
            detail.append(f"n: recomputed={r['n']}, stored={row['n']}")

        checks.append({
            'model': model, 'metric': metric, 'passed': passed,
            'detail': '; '.join(detail) if detail else 'exact match'
        })

    return checks

def verify_composite_scores(records):
    """Verify that each record's composite_trust equals the weighted sum of its pillars."""
    issues = []
    for r in records:
        if r['status'] != 'completed':
            continue
        m = r.get('metrics', {})
        p = m.get('pillars', {})
        ct = m.get('composite_trust')
        if not p or ct is None:
            continue

        recomputed = sum(p.get(k, 0) * (1/6) for k in PILLARS)
        if abs(recomputed - ct) > 1e-12:
            issues.append({
                'run_id': r['run_id'],
                'stored_composite': ct,
                'recomputed_composite': recomputed,
                'diff': abs(recomputed - ct)
            })
    return issues

def verify_pillar_definitions(records):
    """Verify pillar values are correctly derived from base metrics."""
    issues = []
    for r in records:
        if r['status'] != 'completed':
            continue
        m = r.get('metrics', {})
        p = m.get('pillars', {})
        if not p:
            continue

        # Performance = macro_f1
        expected_perf = m.get('macro_f1')
        if expected_perf is not None and abs(p.get('performance', -1) - expected_perf) > 1e-12:
            issues.append(f"{r['run_id']}: performance pillar {p.get('performance')} != macro_f1 {expected_perf}")

        # Explanation = 1 - deletion_auc
        expected_expl = 1 - m.get('deletion_auc', 0) if m.get('deletion_auc') is not None else None
        if expected_expl is not None and abs(p.get('explanation', -1) - expected_expl) > 1e-12:
            issues.append(f"{r['run_id']}: explanation pillar {p.get('explanation')} != 1-deletion_auc {expected_expl}")

        # Calibration = 1 - ECE
        expected_cal = 1 - m.get('ece', 0) if m.get('ece') is not None else None
        if expected_cal is not None and abs(p.get('calibration', -1) - expected_cal) > 1e-12:
            issues.append(f"{r['run_id']}: calibration pillar {p.get('calibration')} != 1-ECE {expected_cal}")

        # Uncertainty = 1 - brier/2
        expected_unc = 1 - m.get('brier', 0)/2 if m.get('brier') is not None else None
        if expected_unc is not None and abs(p.get('uncertainty', -1) - expected_unc) > 1e-12:
            issues.append(f"{r['run_id']}: uncertainty pillar {p.get('uncertainty')} != 1-brier/2 {expected_unc}")

        # Representation = (silhouette+1)/2
        sil = m.get('silhouette')
        expected_rep = (sil + 1) / 2 if sil is not None else None
        if expected_rep is not None and abs(p.get('representation', -1) - expected_rep) > 1e-12:
            issues.append(f"{r['run_id']}: representation pillar {p.get('representation')} != (silhouette+1)/2 {expected_rep}")

        # Robustness = mean of corruption accuracies
        corr = m.get('corruptions', [])
        if corr:
            expected_rob = float(np.mean([c['accuracy'] for c in corr]))
            if abs(p.get('robustness', -1) - expected_rob) > 1e-10:
                issues.append(f"{r['run_id']}: robustness pillar {p.get('robustness')} != mean_corruption_acc {expected_rob}")

        # Check ranges [0, 1]
        for k in PILLARS:
            v = p.get(k)
            if v is not None and (v < -1e-12 or v > 1 + 1e-12):
                issues.append(f"{r['run_id']}: pillar {k}={v} outside [0,1]")

    return issues

def verify_db_consistency(records):
    """Check that SQLite revision_runs matches raw JSON files."""
    if not DB.exists():
        return [{'check': 'db_exists', 'passed': False, 'detail': 'experiment_db.sqlite not found'}]

    issues = []
    conn = sqlite3.connect(str(DB))
    cursor = conn.execute("SELECT run_id, status, sha256 FROM revision_runs")
    db_rows = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
    conn.close()

    for r in records:
        rid = r['run_id']
        if rid not in db_rows:
            issues.append(f"Run {rid} in raw JSON but not in SQLite revision_runs")
        else:
            db_status, db_sha = db_rows[rid]
            if db_status != r['status']:
                issues.append(f"Run {rid}: JSON status={r['status']}, DB status={db_status}")
    
    for rid in db_rows:
        if rid not in {r['run_id'] for r in records}:
            # Could be superseded runs
            pass  # OK, DB may contain more

    return issues

def check_fallback_values(records):
    """Check that no run contains the legacy fallback values 0.65 or 0.50."""
    issues = []
    for r in records:
        if r['status'] != 'completed':
            continue
        ct = r.get('metrics', {}).get('composite_trust')
        if ct is not None:
            if abs(ct - 0.65) < 1e-6:
                issues.append(f"Run {r['run_id']}: composite_trust = 0.65 (legacy fallback!)")
            if abs(ct - 0.50) < 1e-6:
                issues.append(f"Run {r['run_id']}: composite_trust = 0.50 (legacy fallback!)")

        p = r.get('metrics', {}).get('pillars', {})
        for k, v in p.items():
            if v is not None:
                if abs(v - 0.65) < 1e-6 or abs(v - 0.50) < 1e-6:
                    issues.append(f"Run {r['run_id']}: pillar {k} = {v} (possible legacy fallback)")
    return issues

def check_source_consistency(records):
    """Check that all completed benchmark runs share the same source revision."""
    revisions = defaultdict(set)
    for r in records:
        if r['status'] == 'completed' and r['config'].get('purpose', '').endswith('benchmark'):
            cohort = r['config'].get('cohort', 'unknown')
            revisions[cohort].add(r.get('source_revision', 'NONE'))

    issues = []
    for cohort, revs in revisions.items():
        if len(revs) > 1:
            issues.append(f"Cohort {cohort} has {len(revs)} distinct source revisions: {sorted(revs)}")
    return issues

def main():
    print("=" * 80)
    print("INDEPENDENT AUDIT — CropXAI Paper 445")
    print("=" * 80)

    # Load all raw records
    records = load_raw_records()
    print(f"\nTotal raw records: {len(records)}")

    # Classify records
    completed = [r for r in records if r['status'] == 'completed']
    failed = [r for r in records if r['status'] == 'failed']
    print(f"Completed: {len(completed)}, Failed: {len(failed)}")

    # Separate by cohort/purpose
    rgb_benchmark = [r for r in completed if r['config'].get('cohort') == 'rgb_imagenet'
                     and r['config'].get('purpose') == 'controlled_benchmark']
    ms6_benchmark = [r for r in completed if r['config'].get('cohort') == 'ms6_prithvi'
                     and r['config'].get('purpose') == 'multispectral_foundation_benchmark']
    smoke = [r for r in completed if r['config'].get('purpose') == 'real_data_smoke']
    other = [r for r in completed if r not in rgb_benchmark and r not in ms6_benchmark and r not in smoke]

    print(f"\nRGB benchmark runs: {len(rgb_benchmark)}")
    print(f"MS6/Prithvi benchmark runs: {len(ms6_benchmark)}")
    print(f"Smoke runs: {len(smoke)}")
    print(f"Other completed: {len(other)}")

    # ==================== COHORT AUDIT ====================
    print("\n" + "=" * 60)
    print("SECTION 1: EXPERIMENT COHORT AUDIT")
    print("=" * 60)

    rgb_rows, rgb_issues = audit_cohort(rgb_benchmark, 'rgb_imagenet')
    ms6_rows, ms6_issues = audit_cohort(ms6_benchmark, 'ms6_prithvi')

    all_rows = rgb_rows + ms6_rows
    all_cohort_issues = rgb_issues + ms6_issues

    for row in all_rows:
        print(f"  {row['cohort']:15s} | {row['model']:25s} | seed={row['seed']:5d} | "
              f"epochs={row['epochs_actual']}/{row['epochs_config']} | "
              f"status={row['status']:10s} | pillars={'YES' if row['has_all_pillars'] else 'NO':3s} | "
              f"ckpt={'Y' if row['checkpoint_exists'] else 'N'}")

    if all_cohort_issues:
        print(f"\n  COHORT ISSUES ({len(all_cohort_issues)}):")
        for issue in all_cohort_issues:
            print(f"    !! {issue}")
    else:
        print(f"\n  COHORT: NO ISSUES FOUND")

    # ==================== PILLAR DEFINITION VERIFICATION ====================
    print("\n" + "=" * 60)
    print("SECTION 2: PILLAR DEFINITION VERIFICATION")
    print("=" * 60)

    pillar_issues = verify_pillar_definitions(rgb_benchmark + ms6_benchmark)
    if pillar_issues:
        print(f"  PILLAR ISSUES ({len(pillar_issues)}):")
        for issue in pillar_issues:
            print(f"    !! {issue}")
    else:
        print("  All pillar definitions verified correct for all runs")

    # ==================== COMPOSITE SCORE VERIFICATION ====================
    print("\n" + "=" * 60)
    print("SECTION 3: COMPOSITE SCORE VERIFICATION")
    print("=" * 60)

    composite_issues = verify_composite_scores(rgb_benchmark + ms6_benchmark)
    if composite_issues:
        print(f"  COMPOSITE ISSUES ({len(composite_issues)}):")
        for issue in composite_issues:
            print(f"    !! {issue}")
    else:
        print("  All composite scores verified: T = sum(w_i * s_i) with w_i = 1/6")

    # ==================== FALLBACK VALUE CHECK ====================
    print("\n" + "=" * 60)
    print("SECTION 4: LEGACY FALLBACK VALUE CHECK")
    print("=" * 60)

    fallback_issues = check_fallback_values(rgb_benchmark + ms6_benchmark)
    if fallback_issues:
        print(f"  FALLBACK ISSUES ({len(fallback_issues)}):")
        for issue in fallback_issues:
            print(f"    !! {issue}")
    else:
        print("  No legacy fallback values (0.65, 0.50) detected in any benchmark run")

    # ==================== SOURCE CONSISTENCY ====================
    print("\n" + "=" * 60)
    print("SECTION 5: SOURCE REVISION CONSISTENCY")
    print("=" * 60)

    src_issues = check_source_consistency(records)
    if src_issues:
        for issue in src_issues:
            print(f"    !! {issue}")
    else:
        print("  All benchmark runs share a single source revision per cohort")

    # ==================== DB CONSISTENCY ====================
    print("\n" + "=" * 60)
    print("SECTION 6: DATABASE CONSISTENCY")
    print("=" * 60)

    db_issues = verify_db_consistency(records)
    if db_issues:
        for issue in db_issues:
            print(f"    !! {issue}")
    else:
        print("  Raw JSON records and SQLite revision_runs are consistent")

    # ==================== METRIC RECOMPUTATION ====================
    print("\n" + "=" * 60)
    print("SECTION 7: INDEPENDENT METRIC RECOMPUTATION")
    print("=" * 60)

    recomputed = recompute_metrics(records)
    for key, data in sorted(recomputed.items()):
        model, cohort = key
        m = data['metrics']
        print(f"\n  {model} ({cohort}): n={data['n_runs']}, seeds={data['seeds']}")
        for metric_name in ['accuracy', 'pillar_performance', 'pillar_explanation', 'pillar_calibration',
                            'pillar_robustness', 'pillar_uncertainty', 'pillar_representation', 'composite_trust']:
            if metric_name in m:
                r = m[metric_name]
                ci_str = f"[{r['ci95'][0]:.6f}, {r['ci95'][1]:.6f}]" if r['ci95'] else "N/A"
                print(f"    {metric_name:30s}: mean={r['mean']:.6f}  std={r['std']:.6f} if r['std'] else 'N/A'  CI={ci_str}")

    # ==================== TABLE COMPARISON ====================
    print("\n" + "=" * 60)
    print("SECTION 8: COMPARISON WITH STORED TABLE")
    print("=" * 60)

    table_checks = compare_with_stored_table(recomputed, TABLES / 'table1.json')
    passed = sum(1 for c in table_checks if c['passed'])
    failed_checks = [c for c in table_checks if not c['passed']]
    print(f"  {passed}/{len(table_checks)} table entries match exactly")
    if failed_checks:
        for c in failed_checks:
            print(f"    !! MISMATCH: {c['model']} / {c['metric']}: {c['detail']}")

    # ==================== SUMMARY ====================
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    total_issues = len(all_cohort_issues) + len(pillar_issues) + len(composite_issues) + \
                   len(fallback_issues) + len(src_issues) + len(db_issues) + len(failed_checks)

    print(f"  Total issues found: {total_issues}")
    print(f"  Cohort issues: {len(all_cohort_issues)}")
    print(f"  Pillar definition issues: {len(pillar_issues)}")
    print(f"  Composite score issues: {len(composite_issues)}")
    print(f"  Fallback value issues: {len(fallback_issues)}")
    print(f"  Source consistency issues: {len(src_issues)}")
    print(f"  Database consistency issues: {len(db_issues)}")
    print(f"  Table comparison failures: {len(failed_checks)}")

    # Save machine-readable audit
    audit_output = {
        'total_raw_records': len(records),
        'completed': len(completed),
        'failed': len(failed),
        'rgb_benchmark_runs': len(rgb_benchmark),
        'ms6_benchmark_runs': len(ms6_benchmark),
        'cohort_audit': all_rows,
        'cohort_issues': all_cohort_issues,
        'pillar_issues': pillar_issues,
        'composite_issues': composite_issues,
        'fallback_issues': fallback_issues,
        'source_issues': src_issues,
        'db_issues': db_issues,
        'table_comparison': table_checks,
        'total_issues': total_issues,
        'recomputed_metrics': {f"{k[0]}_{k[1]}": v for k, v in recomputed.items()},
    }
    out_path = ROOT / 'docs' / 'revision' / 'evidence' / 'independent_audit_results.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(audit_output, indent=2, default=str) + '\n')
    print(f"\n  Audit results saved to: {out_path}")

    return total_issues

if __name__ == '__main__':
    sys.exit(0 if main() == 0 else 1)
