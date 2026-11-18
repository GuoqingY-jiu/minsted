"""Paired continuous-circle MINSTED validation with time and photon budgets.

No true-position-based filtering is performed. The original estimator and the
Bayesian replay share exactly the same recorded circular scans/photon times.
The EIG policy has its own observations. Full traces permit resolution audits.
"""

import os

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "tmp" / "matplotlib"))

from simulators import CircularBayesianMinsted, CircularScan, MinstedSimulator
from simulators.CircularBayesianMinsted import next_circular_photon


def make_posterior(config, initial, **overrides):
    options = dict(
        prior_mean=initial, prior_sigma=config["prior_sigma"],
        brightness=config["brightness"], bg_global=config["background"],
        grid_size=config["grid_size"], quadrature_order=config["quadrature_order"],
        planning_grid=config["planning_grid"], phase_bins=config["phase_bins"],
        width_levels=config["width_levels"],
    )
    options.update(overrides)
    return CircularBayesianMinsted(**options)


def original_scan(sim):
    return CircularScan(tuple(float(v) for v in sim.C_i), sim.R_i, sim.d_i, sim.f_scan)


def estimate_record(method, budget, truth, estimate, time_ms, photons, scan, covariance=None, reached=True):
    error = np.asarray(estimate) - truth
    return dict(
        method=method, budget=budget,
        estimate_nm=np.asarray(estimate).tolist(), error_nm=error.tolist(),
        radial_error_nm=float(np.linalg.norm(error)), time_ms=float(time_ms),
        photons=int(photons), budget_reached=bool(reached), readout_scan=asdict(scan),
        posterior_sigma_nm=None if covariance is None else float(np.sqrt(np.trace(covariance) / 2)),
    )


def simulate_policy(config, truth, initial, rng_seed, policy):
    posterior = make_posterior(config, initial)
    original = MinstedSimulator(d0=250., d_min=40., r_min=5., bg_global=config["background"])
    original.reset_tracker(initial)
    rng = np.random.default_rng(rng_seed)
    planning_seconds = 0.
    start = perf_counter()
    if policy == "original":
        scan = original_scan(original)
    else:
        plan_start = perf_counter()
        scan = posterior.select_next_scan()
        planning_seconds += perf_counter() - plan_start
    events, checkpoints = [], []
    completed = set()
    count = 0
    time_budget = config["time_ms"]
    photon_budget = config["photons"]

    def capture(budget, at_time, active_scan, event_readout=False, reached=True):
        if event_readout:
            mean, covariance = posterior.get_localization_estimate(), posterior.get_covariance()
        else:
            mean, covariance = posterior.preview_no_photon(active_scan, at_time)
        method = "bayes_replay" if policy == "original" else "bayes_eig_circle"
        checkpoints.append(estimate_record(method, budget, truth, mean, at_time, count, active_scan, covariance, reached))
        if policy == "original":
            checkpoints.append(estimate_record("original", budget, truth, original.get_localization_estimate(), at_time, count, active_scan, reached=reached))
        completed.add(budget)

    while len(completed) < 2:
        arrival = next_circular_photon(scan, truth, posterior.time_ms, rng, config["brightness"], config["background"])
        if "time" not in completed and arrival > time_budget:
            capture("time", time_budget, scan)
        if len(completed) == 2:
            break
        if arrival > config["max_time_ms"]:
            if "photons" not in completed:
                capture("photons", config["max_time_ms"], scan, reached=False)
            break
        posterior.observe_scan_interval(scan, arrival, photon=True)
        count += 1
        events.append(dict(time_ms=float(arrival), scan=asdict(scan)))
        if policy == "original":
            original.register_one_photon(scan.positions(arrival))
        if count == photon_budget:
            capture("photons", arrival, scan, event_readout=True)
        if "time" not in completed and arrival == time_budget:
            capture("time", arrival, scan, event_readout=True)
        if len(completed) == 2:
            break
        if policy == "original":
            scan = original_scan(original)
        else:
            plan_start = perf_counter()
            scan = posterior.select_next_scan(scan)
            planning_seconds += perf_counter() - plan_start
    return dict(
        policy=policy, events=events, checkpoints=checkpoints,
        compute_s=perf_counter() - start, planning_s=planning_seconds,
        total_pruned_mass=posterior.total_pruned_mass,
    )


def run_trial(task):
    config, scenario, run = task
    scenario_id = {"fixed_offset": 0, "prior_matched": 1}[scenario]
    truth_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], scenario_id, run, 91]))
    if scenario == "fixed_offset":
        truth, initial = np.array([10., -5.]), np.array([-50., 40.])
    else:
        initial = np.zeros(2)
        truth = truth_rng.normal(0., config["prior_sigma"], 2)
    # Common random stream couples simulations without treating their events as
    # the same data: each policy has a different true time-dependent intensity.
    seed = np.random.SeedSequence([config["seed"], scenario_id, run, 17])
    return dict(
        scenario=scenario, run=run, truth_nm=truth.tolist(), initial_nm=initial.tolist(),
        paths=[simulate_policy(config, truth, initial, seed, policy) for policy in ["original", "eig"]],
    )


def collect_rows(trials):
    rows = []
    for trial in trials:
        for path in trial["paths"]:
            for record in path["checkpoints"]:
                error = record["error_nm"]
                rows.append(dict(
                    scenario=trial["scenario"], run=trial["run"], method=record["method"], budget=record["budget"],
                    true_x_nm=trial["truth_nm"][0], true_y_nm=trial["truth_nm"][1],
                    estimate_x_nm=record["estimate_nm"][0], estimate_y_nm=record["estimate_nm"][1],
                    error_x_nm=error[0], error_y_nm=error[1], radial_error_nm=record["radial_error_nm"],
                    time_ms=record["time_ms"], photons=record["photons"], budget_reached=record["budget_reached"],
                    posterior_sigma_nm=record["posterior_sigma_nm"],
                ))
    return rows


def summarize(rows, config):
    summary = []
    for scenario in ["fixed_offset", "prior_matched"]:
        for budget in ["time", "photons"]:
            for method in ["original", "bayes_replay", "bayes_eig_circle"]:
                group = [r for r in rows if r["scenario"] == scenario and r["budget"] == budget and r["method"] == method]
                if not group:
                    continue
                errors = np.array([[r["error_x_nm"], r["error_y_nm"]] for r in group])
                radial = np.linalg.norm(errors, axis=1)
                bias = errors.mean(axis=0)
                failures = np.array([not r["budget_reached"] for r in group]) | (radial > config["failure_nm"]) | ~np.isfinite(radial)
                summary.append(dict(
                    scenario=scenario, budget=budget, method=method, repeats=len(group),
                    rmse_single_axis_nm=float(np.sqrt(np.mean(errors**2))),
                    bias_x_nm=float(bias[0]), bias_y_nm=float(bias[1]),
                    bias_single_axis_nm=float(np.linalg.norm(bias) / np.sqrt(2)),
                    spread_single_axis_nm=float(np.sqrt(np.mean(np.var(errors, axis=0)))),
                    median_radial_error_nm=float(np.median(radial)), p95_radial_error_nm=float(np.quantile(radial, .95)),
                    failure_rate=float(failures.mean()), error_gt_5nm_rate=float(np.mean(radial > 5.)),
                    mean_photons=float(np.mean([r["photons"] for r in group])),
                    mean_time_ms=float(np.mean([r["time_ms"] for r in group])),
                    unmet_budget_count=sum(not r["budget_reached"] for r in group),
                ))
    return summary


def paired_intervals(rows, seed):
    result = []
    rng = np.random.default_rng(seed)
    for scenario in ["fixed_offset", "prior_matched"]:
        for budget in ["time", "photons"]:
            groups = {method: sorted([r for r in rows if r["scenario"] == scenario and r["budget"] == budget and r["method"] == method], key=lambda r: r["run"])
                      for method in ["original", "bayes_replay", "bayes_eig_circle"]}
            for reference, comparison in [("original", "bayes_replay"), ("bayes_replay", "bayes_eig_circle"), ("original", "bayes_eig_circle")]:
                left, right = groups[reference], groups[comparison]
                a = np.array([r["radial_error_nm"]**2 for r in left])
                b = np.array([r["radial_error_nm"]**2 for r in right])
                assert [r["run"] for r in left] == [r["run"] for r in right]
                indices = rng.integers(0, len(a), size=(4000, len(a)))
                ratios = np.sqrt(b[indices].mean(axis=1) / np.maximum(a[indices].mean(axis=1), 1e-300))
                ci = np.quantile(ratios, [.025, .975])
                result.append(dict(
                    scenario=scenario, budget=budget, reference=reference, comparison=comparison,
                    rmse_ratio=float(np.sqrt(b.mean() / max(a.mean(), 1e-300))),
                    bootstrap_95_low=float(ci[0]), bootstrap_95_high=float(ci[1]),
                ))
    return result


def replay_path(trial, path, checkpoint, config, **overrides):
    posterior = make_posterior(config, trial["initial_nm"], **overrides)
    consumed = 0
    for event in path["events"]:
        if checkpoint["budget"] == "time" and event["time_ms"] > config["time_ms"]:
            break
        scan = CircularScan(**event["scan"])
        posterior.observe_scan_interval(scan, event["time_ms"])
        consumed += 1
        if checkpoint["budget"] == "photons" and consumed == checkpoint["photons"]:
            break
    if checkpoint["time_ms"] > posterior.time_ms:
        return posterior.preview_no_photon(CircularScan(**checkpoint["readout_scan"]), checkpoint["time_ms"])[0]
    return posterior.get_localization_estimate()


def resolution_audit(trials, config, cases, planning_only=False):
    comparisons, planning = [], []
    for scenario in ["fixed_offset", "prior_matched"]:
        subset = sorted([t for t in trials if t["scenario"] == scenario], key=lambda t: t["run"])[:cases]
        for trial in subset:
            for path in ([] if planning_only else trial["paths"]):
                for cp in path["checkpoints"]:
                    if cp["method"] == "original":
                        continue
                    reference = replay_path(trial, path, cp, config, grid_size=129, quadrature_order=48)
                    change = float(np.linalg.norm(reference - np.asarray(cp["estimate_nm"])))
                    comparisons.append(dict(scenario=scenario, run=trial["run"], method=cp["method"], budget=cp["budget"], estimate_change_nm=change))
            if trial["run"] == 0:
                posterior = make_posterior(config, trial["initial_nm"])
                path = trial["paths"][1]
                for index in range(min(len(path["events"]), 101)):
                    if index in [0, 10, 50, 100]:
                        current = None if index == 0 else CircularScan(**path["events"][index - 1]["scan"])
                        scans = posterior.circular_candidates(current)
                        selected_action = CircularScan(**path["events"][index]["scan"])
                        old_bins, old_grid = posterior.phase_bins, posterior.planning_grid
                        posterior.phase_bins, posterior.planning_grid = 64, 33
                        fine = posterior.first_arrival_eig(scans)
                        posterior.phase_bins, posterior.planning_grid = old_bins, old_grid
                        def key(scan):
                            return (*np.round(scan.center_nm, 8), round(scan.radius_nm, 8), round(scan.fwhm_nm, 8))
                        selected = [key(scan) for scan in scans].index(key(selected_action))
                        regret = (fine.max() - fine[selected]) / max(fine.max(), 1e-300)
                        planning.append(dict(scenario=scenario, photons=index, relative_eig_regret=float(regret)))
                    event = path["events"][index]
                    posterior.observe_scan_interval(CircularScan(**event["scan"]), event["time_ms"])
    return dict(posterior_replays=comparisons, planning_comparisons=planning)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, output_dir, failure_nm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    names = {"original": "Original estimator / original circle", "bayes_replay": "Bayes estimator / SAME record", "bayes_eig_circle": "Bayes estimator / EIG circle"}
    for row_index, scenario in enumerate(["fixed_offset", "prior_matched"]):
        for col_index, budget in enumerate(["time", "photons"]):
            ax = axes[row_index, col_index]
            for method, label in names.items():
                radial = sorted(r["radial_error_nm"] for r in rows if r["scenario"] == scenario and r["budget"] == budget and r["method"] == method)
                ax.step(np.maximum(radial, .01), np.arange(1, len(radial) + 1) / len(radial), where="post", label=label)
            ax.axvline(failure_nm, color="gray", linestyle=":", label=f"Failure threshold {failure_nm:g} nm")
            ax.set(xscale="log", xlabel="Radial localization error (nm); all trials", ylabel="Empirical cumulative probability", title=f"{scenario} | {budget} budget", ylim=(0, 1.01))
            ax.grid(alpha=.25)
            ax.legend(fontsize=7)
    fig.suptitle("Continuous circular MINSTED: paired repeated-trial validation")
    fig.savefig(output_dir / "circular_validation.png", dpi=180)
    plt.close(fig)


def write_report(output_dir, config, summary, intervals, audit, wall_seconds):
    text = ["# 连续圆周 MINSTED 贝叶斯验证", "", "本报告由 validate_circular_bayesian.py 生成，全部样本保留。", "",
            f"每种初始条件 {config['repeats']} 次重复；时间预算 {config['time_ms']} ms；光子预算 {config['photons']} 个总检测光子（含背景）；失败预定义为径向误差 > {config['failure_nm']} nm、非有限估计或未达预算。", "",
            "fixed_offset：真值 [10,-5] nm，初始中心 [-50,40] nm；prior_matched：初始中心 [0,0]，真值从单轴标准差 60 nm 的二维高斯先验抽取。两种条件均使用相同先验宽度、已知亮度和背景。", "",
            "original 与 bayes_replay 使用完全相同的扫描控制和到达时间；bayes_eig_circle 保持连续圆周扫描并逐光子更新控制，其光子记录不同。采集时间不含 CPU 计算和硬件切换开销。", "",
            "| 条件 | 预算 | 方法 | 单轴等效 RMSE (nm) | bias x/y (nm) | 失败率 | 平均光子 | 平均时间 (ms) |", "|---|---|---|---:|---|---:|---:|---:|"]
    for row in summary:
        text.append(f"| {row['scenario']} | {row['budget']} | {row['method']} | {row['rmse_single_axis_nm']:.3f} | {row['bias_x_nm']:.3f}/{row['bias_y_nm']:.3f} | {row['failure_rate']:.1%} | {row['mean_photons']:.1f} | {row['mean_time_ms']:.3f} |")
    text += ["", "RMSE = sqrt(mean((dx²+dy²)/2))；bias 是跨重复的平均坐标误差；spread 使用总体方差，因此 RMSE²=spread²+bias_single_axis²。", "", "## 配对 bootstrap", "", "下表为 comparison/reference 的 RMSE 比值；95% 区间按运行编号配对重采样 4000 次。区间跨 1 时，本批数据不足以确认该项改善。", "", "| 条件 | 预算 | comparison / reference | 比值 | 95% 区间 |", "|---|---|---|---:|---|"]
    for row in intervals:
        text.append(f"| {row['scenario']} | {row['budget']} | {row['comparison']} / {row['reference']} | {row['rmse_ratio']:.3f} | [{row['bootstrap_95_low']:.3f}, {row['bootstrap_95_high']:.3f}] |")
    changes = [r["estimate_change_nm"] for r in audit["posterior_replays"]]
    regrets = [r["relative_eig_regret"] for r in audit["planning_comparisons"]]
    text += ["", "## 数值近似核查", "", f"后验使用 {config['grid_size']}×{config['grid_size']} 自适应网格、{config['quadrature_order']} 点残余弧高斯积分，整圈积分解析计算。首批轨迹用 129×129 网格及 48 点积分重放：{len(changes)} 个预算读数，估计位置变化中位数 {np.median(changes):.4f} nm，最大值 {max(changes):.4f} nm。" if changes else "未执行高分辨率重放。",
             f"规划默认 {config['planning_grid']}×{config['planning_grid']} 后验分块、每圈 {config['phase_bins']} 个相位箱。在抽查状态上，改用 33×33 分块、64 相位箱重新评分，默认所选动作的相对 EIG 损失最大 {max(regrets):.2%}。" if regrets else "未执行规划分辨率抽查。", "",
             "宽后验阶段会对粗筛前 12 名及每个宽度的最佳候选，用至少 33×33 分块和 32 相位箱复评分。规划抽查比较的是轨迹中实际执行的动作，并将当时已有的扫描中心纳入候选。", "",
             "这只检查指定轨迹与状态，不代表所有条件的数值收敛。候选动作范围也有限，不保证连续动作空间的最优。", "",
             "## EIG 的准确含义与边界", "", "控制器评价未来一圈内的首光子到达相位箱或无光子结果的互信息。它是固定时间窗、相位量化的首到达 EIG；不是整条多光子路径的完整 EIG，也不是专门按固定光子预算优化。实际后验使用连续到达时间及完整等待区间的点过程似然。", "",
             "中心候选为后验均值及沿协方差主轴的 ±0.5 标准差偏移，并含当前中心；宽度在 40–250 nm 几何取值；半径为 .25/.5/.75 倍宽度并限制到 5–125 nm；相位连续，控制在光子处更新。候选和数值分辨率仍是可配置近似。", "",
             "假设静态单分子、已知恒定亮度/均匀背景、理想高斯 E-PSF、瞬时控制切换；未建模漂白、闪烁、运动、强度变化的光损伤或实际反馈延迟。局部 EIG 不保证最终 RMSE 最优。bootstrap 区间仅反映本批独立重复的采样波动，无法排除系统性模型或数值偏差。", "",
             f"本次总运行时间 {wall_seconds:.1f} s。完整配置见 config.json；逐试验结果见 trials.csv；全部控制与光子记录见 trial_*.json。", ""]
    (output_dir / "report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--time-ms", type=float, default=6.)
    parser.add_argument("--photons", type=int, default=100)
    parser.add_argument("--grid-size", type=int, default=97)
    parser.add_argument("--planning-grid", type=int, default=17)
    parser.add_argument("--phase-bins", type=int, default=16)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--width-levels", type=int, default=5)
    parser.add_argument("--audit-cases", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "examples" / "circular_validation")
    args = parser.parse_args()
    if args.repeats < 1 or args.workers < 1 or args.photons < 1 or args.time_ms <= 0 or args.audit_cases < 0:
        parser.error("Budgets, workers and repeats must be positive; audit-cases nonnegative.")
    config = dict(
        repeats=args.repeats, seed=args.seed, time_ms=args.time_ms, photons=args.photons,
        grid_size=args.grid_size, planning_grid=args.planning_grid, phase_bins=args.phase_bins,
        quadrature_order=args.quadrature_order, width_levels=args.width_levels,
        prior_sigma=60., brightness=30., background=3., failure_nm=20., max_time_ms=max(100., args.time_ms),
        workers=args.workers, audit_cases=args.audit_cases,
        sampler="exact_thinning", eig_objective="one_revolution_censored_first_arrival",
        refinement_candidates=12, refinement_grid=33, refinement_phase_bins=32,
        d0_nm=250., d_min_nm=40., min_radius_nm=5., max_radius_nm=125., frequency_khz=125.,
        original_alpha=.15, original_gamma=.97,
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
    )
    make_posterior(config, [0., 0.])
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    tasks = [(config, scenario, run) for scenario in ["fixed_offset", "prior_matched"] for run in range(args.repeats)]
    trials = []
    start = perf_counter()
    if args.workers == 1:
        iterator = map(run_trial, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = [executor.submit(run_trial, task) for task in tasks]
        iterator = (future.result() for future in as_completed(futures))
    try:
        for trial in iterator:
            trials.append(trial)
            path = output_dir / f"trial_{trial['scenario']}_{trial['run']:04d}.json"
            path.write_text(json.dumps(trial), encoding="utf-8")
            if len(trials) % 10 == 0 or len(trials) == len(tasks):
                print(f"Completed {len(trials)}/{len(tasks)} paired trials; wall={perf_counter()-start:.1f}s", flush=True)
    finally:
        if executor:
            executor.shutdown(wait=True)
    trials.sort(key=lambda trial: (trial["scenario"], trial["run"]))
    rows = collect_rows(trials)
    summary = summarize(rows, config)
    intervals = paired_intervals(rows, args.seed)
    print('Running higher-resolution record replay and planning audit...', flush=True)
    audit = resolution_audit(trials, config, args.audit_cases)
    wall = perf_counter() - start
    write_csv(output_dir / "trials.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    (output_dir / "paired_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    (output_dir / "resolution_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_report(output_dir, config, summary, intervals, audit, wall)
    make_plot(rows, output_dir, config["failure_nm"])
    for record in summary:
        print(f"{record['scenario']:13} {record['budget']:7} {record['method']:16} RMSE={record['rmse_single_axis_nm']:.3f} nm bias=({record['bias_x_nm']:.3f},{record['bias_y_nm']:.3f}) failure={record['failure_rate']:.1%}")
    print(f"Saved validation report: {output_dir / 'report.md'}; wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
