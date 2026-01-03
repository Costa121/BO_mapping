"""
CEC2017 实验脚本 - GrVS 变体对比
=====================================

对比方法:
1. GrVS-TopK-TailFilter-BO: Top-k + TailFilter + 方向符号
2. GrVS-NoExplicit-TailFilter-BO: No-Explicit + TailFilter + 方向符号

固定配置（与 Gr-VS 论文对齐）:
- 维度: D = 100
- 有效维度: idim = 5
- 迭代次数: 100
- 初始采样: 20
"""
import json
import time
from pathlib import Path
import sys
import numpy as np

# ===== 路径配置 =====
project_root = Path(__file__).parent.parent  # saasbo-main
hdbo_root = project_root / "hdbo-test-main"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(hdbo_root))
sys.path.insert(0, str(project_root / "GrVS_mapping"))

# ===== 导入 CEC2017 函数 =====
from cec2017.functions import all_functions

# ===== 导入两种新方法 =====
from GrVS_TopK_TailFilter_BO import run_grvs_topk_tailfilter_bo
from GrVS_NoExplicit_TailFilter_BO import run_grvs_noexplicit_tailfilter_bo


# ========== 论文模式固定配置 ==========
PAPER_CONFIG = {
    "D": 100,
    "idim": 5,
    "k_explicit": 5,
    "d_embed": 10,        # 2*idim
    "max_iters": 100,
    "init_n": 20,

    # GrVS 参数
    "update_freq": 20,    # K：每段 BO 轮数
    "eci_grid_size": 31,  # 1D 网格点数
    "rho_momentum": 0.5,
    
    # TailFilter 参数
    "tail_filter": True,
    "tail_filter_ratio": 1.5,
}

CEC2017_CONFIG = {
    1: {"name": "Bent_Cigar", "type": "unimodal"},
    2: {"name": "Sum_Diff_Power", "type": "unimodal"},
    3: {"name": "Zakharov", "type": "unimodal"},
    4: {"name": "Rosenbrock", "type": "multimodal"},
    5: {"name": "Rastrigin", "type": "multimodal"},
    6: {"name": "Schaffer", "type": "multimodal"},
    7: {"name": "Lunacek", "type": "multimodal"},
    8: {"name": "Non_Cont_Rastrigin", "type": "multimodal"},
    9: {"name": "Levy", "type": "multimodal"},
    10: {"name": "Schwefel", "type": "multimodal"},
    11: {"name": "Hybrid_1", "type": "hybrid"},
    12: {"name": "Hybrid_2", "type": "hybrid"},
    13: {"name": "Hybrid_3", "type": "hybrid"},
    14: {"name": "Hybrid_4", "type": "hybrid"},
    15: {"name": "Hybrid_5", "type": "hybrid"},
    16: {"name": "Hybrid_6", "type": "hybrid"},
    17: {"name": "Hybrid_7", "type": "hybrid"},
    18: {"name": "Hybrid_8", "type": "hybrid"},
    19: {"name": "Hybrid_9", "type": "hybrid"},
    20: {"name": "Hybrid_10", "type": "hybrid"},
    21: {"name": "Composition_1", "type": "composition"},
    22: {"name": "Composition_2", "type": "composition"},
    23: {"name": "Composition_3", "type": "composition"},
    24: {"name": "Composition_4", "type": "composition"},
    25: {"name": "Composition_5", "type": "composition"},
    26: {"name": "Composition_6", "type": "composition"},
    27: {"name": "Composition_7", "type": "composition"},
    28: {"name": "Composition_8", "type": "composition"},
    29: {"name": "Composition_9", "type": "composition"},
    30: {"name": "Composition_10", "type": "composition"},
}


def create_cec2017_objective(func_idx: int, D: int):
    """
    创建 CEC2017 目标函数包装器
      - BO/ECI/映射空间：x ∈ [-1,1]^D
      - 实际 CEC2017 输入：X = 100*x ∈ [-100,100]^D
    """
    cec_func = all_functions[func_idx - 1]
    func_info = CEC2017_CONFIG[func_idx]

    def objective(x):
        x = np.asarray(x, dtype=float).reshape(-1)
        x = np.clip(x, -1.0, 1.0)
        X = (x * 100.0).reshape(1, -1)  # [-1,1] → [-100,100]
        y = cec_func(X)
        return float(y[0])

    objective.func_name = f"F{func_idx}_{func_info['name']}"
    objective.func_idx = func_idx
    objective.func_type = func_info["type"]
    objective.D = D
    objective.global_opt = func_idx * 100

    return objective


def run_topk_tailfilter_experiments(func_indices, seeds, results_dir):
    """运行 GrVS-TopK-TailFilter-BO 实验"""
    cfg = PAPER_CONFIG
    D = cfg["D"]

    total = len(func_indices) * len(seeds)
    current = 0
    success, failed = [], []

    print("\n" + "=" * 80)
    print("GrVS-TopK-TailFilter-BO 实验")
    print("=" * 80)

    for func_idx in func_indices:
        func_info = CEC2017_CONFIG[func_idx]
        func = create_cec2017_objective(func_idx, D)

        for seed in seeds:
            current += 1
            print(f"\n[{current}/{total}] GrVS-TopK-TailFilter-BO | F{func_idx} | seed={seed}")

            try:
                start = time.time()

                X, y = run_grvs_topk_tailfilter_bo(
                    f=func,
                    D=D,
                    d=cfg["d_embed"],
                    k_explicit=cfg["k_explicit"],
                    init_n=cfg["init_n"],
                    max_iters=cfg["max_iters"],
                    update_freq=cfg["update_freq"],
                    eci_grid_size=cfg["eci_grid_size"],
                    rho_momentum=cfg["rho_momentum"],
                    tail_filter=cfg["tail_filter"],
                    tail_filter_ratio=cfg["tail_filter_ratio"],
                    seed=seed,
                    save_results=False,
                    task_id_prefix=f"topk_tailfilter_F{func_idx}_s{seed}",
                )

                duration = time.time() - start
                y = np.asarray(y, dtype=float)
                convergence = np.minimum.accumulate(y).tolist()

                best = float(y.min())
                regret = best - float(func.global_opt)

                result = {
                    "algorithm": "GrVS-TopK-TailFilter-BO",
                    "function": f"F{func_idx}",
                    "function_name": func_info["name"],
                    "function_type": func_info["type"],
                    "seed": seed,
                    "dimension": D,
                    "idim": cfg["idim"],
                    "k_explicit": cfg["k_explicit"],
                    "d_embed": cfg["d_embed"],
                    "tail_filter": cfg["tail_filter"],
                    "tail_filter_ratio": cfg["tail_filter_ratio"],
                    "global_opt": func.global_opt,
                    "convergence_curve": convergence,
                    "final_value": best,
                    "final_regret": regret,
                    "duration": duration,
                    "config": cfg,
                }

                out = results_dir / f"GrVS-TopK-TailFilter-BO_F{func_idx}_seed{seed}.json"
                with open(out, "w") as f:
                    json.dump(result, f, indent=2)

                success.append(result)
                print(f"  ✓ done  best={best:.2f}, regret={regret:.2f}, time={duration:.1f}s")

            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append({
                    "algorithm": "GrVS-TopK-TailFilter-BO",
                    "function": f"F{func_idx}",
                    "seed": seed,
                    "error": str(e)
                })
                print(f"  ✗ failed: {e}")

    return success, failed


def run_noexplicit_tailfilter_experiments(func_indices, seeds, results_dir):
    """运行 GrVS-NoExplicit-TailFilter-BO 实验"""
    cfg = PAPER_CONFIG
    D = cfg["D"]

    total = len(func_indices) * len(seeds)
    current = 0
    success, failed = [], []

    print("\n" + "=" * 80)
    print("GrVS-NoExplicit-TailFilter-BO 实验")
    print("=" * 80)

    for func_idx in func_indices:
        func_info = CEC2017_CONFIG[func_idx]
        func = create_cec2017_objective(func_idx, D)

        for seed in seeds:
            current += 1
            print(f"\n[{current}/{total}] GrVS-NoExplicit-TailFilter-BO | F{func_idx} | seed={seed}")

            try:
                start = time.time()

                X, y = run_grvs_noexplicit_tailfilter_bo(
                    f=func,
                    D=D,
                    d=cfg["d_embed"],
                    init_n=cfg["init_n"],
                    max_iters=cfg["max_iters"],
                    update_freq=cfg["update_freq"],
                    eci_grid_size=cfg["eci_grid_size"],
                    rho_momentum=cfg["rho_momentum"],
                    tail_filter=cfg["tail_filter"],
                    tail_filter_ratio=cfg["tail_filter_ratio"],
                    seed=seed,
                    save_results=False,
                    task_id_prefix=f"noexplicit_tailfilter_F{func_idx}_s{seed}",
                )

                duration = time.time() - start
                y = np.asarray(y, dtype=float)
                convergence = np.minimum.accumulate(y).tolist()

                best = float(y.min())
                regret = best - float(func.global_opt)

                result = {
                    "algorithm": "GrVS-NoExplicit-TailFilter-BO",
                    "function": f"F{func_idx}",
                    "function_name": func_info["name"],
                    "function_type": func_info["type"],
                    "seed": seed,
                    "dimension": D,
                    "idim": cfg["idim"],
                    "k_explicit": 0,  # No-Explicit
                    "d_embed": cfg["d_embed"],
                    "tail_filter": cfg["tail_filter"],
                    "tail_filter_ratio": cfg["tail_filter_ratio"],
                    "global_opt": func.global_opt,
                    "convergence_curve": convergence,
                    "final_value": best,
                    "final_regret": regret,
                    "duration": duration,
                    "config": cfg,
                }

                out = results_dir / f"GrVS-NoExplicit-TailFilter-BO_F{func_idx}_seed{seed}.json"
                with open(out, "w") as f:
                    json.dump(result, f, indent=2)

                success.append(result)
                print(f"  ✓ done  best={best:.2f}, regret={regret:.2f}, time={duration:.1f}s")

            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append({
                    "algorithm": "GrVS-NoExplicit-TailFilter-BO",
                    "function": f"F{func_idx}",
                    "seed": seed,
                    "error": str(e)
                })
                print(f"  ✗ failed: {e}")

    return success, failed


def run_all_experiments(func_indices, seeds, methods=None):
    """运行所有实验"""
    cfg = PAPER_CONFIG
    
    results_dir = Path("results_cec2017_grvs_variants")
    results_dir.mkdir(exist_ok=True)

    if methods is None:
        methods = ["topk", "noexplicit"]

    print("=" * 80)
    print("CEC2017 GrVS 变体对比实验")
    print("=" * 80)
    print(f"配置: D={cfg['D']}, d={cfg['d_embed']}, k={cfg['k_explicit']}")
    print(f"      iters={cfg['max_iters']}, init={cfg['init_n']}, update_freq={cfg['update_freq']}")
    print(f"      tail_filter={cfg['tail_filter']}, tail_filter_ratio={cfg['tail_filter_ratio']}")
    print(f"函数: {func_indices}")
    print(f"种子: {seeds}")
    print(f"方法: {methods}")
    print(f"结果目录: {results_dir}")
    print("=" * 80)

    all_success = []
    all_failed = []

    # 1. TopK-TailFilter
    if "topk" in methods:
        success, failed = run_topk_tailfilter_experiments(func_indices, seeds, results_dir)
        all_success.extend(success)
        all_failed.extend(failed)

    # 2. NoExplicit-TailFilter
    if "noexplicit" in methods:
        success, failed = run_noexplicit_tailfilter_experiments(func_indices, seeds, results_dir)
        all_success.extend(success)
        all_failed.extend(failed)

    # 汇总
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"success": all_success, "failed": all_failed}, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ 实验完成: 成功={len(all_success)} 失败={len(all_failed)}")
    print(f"汇总文件: {summary_path}")
    print("=" * 80)

    # 打印简要结果对比
    print_results_summary(all_success)


def print_results_summary(results):
    """打印结果汇总"""
    if not results:
        return

    print("\n" + "=" * 80)
    print("结果汇总")
    print("=" * 80)

    # 按算法和函数分组
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    
    for r in results:
        alg = r["algorithm"]
        func = r["function"]
        grouped[func][alg].append(r["final_value"])

    # 打印表格
    algorithms = sorted(set(r["algorithm"] for r in results))
    functions = sorted(grouped.keys(), key=lambda x: int(x[1:]))

    # 表头
    header = f"{'Function':<10}"
    for alg in algorithms:
        short_name = alg.replace("GrVS-", "").replace("-BO", "")
        header += f"{short_name:<25}"
    print(header)
    print("-" * len(header))

    # 数据行
    for func in functions:
        row = f"{func:<10}"
        for alg in algorithms:
            if alg in grouped[func]:
                values = grouped[func][alg]
                mean_val = np.mean(values)
                std_val = np.std(values)
                row += f"{mean_val:.2e} ± {std_val:.2e}  "
            else:
                row += f"{'N/A':<25}"
        print(row)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GrVS variants on CEC2017")
    parser.add_argument("--funcs", nargs="+", type=int, default=[1, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--methods", nargs="+", type=str, default=["topk", "noexplicit"],
                        choices=["topk", "noexplicit"],
                        help="Methods to run: topk (GrVS-TopK-TailFilter-BO), noexplicit (GrVS-NoExplicit-TailFilter-BO)")
    args = parser.parse_args()

    run_all_experiments(args.funcs, args.seeds, args.methods)


if __name__ == "__main__":
    main()
