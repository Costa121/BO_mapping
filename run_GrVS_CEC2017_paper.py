"""
CEC2017 实验脚本 - GrVS-Mapping-BO
=====================================

固定配置（与 Gr-VS 论文对齐）:
- 维度: D = 100
- 有效维度: idim = 5
- 迭代次数: 100
- 初始采样: 20

方法参数:
- GrVS-Mapping-BO: k = 5, d = 10, update_freq = 20
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

# ===== 导入 GrVS-Mapping-BO =====
from GrVS_Mapping_BO import run_grvs_mapping_bo


# ========== 论文模式固定配置 ==========
PAPER_CONFIG = {
    "D": 100,
    "idim": 5,
    "k_explicit": 5,
    "d_embed": 10,        # 2*idim
    "max_iters": 100,
    "init_n": 20,

    # GrVS-Mapping-BO 额外参数
    "update_freq": 20,    # K：每段 BO 轮数
    "eci_grid_size": 31,  # 1D 网格点数
    "rho_momentum": 0.5,
    "use_random_signs": True,
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


def run_grvs_mapping_experiments(func_indices, seeds):
    cfg = PAPER_CONFIG
    D = cfg["D"]

    results_dir = Path("results_cec2017_grvs_mapping_paper_mode")
    results_dir.mkdir(exist_ok=True)

    total = len(func_indices) * len(seeds)
    current = 0

    success, failed = [], []

    print("=" * 80)
    print("CEC2017 GrVS-Mapping-BO 实验（paper-mode）")
    print("=" * 80)
    print(f"config: D={D}, d={cfg['d_embed']}, k={cfg['k_explicit']}, iters={cfg['max_iters']}, init={cfg['init_n']}")
    print(f"       update_freq={cfg['update_freq']}, eci_grid_size={cfg['eci_grid_size']}, rho_momentum={cfg['rho_momentum']}")
    print(f"results: {results_dir}")
    print("=" * 80)

    for func_idx in func_indices:
        func_info = CEC2017_CONFIG[func_idx]
        func = create_cec2017_objective(func_idx, D)

        print(f"\n{'='*80}")
        print(f"CEC2017 F{func_idx}: {func_info['name']} ({func_info['type']})  global_opt={func.global_opt}")
        print("="*80)

        for seed in seeds:
            current += 1
            print(f"\n[{current}/{total}] GrVS-Mapping-BO | F{func_idx} | seed={seed}")

            try:
                start = time.time()

                X, y = run_grvs_mapping_bo(
                    f=func,
                    D=D,
                    d=cfg["d_embed"],
                    k_explicit=cfg["k_explicit"],
                    init_n=cfg["init_n"],
                    max_iters=cfg["max_iters"],
                    update_freq=cfg["update_freq"],
                    eci_grid_size=cfg["eci_grid_size"],
                    rho_momentum=cfg["rho_momentum"],
                    use_random_signs=cfg["use_random_signs"],
                    seed=seed,
                    save_results=False,
                    task_id_prefix=f"paper_grvs_mapping_F{func_idx}_s{seed}",
                )

                duration = time.time() - start
                y = np.asarray(y, dtype=float)
                convergence = np.minimum.accumulate(y).tolist()

                best = float(y.min())
                regret = best - float(func.global_opt)

                result = {
                    "algorithm": "GrVS-Mapping-BO",
                    "function": f"F{func_idx}",
                    "function_name": func_info["name"],
                    "function_type": func_info["type"],
                    "seed": seed,
                    "dimension": D,
                    "idim": cfg["idim"],
                    "k_explicit": cfg["k_explicit"],
                    "d_embed": cfg["d_embed"],
                    "global_opt": func.global_opt,
                    "convergence_curve": convergence,
                    "final_value": best,
                    "final_regret": regret,
                    "duration": duration,
                    "config": cfg,
                }

                out = results_dir / f"GrVS-Mapping-BO_F{func_idx}_seed{seed}.json"
                with open(out, "w") as f:
                    json.dump(result, f, indent=2)

                success.append(result)
                print(f"  ✓ done  best={best:.2f}, regret={regret:.2f}, time={duration:.1f}s")

            except Exception as e:
                import traceback
                traceback.print_exc()
                failed.append({"algorithm": "GrVS-Mapping-BO", "function": f"F{func_idx}", "seed": seed, "error": str(e)})
                print(f"  ✗ failed: {e}")

    # 汇总
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({"success": success, "failed": failed}, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ finished: success={len(success)} failed={len(failed)}")
    print(f"summary: {summary_path}")
    print("=" * 80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run GrVS-Mapping-BO on CEC2017 (paper-mode config)")
    parser.add_argument("--funcs", nargs="+", type=int, default=[1, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    run_grvs_mapping_experiments(args.funcs, args.seeds)


if __name__ == "__main__":
    main()