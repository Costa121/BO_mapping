"""
CEC2017 全方法对比分析脚本
=====================================

对比方法:
1. GrVS-Mapping-BO (原版: Top-k + 轮询 + 随机符号)
2. GrVS-TopK-TailFilter-BO (Top-k + TailFilter + 方向符号)
3. GrVS-NoExplicit-TailFilter-BO (No-Explicit + TailFilter + 方向符号)
4. Baseline: Add-GP-UCB, REMBO, ALEBO, Gr-VS, BAxUS

生成:
1. 性能对比表格 (CSV, Excel, TXT)
2. 收敛曲线图 (每个函数单独 + 合并图)
3. 算法排名表
4. 统计检验结果
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
from scipy import stats

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


class AllMethodsAnalyzer:
    """CEC2017 全方法对比分析器"""
    
    def __init__(self, base_dir=None):
        """
        Args:
            base_dir: 项目根目录，默认为当前文件所在目录的父目录
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        
        # 结果目录配置
        self.results_dirs = {
            # GrVS-Mapping-BO 原版
            'GrVS-Mapping-BO': self.base_dir / "GrVS_mapping" / "results_cec2017_grvs_mapping_paper_mode",
            # GrVS 变体
            'GrVS-TopK-TailFilter-BO': self.base_dir / "GrVS_mapping" / "results_cec2017_grvs_variants",
            'GrVS-NoExplicit-TailFilter-BO': self.base_dir / "GrVS_mapping" / "results_cec2017_grvs_variants",
            # Baseline
            'Baseline': self.base_dir / "results" / "cec2017_paper_mode" / "GPrs",
        }
        
        # 输出目录
        self.output_dir = self.base_dir / "GrVS_mapping" / "analysis_results_all_methods"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 我们的方法 (GrVS 系列)
        self.our_algorithms = [
            'GrVS-Mapping-BO',
            'GrVS-TopK-TailFilter-BO',
            'GrVS-NoExplicit-TailFilter-BO',
        ]
        
        # Baseline 方法
        self.baseline_algorithms = [
            'Add-GP-UCB',
            'REMBO',
            'ALEBO',
            'Gr-VS',
            'BAxUS',
        ]
        
        # 所有方法
        self.all_algorithms = self.our_algorithms + self.baseline_algorithms
        
        # 颜色配置
        self.colors = {
            # 我们的方法 - 红色系
            'GrVS-Mapping-BO': '#e74c3c',              # 红色
            'GrVS-TopK-TailFilter-BO': '#c0392b',      # 深红
            'GrVS-NoExplicit-TailFilter-BO': '#ff6b6b', # 浅红
            # Baseline - 其他颜色
            'Add-GP-UCB': '#3498db',   # 蓝色
            'REMBO': '#2ecc71',        # 绿色
            'ALEBO': '#9b59b6',        # 紫色
            'Gr-VS': '#1abc9c',        # 青色
            'BAxUS': '#f39c12',        # 橙色
        }
        
        # 线型配置
        self.linestyles = {
            'GrVS-Mapping-BO': '-',
            'GrVS-TopK-TailFilter-BO': '-',
            'GrVS-NoExplicit-TailFilter-BO': '-',
            'Add-GP-UCB': '--',
            'REMBO': '--',
            'ALEBO': ':',
            'Gr-VS': '-.',
            'BAxUS': '-.',
        }
        
        # 标记配置
        self.markers = {
            'GrVS-Mapping-BO': 'D',
            'GrVS-TopK-TailFilter-BO': 's',
            'GrVS-NoExplicit-TailFilter-BO': '^',
            'Add-GP-UCB': 'o',
            'REMBO': 'v',
            'ALEBO': '<',
            'Gr-VS': '>',
            'BAxUS': 'p',
        }
        
        # 简短名称 (用于图例)
        self.short_names = {
            'GrVS-Mapping-BO': 'GrVS-Map',
            'GrVS-TopK-TailFilter-BO': 'TopK-TailFilter',
            'GrVS-NoExplicit-TailFilter-BO': 'NoExp-TailFilter',
            'Add-GP-UCB': 'Add-GP-UCB',
            'REMBO': 'REMBO',
            'ALEBO': 'ALEBO',
            'Gr-VS': 'Gr-VS',
            'BAxUS': 'BAxUS',
        }
        
        # CEC2017 目标函数
        self.target_functions = [f'F{i}' for i in [1, 5, 10, 15, 20, 25, 30]]
        self.seeds = [0, 1, 2, 3, 4]
        
        # 全局最优值
        self.global_optima = {f'F{i}': i * 100 for i in range(1, 31)}
        
        # 存储结果 {algorithm: {function: {seed: convergence_curve}}}
        self.results = defaultdict(lambda: defaultdict(dict))
    
    # =========================
    # 数据加载
    # =========================
    def load_all_results(self):
        """加载所有方法的结果"""
        print("\n" + "=" * 80)
        print("加载所有方法的实验结果")
        print("=" * 80)
        
        # 1. 加载 GrVS-Mapping-BO 原版
        self._load_grvs_mapping_results()
        
        # 2. 加载 GrVS 变体
        self._load_grvs_variants_results()
        
        # 3. 加载 Baseline
        self._load_baseline_results()
        
        # 4. 打印统计
        self._print_load_statistics()
        
        return True
    
    def _load_grvs_mapping_results(self):
        """加载 GrVS-Mapping-BO 原版结果"""
        results_dir = self.results_dirs['GrVS-Mapping-BO']
        algorithm = 'GrVS-Mapping-BO'
        
        print(f"\n📁 加载 {algorithm}: {results_dir}")
        
        if not results_dir.exists():
            print(f"  ⚠ 目录不存在")
            return
        
        count = 0
        for json_file in results_dir.glob("*.json"):
            if json_file.name == "summary.json":
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 检查算法名
                file_algorithm = data.get('algorithm', algorithm)
                if file_algorithm != algorithm:
                    continue
                
                function = data.get('function', '')
                seed = data.get('seed', -1)
                
                if function and seed >= 0:
                    convergence = self._extract_convergence(data)
                    if convergence is not None and len(convergence) > 0:
                        self.results[algorithm][function][seed] = np.array(convergence)
                        count += 1
            
            except Exception as e:
                print(f"  ⚠ 加载失败 {json_file.name}: {e}")
        
        print(f"  ✓ 加载 {count} 个结果")
    
    def _load_grvs_variants_results(self):
        """加载 GrVS 变体结果"""
        results_dir = self.results_dirs['GrVS-TopK-TailFilter-BO']
        
        print(f"\n📁 加载 GrVS 变体: {results_dir}")
        
        if not results_dir.exists():
            print(f"  ⚠ 目录不存在")
            return
        
        count = 0
        for json_file in results_dir.glob("*.json"):
            if json_file.name == "summary.json":
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                algorithm = data.get('algorithm', '')
                
                # 只加载我们的变体
                if algorithm not in ['GrVS-TopK-TailFilter-BO', 'GrVS-NoExplicit-TailFilter-BO']:
                    continue
                
                function = data.get('function', '')
                seed = data.get('seed', -1)
                
                if function and seed >= 0:
                    convergence = self._extract_convergence(data)
                    if convergence is not None and len(convergence) > 0:
                        self.results[algorithm][function][seed] = np.array(convergence)
                        count += 1
            
            except Exception as e:
                print(f"  ⚠ 加载失败 {json_file.name}: {e}")
        
        print(f"  ✓ 加载 {count} 个结果")
    
    def _load_baseline_results(self):
        """加载 Baseline 结果"""
        results_dir = self.results_dirs['Baseline']
        
        print(f"\n📁 加载 Baseline: {results_dir}")
        
        if not results_dir.exists():
            print(f"  ⚠ 目录不存在")
            return
        
        count = 0
        for json_file in results_dir.glob("*.json"):
            try:
                filename = json_file.name
                
                # 解析算法名
                algorithm = None
                if 'Add-GP-UCB' in filename:
                    algorithm = 'Add-GP-UCB'
                elif 'ALEBO' in filename:
                    algorithm = 'ALEBO'
                elif 'REMBO' in filename:
                    algorithm = 'REMBO'
                elif 'BAxUS' in filename:
                    algorithm = 'BAxUS'
                elif 'Gr-VS' in filename or 'GrVS' in filename:
                    if 'Mapping' not in filename and 'TailFilter' not in filename:
                        algorithm = 'Gr-VS'
                
                if algorithm is None:
                    continue
                
                # 解析函数名
                func_match = re.search(r'_F(\d+)_', filename)
                if not func_match:
                    continue
                
                f_idx = int(func_match.group(1))
                if f_idx not in [1, 5, 10, 15, 20, 25, 30]:
                    continue
                function = f'F{f_idx}'
                
                # 解析种子
                seed_match = re.search(r'_s(\d+)_', filename)
                if not seed_match:
                    continue
                seed = int(seed_match.group(1))
                
                # 加载数据
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                convergence = self._extract_convergence(data)
                
                if convergence is not None and len(convergence) > 0:
                    self.results[algorithm][function][seed] = np.array(convergence)
                    count += 1
            
            except Exception as e:
                print(f"  ⚠ 加载失败 {json_file.name}: {e}")
        
        print(f"  ✓ 加载 {count} 个结果")
    
    def _extract_convergence(self, data):
        """从数据中提取收敛曲线"""
        # 方式1: convergence_curve
        if 'convergence_curve' in data:
            return data['convergence_curve']
        
        # 方式2: convergence
        if 'convergence' in data:
            return data['convergence']
        
        # 方式3: observations
        if 'observations' in data and isinstance(data['observations'], list):
            objectives = []
            for obs in data['observations']:
                if isinstance(obs, dict) and 'objectives' in obs:
                    if obs['objectives'] and len(obs['objectives']) > 0:
                        objectives.append(obs['objectives'][0])
            
            if objectives:
                return np.minimum.accumulate(objectives).tolist()
        
        # 方式4: y
        if 'y' in data:
            y_values = data['y']
            if isinstance(y_values, list) and len(y_values) > 0:
                return np.minimum.accumulate(y_values).tolist()
        
        return None
    
    def _print_load_statistics(self):
        """打印加载统计"""
        print("\n" + "=" * 80)
        print("加载统计")
        print("=" * 80)
        
        print("\n我们的方法:")
        for algorithm in self.our_algorithms:
            if algorithm in self.results:
                total = sum(len(seeds) for seeds in self.results[algorithm].values())
                functions = sorted(self.results[algorithm].keys())
                print(f"  {algorithm:<35}: {total:2d} 次运行, 函数: {functions}")
            else:
                print(f"  {algorithm:<35}:  0 次运行")
        
        print("\nBaseline 方法:")
        for algorithm in self.baseline_algorithms:
            if algorithm in self.results:
                total = sum(len(seeds) for seeds in self.results[algorithm].values())
                functions = sorted(self.results[algorithm].keys())
                print(f"  {algorithm:<35}: {total:2d} 次运行, 函数: {functions}")
            else:
                print(f"  {algorithm:<35}:  0 次运行")
        
        # 统计共同函数
        all_functions = set()
        for alg_results in self.results.values():
            all_functions.update(alg_results.keys())
        
        print(f"\n共有 {len(all_functions)} 个函数有结果: {sorted(all_functions)}")
    
    # =========================
    # 性能表格
    # =========================
    def create_performance_table(self):
        """创建性能对比表格"""
        print("\n" + "=" * 80)
        print("生成性能对比表格")
        print("=" * 80)
        
        # 获取所有有结果的函数
        all_functions = set()
        for alg_results in self.results.values():
            all_functions.update(alg_results.keys())
        
        all_functions = sorted(all_functions, key=lambda x: int(x[1:]))
        
        if not all_functions:
            print("  ⚠ 没有结果数据")
            return None
        
        # 准备数据
        table_data = []
        
        for function in all_functions:
            global_opt = self.global_optima.get(function, 0)
            row = {'Function': function, 'Global_Opt': global_opt}
            
            for algorithm in self.all_algorithms:
                if function in self.results[algorithm]:
                    seeds_data = self.results[algorithm][function]
                    final_values = [curve[-1] for curve in seeds_data.values()]
                    
                    if final_values:
                        mean_val = np.mean(final_values)
                        std_val = np.std(final_values)
                        best_val = np.min(final_values)
                        regret = mean_val - global_opt
                        
                        row[f'{algorithm}_mean'] = mean_val
                        row[f'{algorithm}_std'] = std_val
                        row[f'{algorithm}_best'] = best_val
                        row[f'{algorithm}_regret'] = regret
                    else:
                        row[f'{algorithm}_mean'] = np.nan
                        row[f'{algorithm}_std'] = np.nan
                        row[f'{algorithm}_best'] = np.nan
                        row[f'{algorithm}_regret'] = np.nan
                else:
                    row[f'{algorithm}_mean'] = np.nan
                    row[f'{algorithm}_std'] = np.nan
                    row[f'{algorithm}_best'] = np.nan
                    row[f'{algorithm}_regret'] = np.nan
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # 保存 CSV
        csv_path = self.output_dir / "performance_comparison_all.csv"
        df.to_csv(csv_path, index=False)
        print(f"  ✓ CSV: {csv_path}")
        
        # 保存 Excel
        try:
            xlsx_path = self.output_dir / "performance_comparison_all.xlsx"
            df.to_excel(xlsx_path, index=False, engine='openpyxl')
            print(f"  ✓ Excel: {xlsx_path}")
        except ImportError:
            print("  ⚠ openpyxl 未安装，跳过 Excel")
        
        # 打印简化表格
        self._print_summary_table(all_functions)
        
        # 保存详细 TXT
        self._save_detailed_txt(all_functions)
        
        return df
    
    def _print_summary_table(self, functions):
        """打印简化的对比表格"""
        print("\n" + "-" * 150)
        print(f"{'Function':<10}", end="")
        for alg in self.all_algorithms:
            short = self.short_names.get(alg, alg)[:15]
            print(f"{short:<20}", end="")
        print()
        print("-" * 150)
        
        for function in functions:
            print(f"{function:<10}", end="")
            for alg in self.all_algorithms:
                if function in self.results[alg]:
                    seeds_data = self.results[alg][function]
                    final_values = [curve[-1] for curve in seeds_data.values()]
                    if final_values:
                        mean_val = np.mean(final_values)
                        std_val = np.std(final_values)
                        print(f"{mean_val:.2e}±{std_val:.2e} ", end="")
                    else:
                        print(f"{'N/A':<20}", end="")
                else:
                    print(f"{'N/A':<20}", end="")
            print()
        print("-" * 150)
    
    def _save_detailed_txt(self, functions):
        """保存详细的 TXT 表格"""
        txt_path = self.output_dir / "performance_comparison_all.txt"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 150 + "\n")
            f.write("CEC2017 全方法对比分析\n")
            f.write("=" * 150 + "\n\n")
            
            f.write("方法分组:\n")
            f.write("  - 我们的方法: GrVS-Mapping-BO, GrVS-TopK-TailFilter-BO, GrVS-NoExplicit-TailFilter-BO\n")
            f.write("  - Baseline: Add-GP-UCB, REMBO, ALEBO, Gr-VS, BAxUS\n\n")
            
            f.write("格式: 均值 ± 标准差 (Regret = 均值 - 全局最优)\n\n")
            
            # 表头
            f.write(f"{'Function':<10}{'Global_Opt':<12}")
            for alg in self.all_algorithms:
                short = self.short_names.get(alg, alg)[:15]
                f.write(f"{short:<22}")
            f.write("\n")
            f.write("-" * 150 + "\n")
            
            # 数据行
            for function in functions:
                global_opt = self.global_optima.get(function, 0)
                f.write(f"{function:<10}{global_opt:<12}")
                
                for alg in self.all_algorithms:
                    if function in self.results[alg]:
                        seeds_data = self.results[alg][function]
                        final_values = [curve[-1] for curve in seeds_data.values()]
                        if final_values:
                            mean_val = np.mean(final_values)
                            std_val = np.std(final_values)
                            f.write(f"{mean_val:.2e}±{std_val:.2e} ")
                        else:
                            f.write(f"{'N/A':<22}")
                    else:
                        f.write(f"{'N/A':<22}")
                f.write("\n")
            
            # 详细统计
            f.write("\n" + "=" * 150 + "\n")
            f.write("详细统计 (按函数)\n")
            f.write("=" * 150 + "\n")
            
            for function in functions:
                global_opt = self.global_optima.get(function, 0)
                f.write(f"\n{function} (Global Opt: {global_opt}):\n")
                f.write("-" * 100 + "\n")
                
                # 收集该函数的所有结果并排序
                func_results = []
                for alg in self.all_algorithms:
                    if function in self.results[alg]:
                        seeds_data = self.results[alg][function]
                        final_values = [curve[-1] for curve in seeds_data.values()]
                        if final_values:
                            mean_val = np.mean(final_values)
                            std_val = np.std(final_values)
                            regret = mean_val - global_opt
                            func_results.append((alg, mean_val, std_val, regret, len(final_values)))
                
                # 按 regret 排序
                func_results.sort(key=lambda x: x[3])
                
                for rank, (alg, mean_val, std_val, regret, n_runs) in enumerate(func_results, 1):
                    marker = "★" if alg in self.our_algorithms else " "
                    f.write(f"  {rank}. {marker} {alg:<35}: {mean_val:.6e} ± {std_val:.6e} "
                           f"(regret={regret:.2e}, n={n_runs})\n")
        
        print(f"  ✓ TXT: {txt_path}")
    
    # =========================
    # 收敛曲线图
    # =========================
    def plot_convergence_curves(self):
        """绘制收敛曲线"""
        print("\n" + "=" * 80)
        print("绘制收敛曲线")
        print("=" * 80)
        
        # 获取有结果的函数
        all_functions = set()
        for alg_results in self.results.values():
            all_functions.update(alg_results.keys())
        
        functions = sorted(all_functions, key=lambda x: int(x[1:]))
        
        if not functions:
            print("  ⚠ 没有数据可绘制")
            return
        
        # 每个函数单独绘图
        for function in functions:
            self._plot_single_function(function)
        
        # 合并图
        self._plot_combined(functions)
        
        # 分组对比图 (我们的方法 vs Baseline)
        self._plot_grouped_comparison(functions)
    
    def _plot_single_function(self, function):
        """绘制单个函数的收敛曲线"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        global_opt = self.global_optima.get(function, 0)
        plotted = []
        
        for algorithm in self.all_algorithms:
            if function not in self.results[algorithm]:
                continue
            
            seeds_data = self.results[algorithm][function]
            curves = list(seeds_data.values())
            
            if not curves:
                continue
            
            # 对齐长度
            min_len = min(len(c) for c in curves)
            curves_aligned = np.array([c[:min_len] for c in curves])
            
            mean_curve = np.mean(curves_aligned, axis=0)
            std_curve = np.std(curves_aligned, axis=0)
            
            x = np.arange(1, len(mean_curve) + 1)
            
            # 绘制均值曲线
            linewidth = 3.0 if algorithm in self.our_algorithms else 2.0
            ax.plot(x, mean_curve,
                   label=f"{self.short_names.get(algorithm, algorithm)} (n={len(curves)})",
                   color=self.colors.get(algorithm, 'gray'),
                   linestyle=self.linestyles.get(algorithm, '-'),
                   linewidth=linewidth,
                   marker=self.markers.get(algorithm, 'o'),
                   markevery=max(1, len(x) // 8),
                   markersize=7)
            
            # 标准差区域
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                           color=self.colors.get(algorithm, 'gray'),
                           alpha=0.15)
            
            plotted.append(algorithm)
        
        if not plotted:
            plt.close(fig)
            return
        
        # 全局最优线
        ax.axhline(y=global_opt, color='black', linestyle='--',
                  linewidth=2, label=f'Global Opt ({global_opt})', alpha=0.7)
        
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Best Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{function} - Convergence Comparison', fontsize=16, fontweight='bold')
        
        # 图例分两列
        ncol = 2 if len(plotted) > 5 else 1
        ax.legend(fontsize=9, loc='upper right', framealpha=0.9, ncol=ncol)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 自动选择 y 轴刻度
        y_range = ax.get_ylim()
        if y_range[1] / max(abs(y_range[0]), 1e-10) > 1000:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存
        png_path = self.output_dir / f"convergence_{function}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        
        pdf_path = self.output_dir / f"convergence_{function}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.close(fig)
        print(f"  ✓ {function}: {png_path.name}")
    
    def _plot_combined(self, functions):
        """绘制合并的收敛曲线图"""
        n_funcs = len(functions)
        n_cols = min(3, n_funcs)
        n_rows = (n_funcs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        
        if n_funcs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, function in enumerate(functions):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            global_opt = self.global_optima.get(function, 0)
            
            for algorithm in self.all_algorithms:
                if function not in self.results[algorithm]:
                    continue
                
                seeds_data = self.results[algorithm][function]
                curves = list(seeds_data.values())
                
                if not curves:
                    continue
                
                min_len = min(len(c) for c in curves)
                curves_aligned = np.array([c[:min_len] for c in curves])
                mean_curve = np.mean(curves_aligned, axis=0)
                
                x = np.arange(1, len(mean_curve) + 1)
                
                linewidth = 2.5 if algorithm in self.our_algorithms else 1.5
                ax.plot(x, mean_curve,
                       color=self.colors.get(algorithm, 'gray'),
                       linestyle=self.linestyles.get(algorithm, '-'),
                       linewidth=linewidth,
                       label=self.short_names.get(algorithm, algorithm))
            
            ax.axhline(y=global_opt, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax.set_title(f'{function}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Best Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # 自动 log 刻度
            y_range = ax.get_ylim()
            if y_range[1] / max(abs(y_range[0]), 1e-10) > 1000:
                ax.set_yscale('log')
            
            if idx == 0:
                ax.legend(fontsize=7, loc='upper right', ncol=2)
        
        # 隐藏多余子图
        for idx in range(n_funcs, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('All Methods Convergence Comparison on CEC2017',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / "convergence_combined.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 合并图: {save_path}")
    
    def _plot_grouped_comparison(self, functions):
        """绘制分组对比图 (我们的方法 vs Baseline 最佳)"""
        n_funcs = len(functions)
        n_cols = min(3, n_funcs)
        n_rows = (n_funcs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        
        if n_funcs == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, function in enumerate(functions):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            global_opt = self.global_optima.get(function, 0)
            
            # 绘制我们的方法
            for algorithm in self.our_algorithms:
                if function not in self.results[algorithm]:
                    continue
                
                seeds_data = self.results[algorithm][function]
                curves = list(seeds_data.values())
                
                if not curves:
                    continue
                
                min_len = min(len(c) for c in curves)
                curves_aligned = np.array([c[:min_len] for c in curves])
                mean_curve = np.mean(curves_aligned, axis=0)
                std_curve = np.std(curves_aligned, axis=0)
                
                x = np.arange(1, len(mean_curve) + 1)
                
                ax.plot(x, mean_curve,
                       color=self.colors.get(algorithm, 'gray'),
                       linestyle='-',
                       linewidth=2.5,
                       label=self.short_names.get(algorithm, algorithm))
                
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                               color=self.colors.get(algorithm, 'gray'), alpha=0.2)
            
            # 找到该函数上最佳的 Baseline
            best_baseline = None
            best_baseline_final = float('inf')
            
            for algorithm in self.baseline_algorithms:
                if function not in self.results[algorithm]:
                    continue
                
                seeds_data = self.results[algorithm][function]
                final_values = [curve[-1] for curve in seeds_data.values()]
                
                if final_values and np.mean(final_values) < best_baseline_final:
                    best_baseline_final = np.mean(final_values)
                    best_baseline = algorithm
            
            # 绘制最佳 Baseline
            if best_baseline and function in self.results[best_baseline]:
                seeds_data = self.results[best_baseline][function]
                curves = list(seeds_data.values())
                
                min_len = min(len(c) for c in curves)
                curves_aligned = np.array([c[:min_len] for c in curves])
                mean_curve = np.mean(curves_aligned, axis=0)
                std_curve = np.std(curves_aligned, axis=0)
                
                x = np.arange(1, len(mean_curve) + 1)
                
                ax.plot(x, mean_curve,
                       color=self.colors.get(best_baseline, 'gray'),
                       linestyle='--',
                       linewidth=2,
                       label=f'Best Baseline: {best_baseline}')
                
                ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve,
                               color=self.colors.get(best_baseline, 'gray'), alpha=0.15)
            
            ax.axhline(y=global_opt, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.set_title(f'{function}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Iteration', fontsize=10)
            ax.set_ylabel('Best Value', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            
            y_range = ax.get_ylim()
            if y_range[1] / max(abs(y_range[0]), 1e-10) > 1000:
                ax.set_yscale('log')
        
        for idx in range(n_funcs, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.suptitle('Our Methods vs Best Baseline on CEC2017',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = self.output_dir / "convergence_grouped.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 分组对比图: {save_path}")
    
    # =========================
    # 算法排名
    # =========================
    def create_ranking_table(self):
        """创建算法排名表"""
        print("\n" + "=" * 80)
        print("生成算法排名表")
        print("=" * 80)
        
        # 获取有结果的函数
        all_functions = set()
        for alg_results in self.results.values():
            all_functions.update(alg_results.keys())
        
        functions = sorted(all_functions, key=lambda x: int(x[1:]))
        
        if not functions:
            print("  ⚠ 没有数据")
            return
        
        # 计算排名
        rankings = defaultdict(list)
        wins = defaultdict(int)
        
        ranking_details = []
        
        for function in functions:
            global_opt = self.global_optima.get(function, 0)
            
            # 收集该函数的结果
            func_results = []
            for algorithm in self.all_algorithms:
                if function in self.results[algorithm]:
                    seeds_data = self.results[algorithm][function]
                    final_values = [curve[-1] for curve in seeds_data.values()]
                    
                    if final_values:
                        mean_val = np.mean(final_values)
                        func_results.append((algorithm, mean_val))
            
            if not func_results:
                continue
            
            # 按均值排序
            func_results.sort(key=lambda x: x[1])
            
            # 记录排名
            for rank, (algorithm, mean_val) in enumerate(func_results, 1):
                rankings[algorithm].append(rank)
                ranking_details.append({
                    'Function': function,
                    'Rank': rank,
                    'Algorithm': algorithm,
                    'Mean_Value': mean_val,
                    'Regret': mean_val - global_opt,
                })
            
            # 记录获胜
            if func_results:
                wins[func_results[0][0]] += 1
        
        # 保存详细排名
        df_ranking = pd.DataFrame(ranking_details)
        ranking_path = self.output_dir / "algorithm_ranking_details.csv"
        df_ranking.to_csv(ranking_path, index=False)
        print(f"  ✓ 详细排名: {ranking_path}")
        
        # 打印排名摘要
        print("\n" + "-" * 60)
        print("算法平均排名")
        print("-" * 60)
        
        avg_rank_summary = []
        for algorithm in self.all_algorithms:
            if algorithm in rankings:
                ranks = rankings[algorithm]
                avg_rank = np.mean(ranks)
                avg_rank_summary.append((algorithm, avg_rank, len(ranks)))
        
        avg_rank_summary.sort(key=lambda x: x[1])
        
        for algorithm, avg_rank, n_funcs in avg_rank_summary:
            marker = "★" if algorithm in self.our_algorithms else " "
            print(f"  {marker} {algorithm:<35}: 平均排名 {avg_rank:.2f} (共 {n_funcs} 个函数)")
        
        # 保存平均排名
        avg_rank_df = pd.DataFrame(avg_rank_summary, columns=['Algorithm', 'Avg_Rank', 'Num_Functions'])
        avg_rank_df['Is_Ours'] = avg_rank_df['Algorithm'].apply(lambda x: x in self.our_algorithms)
        avg_rank_path = self.output_dir / "algorithm_average_ranking.csv"
        avg_rank_df.to_csv(avg_rank_path, index=False)
        print(f"  ✓ 平均排名: {avg_rank_path}")
        
        # 打印获胜次数
        print("\n" + "-" * 60)
        print("获胜次数 (排名第一)")
        print("-" * 60)
        
        wins_sorted = sorted(wins.items(), key=lambda x: -x[1])
        for algorithm, count in wins_sorted:
            marker = "★" if algorithm in self.our_algorithms else " "
            print(f"  {marker} {algorithm:<35}: {count} 次")
        
        # 绘制排名柱状图
        self._plot_ranking_bar(avg_rank_summary, wins)
    
    def _plot_ranking_bar(self, avg_rank_summary, wins):
        """绘制排名柱状图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图: 平均排名
        ax1 = axes[0]
        algorithms = [x[0] for x in avg_rank_summary]
        avg_ranks = [x[1] for x in avg_rank_summary]
        colors = [self.colors.get(alg, 'gray') for alg in algorithms]
        
        bars1 = ax1.barh(range(len(algorithms)), avg_ranks, color=colors, edgecolor='black')
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_yticklabels([self.short_names.get(alg, alg) for alg in algorithms])
        ax1.set_xlabel('Average Rank (lower is better)', fontsize=12)
        ax1.set_title('Average Ranking Across Functions', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # 标注我们的方法
        for i, alg in enumerate(algorithms):
            if alg in self.our_algorithms:
                ax1.annotate('★', xy=(avg_ranks[i] + 0.1, i), fontsize=14, color='red', va='center')
        
        # 右图: 获胜次数
        ax2 = axes[1]
        wins_sorted = sorted(wins.items(), key=lambda x: -x[1])
        win_algorithms = [x[0] for x in wins_sorted]
        win_counts = [x[1] for x in wins_sorted]
        win_colors = [self.colors.get(alg, 'gray') for alg in win_algorithms]
        
        bars2 = ax2.barh(range(len(win_algorithms)), win_counts, color=win_colors, edgecolor='black')
        ax2.set_yticks(range(len(win_algorithms)))
        ax2.set_yticklabels([self.short_names.get(alg, alg) for alg in win_algorithms])
        ax2.set_xlabel('Number of Wins', fontsize=12)
        ax2.set_title('Number of Functions Ranked 1st', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 标注我们的方法
        for i, alg in enumerate(win_algorithms):
            if alg in self.our_algorithms:
                ax2.annotate('★', xy=(win_counts[i] + 0.1, i), fontsize=14, color='red', va='center')
        
        plt.tight_layout()
        
        save_path = self.output_dir / "ranking_comparison.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 排名图: {save_path}")
    
    # =========================
    # 统计检验
    # =========================
    def perform_statistical_tests(self):
        """执行统计检验"""
        print("\n" + "=" * 80)
        print("统计检验 (Wilcoxon Signed-Rank Test)")
        print("=" * 80)
        
        # 获取有结果的函数
        all_functions = set()
        for alg_results in self.results.values():
            all_functions.update(alg_results.keys())
        
        functions = sorted(all_functions, key=lambda x: int(x[1:]))
        
        if not functions:
            print("  ⚠ 没有数据")
            return
        
        # 对每对方法进行检验
        test_results = []
        
        # 我们的最佳方法 vs Baselines
        for our_alg in self.our_algorithms:
            for baseline_alg in self.baseline_algorithms:
                result = self._pairwise_test(our_alg, baseline_alg, functions)
                if result:
                    test_results.append(result)
        
        # 我们的方法之间的对比
        for i, alg1 in enumerate(self.our_algorithms):
            for alg2 in self.our_algorithms[i+1:]:
                result = self._pairwise_test(alg1, alg2, functions)
                if result:
                    test_results.append(result)
        
        if test_results:
            df_tests = pd.DataFrame(test_results)
            test_path = self.output_dir / "statistical_tests.csv"
            df_tests.to_csv(test_path, index=False)
            print(f"\n  ✓ 统计检验结果: {test_path}")
            
            # 打印摘要
            print("\n显著性检验摘要 (α=0.05):")
            print("-" * 80)
            for result in test_results:
                sig = "✓" if result['p_value'] < 0.05 else "✗"
                winner = result['winner'] if result['p_value'] < 0.05 else "N/A"
                print(f"  {result['Algorithm_1']:<30} vs {result['Algorithm_2']:<30}")
                print(f"    p-value={result['p_value']:.4f} {sig}  Winner: {winner}")
    
    def _pairwise_test(self, alg1, alg2, functions):
        """两两对比统计检验"""
        values1 = []
        values2 = []
        
        for function in functions:
            if function in self.results[alg1] and function in self.results[alg2]:
                finals1 = [curve[-1] for curve in self.results[alg1][function].values()]
                finals2 = [curve[-1] for curve in self.results[alg2][function].values()]
                
                if finals1 and finals2:
                    values1.append(np.mean(finals1))
                    values2.append(np.mean(finals2))
        
        if len(values1) < 3:
            return None
        
        try:
            stat, p_value = stats.wilcoxon(values1, values2)
            
            # 判断谁更好
            mean1 = np.mean(values1)
            mean2 = np.mean(values2)
            winner = alg1 if mean1 < mean2 else alg2
            
            return {
                'Algorithm_1': alg1,
                'Algorithm_2': alg2,
                'n_functions': len(values1),
                'statistic': stat,
                'p_value': p_value,
                'mean_1': mean1,
                'mean_2': mean2,
                'winner': winner,
                'significant': p_value < 0.05,
            }
        except Exception as e:
            return None
    
    # =========================
    # 主流程
    # =========================
    def run_analysis(self):
        """运行完整分析"""
        print("\n" + "=" * 80)
        print("CEC2017 全方法对比分析")
        print("=" * 80)
        print(f"输出目录: {self.output_dir}")
        
        # 1. 加载数据
        self.load_all_results()
        
        # 2. 性能表格
        self.create_performance_table()
        
        # 3. 收敛曲线
        self.plot_convergence_curves()
        
        # 4. 算法排名
        self.create_ranking_table()
        
        # 5. 统计检验
        self.perform_statistical_tests()
        
        print("\n" + "=" * 80)
        print("✅ 分析完成!")
        print("=" * 80)
        print(f"\n生成的文件:")
        print(f"  📊 性能表格: performance_comparison_all.csv/xlsx/txt")
        print(f"  📈 收敛曲线: convergence_F*.png/pdf, convergence_combined.png")
        print(f"  📈 分组对比: convergence_grouped.png")
        print(f"  🏆 算法排名: algorithm_ranking_details.csv, algorithm_average_ranking.csv")
        print(f"  📉 排名图: ranking_comparison.png")
        print(f"  📐 统计检验: statistical_tests.csv")
        print(f"\n所有文件保存在: {self.output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze all methods on CEC2017")
    parser.add_argument("--base_dir", type=str, default=None,
                       help="Project base directory")
    args = parser.parse_args()
    
    analyzer = AllMethodsAnalyzer(base_dir=args.base_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()