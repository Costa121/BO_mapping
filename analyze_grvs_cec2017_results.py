# -*- coding: utf-8 -*-
"""
CEC2017 GrVS-Mapping-BO 与 Baseline 对比分析
- GrVS-Mapping-BO
- Add-GP-UCB, REMBO, ALEBO, Gr-VS, BAxUS
生成: 1) 性能对比表格  2) 收敛曲线图  3) 算法排名
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')


class GrVSCEC2017Analyzer:
    def __init__(self):
        # 结果目录
        self.grvs_mapping_results_dir = Path("/kmtian/saasbo-main/GrVS_mapping/results_cec2017_grvs_mapping_paper_mode")
        self.baseline_results_dir = Path("/kmtian/saasbo-main/results/cec2017_paper_mode/GPrs")
        self.output_dir = Path("/kmtian/saasbo-main/GrVS_mapping/analysis_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GrVS-Mapping-BO 算法
        self.grvs_algorithms = ['GrVS-Mapping-BO']
        
        # Baseline 算法
        self.baseline_algorithms = ['Add-GP-UCB', 'REMBO', 'ALEBO', 'Gr-VS', 'BAxUS']
        
        # 所有算法
        self.all_algorithms = self.grvs_algorithms + self.baseline_algorithms
        
        # 颜色配置
        self.colors = {
            'GrVS-Mapping-BO': '#e74c3c',     # 红色 (我们的方法)
            'Add-GP-UCB': '#3498db',           # 蓝色
            'REMBO': '#2ecc71',                # 绿色
            'ALEBO': '#9b59b6',                # 紫色
            'Gr-VS': '#1abc9c',                # 青色
            'BAxUS': '#f39c12',                # 橙色
        }
        
        # 线型配置
        self.linestyles = {
            'GrVS-Mapping-BO': '-',
            'Add-GP-UCB': '--',
            'REMBO': '-.',
            'ALEBO': ':',
            'Gr-VS': '-',
            'BAxUS': '--',
        }
        
        # 标记配置
        self.markers = {
            'GrVS-Mapping-BO': 'D',
            'Add-GP-UCB': 'o',
            'REMBO': 'v',
            'ALEBO': '<',
            'Gr-VS': '>',
            'BAxUS': 's',
        }
        
        # CEC2017 函数
        self.all_functions = [f'F{i}' for i in [1, 5, 10, 15, 20, 25, 30]]
        
        # 种子
        self.seeds = [0, 1, 2, 3, 4]
        
        # 全局最优值 (CEC2017: f(x*) = 100*i)
        self.global_optima = {f'F{i}': i * 100 for i in range(1, 31)}
        
        # 存储结果 {algorithm: {function: {seed: convergence_curve}}}
        self.results = defaultdict(lambda: defaultdict(dict))
    
    def load_results(self):
        """加载所有结果"""
        print("\n" + "="*80)
        print("加载 GrVS-Mapping-BO 和 Baseline 结果")
        print("="*80)
        
        # 1. 加载 GrVS-Mapping-BO 结果
        print(f"\n📁 加载 GrVS-Mapping-BO 结果: {self.grvs_mapping_results_dir}")
        if self.grvs_mapping_results_dir.exists():
            self._load_grvs_mapping_results()
        else:
            print(f"  ⚠ GrVS-Mapping-BO 结果目录不存在")
        
        # 2. 加载 Baseline 结果
        print(f"\n📁 加载 Baseline 结果: {self.baseline_results_dir}")
        if self.baseline_results_dir.exists():
            self._load_baseline_results()
        else:
            print(f"  ⚠ Baseline 结果目录不存在")
        
        # 3. 统计
        self._print_statistics()
        
        return True
    
    def _load_grvs_mapping_results(self):
        """加载 GrVS-Mapping-BO 结果"""
        json_files = list(self.grvs_mapping_results_dir.glob("GrVS-Mapping-BO_*.json"))
        print(f"  找到 {len(json_files)} 个文件")
        
        loaded = 0
        for json_file in json_files:
            if json_file.name == "summary.json":
                continue
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                algorithm = data.get('algorithm', 'GrVS-Mapping-BO')
                function = data.get('function')
                seed = data.get('seed')
                
                if function and seed is not None:
                    convergence = data.get('convergence_curve', [])
                    
                    if convergence and len(convergence) > 0:
                        self.results[algorithm][function][seed] = convergence
                        loaded += 1
                        print(f"    ✓ {algorithm:<20} {function:<5} seed={seed} "
                              f"({len(convergence)} iters, final={convergence[-1]:.2e})")
            
            except Exception as e:
                print(f"    ✗ {json_file.name}: {e}")
        
        print(f"  成功加载 {loaded} 个 GrVS-Mapping-BO 结果")
    
    def _load_baseline_results(self):
        """加载 Baseline 结果"""
        json_files = list(self.baseline_results_dir.glob("*.json"))
        print(f"  找到 {len(json_files)} 个文件")
        
        loaded = 0
        for json_file in json_files:
            try:
                filename = json_file.name
                
                # 解析算法名 - 按优先级匹配
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
                    # 排除 GrVS-Mapping-BO
                    if 'Mapping' not in filename:
                        algorithm = 'Gr-VS'
                
                if not algorithm:
                    continue
                
                # 解析函数名 - 修复: 匹配 _F10_ 这种格式
                function = None
                import re
                # 匹配 _F1_, _F5_, _F10_ 等格式
                func_match = re.search(r'_F(\d+)_', filename)
                if func_match:
                    f_idx = int(func_match.group(1))
                    if f_idx in [1, 5, 10, 15, 20, 25, 30]:
                        function = f'F{f_idx}'
                
                if not function:
                    continue
                
                # 解析种子 - 修复: 匹配 _s0_, _s1_ 等格式
                seed = None
                # 匹配 _s0_, _s1_, _s2_ 等格式 (取第一个出现的)
                seed_match = re.search(r'_s(\d+)_', filename)
                if seed_match:
                    seed = int(seed_match.group(1))
                
                if seed is None:
                    continue
                
                # 加载数据
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取收敛曲线
                convergence = self._extract_convergence(data)
                
                if convergence and len(convergence) > 0:
                    self.results[algorithm][function][seed] = convergence
                    loaded += 1
                    print(f"    ✓ {algorithm:<15} {function:<5} seed={seed} "
                          f"({len(convergence)} iters, final={convergence[-1]:.2e})")
            
            except Exception as e:
                print(f"    ✗ {json_file.name}: {e}")
        
        print(f"  成功加载 {loaded} 个 Baseline 结果")
    
    def _extract_convergence(self, data):
        """从数据中提取收敛曲线 (best-so-far)"""
        convergence = []
        
        # 方式1: 直接有 convergence_curve
        if 'convergence_curve' in data:
            return data['convergence_curve']
        
        # 方式2: 从 observations 提取
        if 'observations' in data and isinstance(data['observations'], list):
            objectives = []
            for obs in data['observations']:
                if isinstance(obs, dict) and 'objectives' in obs:
                    if obs['objectives'] and len(obs['objectives']) > 0:
                        objectives.append(obs['objectives'][0])
            
            if objectives:
                best = float('inf')
                for obj in objectives:
                    best = min(best, obj)
                    convergence.append(best)
        
        # 方式3: 从 y 字段提取
        if not convergence and 'y' in data:
            y_values = data['y']
            if isinstance(y_values, list):
                best = float('inf')
                for val in y_values:
                    best = min(best, val)
                    convergence.append(best)
        
        return convergence
    
    def _print_statistics(self):
        """打印加载统计"""
        print("\n" + "="*80)
        print("加载统计")
        print("="*80)
        
        for algorithm in self.all_algorithms:
            if algorithm in self.results:
                total = sum(len(seeds) for seeds in self.results[algorithm].values())
                functions = sorted(self.results[algorithm].keys())
                print(f"  {algorithm:<20}: {total:2d} 次运行, 函数: {functions}")
            else:
                print(f"  {algorithm:<20}:  0 次运行")
        
        # 找出共同函数
        grvs_functions = set()
        for alg in self.grvs_algorithms:
            if alg in self.results:
                grvs_functions.update(self.results[alg].keys())
        
        baseline_functions = set()
        for alg in self.baseline_algorithms:
            if alg in self.results:
                baseline_functions.update(self.results[alg].keys())
        
        common_functions = grvs_functions & baseline_functions
        
        print(f"\n函数覆盖情况:")
        print(f"  GrVS-Mapping-BO 函数: {sorted(grvs_functions)}")
        print(f"  Baseline 函数: {sorted(baseline_functions)}")
        print(f"  共同函数: {sorted(common_functions)}")
    
    def create_performance_table(self):
        """创建性能对比表格"""
        print("\n" + "="*80)
        print("生成性能对比表格")
        print("="*80)
        
        # 获取所有有结果的函数
        all_functions_with_results = set()
        for alg_results in self.results.values():
            all_functions_with_results.update(alg_results.keys())
        
        all_functions_with_results = sorted(all_functions_with_results, 
                                           key=lambda x: int(x[1:]))
        
        if not all_functions_with_results:
            print("  ⚠ 没有结果数据")
            return
        
        # 准备表格数据
        table_data = []
        
        for function in all_functions_with_results:
            row = {'Function': function}
            global_opt = self.global_optima.get(function, 0)
            
            for algorithm in self.all_algorithms:
                if function in self.results[algorithm]:
                    final_values = [curve[-1] for curve in self.results[algorithm][function].values()
                                  if curve and len(curve) > 0]
                    
                    if final_values:
                        mean_val = np.mean(final_values)
                        std_val = np.std(final_values)
                        error = mean_val - global_opt
                        
                        row[algorithm] = f"{error:.2e} ± {std_val:.2e}"
                    else:
                        row[algorithm] = "-"
                else:
                    row[algorithm] = "-"
            
            table_data.append(row)
        
        # 创建 DataFrame
        df = pd.DataFrame(table_data)
        columns = ['Function'] + self.all_algorithms
        df = df[[col for col in columns if col in df.columns]]
        
        # 保存为 CSV
        csv_path = self.output_dir / "grvs_performance_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ CSV 表格: {csv_path}")
        
        # 保存为 Excel
        try:
            xlsx_path = self.output_dir / "grvs_performance_table.xlsx"
            df.to_excel(xlsx_path, index=False, engine='openpyxl')
            print(f"✓ Excel 表格: {xlsx_path}")
        except ImportError:
            print("⚠ openpyxl 未安装，跳过 Excel 文件")
        
        # 打印表格
        print("\n性能对比表格 (误差 = 最终值 - 全局最优):")
        print("-" * 140)
        print(df.to_string(index=False))
        print("-" * 140)
        
        # 保存详细 TXT
        self._save_detailed_table(all_functions_with_results)
        
        return df
    
    def _save_detailed_table(self, functions):
        """保存详细表格"""
        txt_path = self.output_dir / "grvs_performance_table.txt"
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("="*140 + "\n")
            f.write("CEC2017 GrVS-Mapping-BO 与 Baseline 性能对比\n")
            f.write("="*140 + "\n\n")
            f.write("误差 = 最终值 - 全局最优值\n")
            f.write("格式: 均值误差 ± 标准差\n\n")
            
            # 表头
            f.write(f"{'函数':<10}")
            for alg in self.all_algorithms:
                f.write(f"{alg:<25}")
            f.write("\n")
            f.write("-" * 140 + "\n")
            
            # 数据行
            for function in functions:
                f.write(f"{function:<10}")
                global_opt = self.global_optima.get(function, 0)
                
                for algorithm in self.all_algorithms:
                    if function in self.results[algorithm]:
                        final_values = [curve[-1] for curve in self.results[algorithm][function].values()
                                      if curve and len(curve) > 0]
                        
                        if final_values:
                            mean_val = np.mean(final_values)
                            std_val = np.std(final_values)
                            error = mean_val - global_opt
                            f.write(f"{error:.2e}±{std_val:.2e}  ")
                        else:
                            f.write(f"{'N/A':<25}")
                    else:
                        f.write(f"{'N/A':<25}")
                f.write("\n")
            
            # 详细统计
            f.write("\n" + "="*140 + "\n")
            f.write("详细统计\n")
            f.write("="*140 + "\n\n")
            
            for function in functions:
                f.write(f"\n{function} (全局最优: {self.global_optima.get(function, 0)}):\n")
                f.write("-" * 100 + "\n")
                
                for algorithm in self.all_algorithms:
                    if function in self.results[algorithm]:
                        final_values = [curve[-1] for curve in self.results[algorithm][function].values()
                                      if curve and len(curve) > 0]
                        
                        if final_values:
                            mean_val = np.mean(final_values)
                            std_val = np.std(final_values)
                            min_val = np.min(final_values)
                            n_runs = len(final_values)
                            
                            f.write(f"  {algorithm:<20}: {mean_val:.6e} ± {std_val:.6e} "
                                  f"(best={min_val:.6e}, n={n_runs})\n")
                        else:
                            f.write(f"  {algorithm:<20}: No data\n")
                    else:
                        f.write(f"  {algorithm:<20}: No data\n")
        
        print(f"✓ TXT 表格: {txt_path}")
    
    def plot_convergence_curves(self):
        """绘制收敛曲线 - 每个函数一张图"""
        print("\n" + "="*80)
        print("生成收敛曲线图")
        print("="*80)
        
        # 获取所有有结果的函数
        all_functions_with_results = set()
        for alg_results in self.results.values():
            all_functions_with_results.update(alg_results.keys())
        
        all_functions_with_results = sorted(all_functions_with_results,
                                           key=lambda x: int(x[1:]))
        
        for function in all_functions_with_results:
            self._plot_single_function(function)
    
    def _plot_single_function(self, function):
        """绘制单个函数的收敛曲线"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        plotted_algorithms = []
        
        for algorithm in self.all_algorithms:
            if function not in self.results[algorithm]:
                continue
            
            curves = list(self.results[algorithm][function].values())
            
            if not curves:
                continue
            
            # 对齐曲线长度
            min_len = min(len(c) for c in curves)
            curves_aligned = np.array([c[:min_len] for c in curves])
            
            # 计算均值和标准差
            mean_curve = curves_aligned.mean(axis=0)
            std_curve = curves_aligned.std(axis=0)
            
            iterations = np.arange(1, len(mean_curve) + 1)
            
            # 绘制均值曲线
            ax.plot(iterations, mean_curve,
                   label=f"{algorithm} (n={len(curves)})",
                   color=self.colors.get(algorithm, 'gray'),
                   linestyle=self.linestyles.get(algorithm, '-'),
                   linewidth=2.5,
                   marker=self.markers.get(algorithm, 'o'),
                   markevery=max(1, len(iterations)//10),
                   markersize=6)
            
            # 绘制标准差区域
            ax.fill_between(iterations,
                           mean_curve - std_curve,
                           mean_curve + std_curve,
                           color=self.colors.get(algorithm, 'gray'),
                           alpha=0.15)
            
            plotted_algorithms.append(algorithm)
            print(f"  ✓ {function}: {algorithm} ({len(curves)} runs)")
        
        if not plotted_algorithms:
            print(f"  ⚠ {function}: 无数据")
            plt.close(fig)
            return
        
        # 添加全局最优线
        global_opt = self.global_optima.get(function, 0)
        ax.axhline(y=global_opt, color='red', linestyle='--',
                  linewidth=2, label=f'Global Optimum ({global_opt})', alpha=0.7)
        
        # 设置标签和标题
        ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
        ax.set_ylabel('Best Objective Value', fontsize=14, fontweight='bold')
        ax.set_title(f'{function} - Convergence Curves (GrVS-Mapping-BO vs Baselines)',
                    fontsize=16, fontweight='bold')
        
        # 图例
        ncol = 2 if len(plotted_algorithms) > 4 else 1
        ax.legend(fontsize=10, loc='best', framealpha=0.9, ncol=ncol)
        
        # 网格
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置y轴为对数刻度（如果数值范围很大）
        y_range = ax.get_ylim()
        if y_range[1] / max(abs(y_range[0]), 1e-10) > 1000:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存 PNG
        png_path = self.output_dir / f"convergence_{function}.png"
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"    ✓ 保存PNG: {png_path}")
        
        # 保存 PDF
        pdf_path = self.output_dir / f"convergence_{function}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"    ✓ 保存PDF: {pdf_path}")
        
        plt.close(fig)
    
    def create_ranking_table(self):
        """创建算法排名表"""
        print("\n" + "="*80)
        print("生成算法排名表")
        print("="*80)
        
        # 获取所有有结果的函数
        all_functions_with_results = set()
        for alg_results in self.results.values():
            all_functions_with_results.update(alg_results.keys())
        
        all_functions_with_results = sorted(all_functions_with_results,
                                           key=lambda x: int(x[1:]))
        
        ranking_data = []
        avg_ranks = defaultdict(list)
        
        for function in all_functions_with_results:
            global_opt = self.global_optima.get(function, 0)
            
            # 收集该函数上所有算法的性能
            function_results = []
            
            for algorithm in self.all_algorithms:
                if function in self.results[algorithm]:
                    final_values = [curve[-1] for curve in self.results[algorithm][function].values()
                                  if curve and len(curve) > 0]
                    
                    if final_values:
                        mean_val = np.mean(final_values)
                        error = abs(mean_val - global_opt)
                        function_results.append((algorithm, mean_val, error))
            
            # 按误差排序
            function_results.sort(key=lambda x: x[2])
            
            # 记录排名
            for rank, (algorithm, mean_val, error) in enumerate(function_results, 1):
                ranking_data.append({
                    'Function': function,
                    'Rank': rank,
                    'Algorithm': algorithm,
                    'Mean_Value': mean_val,
                    'Error': error
                })
                avg_ranks[algorithm].append(rank)
        
        if ranking_data:
            df = pd.DataFrame(ranking_data)
            ranking_file = self.output_dir / "grvs_algorithm_ranking.csv"
            df.to_csv(ranking_file, index=False, encoding='utf-8-sig')
            print(f"\n✓ 排名表格: {ranking_file}")
            
            # 打印排名摘要
            print("\n排名摘要 (每个函数的前3名):")
            for function in all_functions_with_results:
                func_df = df[df['Function'] == function].head(3)
                print(f"\n{function}:")
                for _, row in func_df.iterrows():
                    print(f"  {int(row['Rank'])}. {row['Algorithm']:<20}: {row['Error']:.2e}")
            
            # 平均排名
            print("\n" + "="*60)
            print("算法平均排名:")
            print("="*60)
            avg_rank_summary = []
            for algorithm in self.all_algorithms:
                if algorithm in avg_ranks and avg_ranks[algorithm]:
                    avg = np.mean(avg_ranks[algorithm])
                    avg_rank_summary.append((algorithm, avg, len(avg_ranks[algorithm])))
            
            avg_rank_summary.sort(key=lambda x: x[1])
            for algorithm, avg, count in avg_rank_summary:
                print(f"  {algorithm:<20}: {avg:.2f} (参与 {count} 个函数)")
            
            # 保存平均排名
            avg_rank_df = pd.DataFrame(avg_rank_summary, 
                                       columns=['Algorithm', 'Avg_Rank', 'Num_Functions'])
            avg_rank_file = self.output_dir / "grvs_average_ranking.csv"
            avg_rank_df.to_csv(avg_rank_file, index=False)
            print(f"\n✓ 平均排名: {avg_rank_file}")
    
    def run_analysis(self):
        """运行完整分析流程"""
        print("\n" + "="*80)
        print("CEC2017 GrVS-Mapping-BO 与 Baseline 对比分析")
        print("="*80)
        
        # 1. 加载结果
        if not self.load_results():
            print("\n✗ 加载结果失败")
            return False
        
        # 2. 生成性能表格
        self.create_performance_table()
        
        # 3. 绘制收敛曲线
        self.plot_convergence_curves()
        
        # 4. 生成排名表
        self.create_ranking_table()
        
        print("\n" + "="*80)
        print("分析完成！")
        print("="*80)
        print(f"\n生成的文件:")
        print(f"  📊 性能表格: {self.output_dir}/grvs_performance_table.csv/xlsx/txt")
        print(f"  📈 收敛曲线: {self.output_dir}/convergence_F*.png/pdf")
        print(f"  🏆 算法排名: {self.output_dir}/grvs_algorithm_ranking.csv")
        print(f"  🏆 平均排名: {self.output_dir}/grvs_average_ranking.csv")
        
        return True


def main():
    analyzer = GrVSCEC2017Analyzer()
    analyzer.run_analysis()


if __name__ == "__main__":
    main()