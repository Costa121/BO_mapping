'''结果分析和可视化脚本'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

def collect_results():
    """收集所有实验结果"""
    
    algorithms = ['Add-GP-UCB', 'PP-GP-UCB', 'RPP-GP-UCB', 'TAS-BO']
    functions = ['Branin', 'Hartmann6', 'Gramacy']
    seeds = [42, 123, 456]
    
    results = {}
    
    for algorithm in algorithms:
        results[algorithm] = {}
        for function in functions:
            results[algorithm][function] = {}
            
            convergence_curves = []
            final_values = []
            
            for seed in seeds:
                # 构建文件路径模式 - 根据实际文件名修改
                if algorithm == 'TAS-BO':
                    # TAS-BO的文件名模式：TAS-BO_Branin_seed42_*
                    file_pattern = f"{algorithm}_{function}_seed{seed}_*TAS-BO*.json"
                else:
                    # 其他算法的文件名模式：Add-GP-UCB_Branin_seed42_*SMAC*Add-GP-UCB*
                    file_pattern = f"{algorithm}_{function}_seed{seed}_*SMAC*{algorithm}*.json"
                
                # 查找结果文件
                result_files = list(Path("results/test_complete/SMACrs").glob(file_pattern))
                
                if result_files:
                    try:
                        # 读取JSON文件
                        with open(result_files[0], 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 从observations中提取收敛曲线
                        if 'observations' in data:
                            observations = data['observations']
                            objectives = [obs['objectives'][0] for obs in observations]
                            
                            # 计算收敛曲线（每次迭代的最优值）
                            curve = []
                            best_so_far = float('inf')
                            for obj in objectives:
                                best_so_far = min(best_so_far, obj)
                                curve.append(best_so_far)
                            
                            convergence_curves.append(curve)
                            final_values.append(curve[-1] if curve else float('inf'))
                            
                            print(f"✓ 加载: {algorithm} - {function} - seed{seed} "
                                  f"(最终值: {curve[-1]:.6f})")
                        else:
                            print(f"✗ 文件格式错误: {result_files[0]} - 缺少observations字段")
                            
                    except Exception as e:
                        print(f"✗ 加载失败: {result_files[0]} - {e}")
                        
                else:
                    print(f"✗ 文件未找到: {algorithm} - {function} - seed{seed}")
                    print(f"   查找模式: {file_pattern}")
            
            # 存储结果
            results[algorithm][function] = {
                'curves': convergence_curves,
                'final_values': final_values,
                'mean_curve': np.mean(convergence_curves, axis=0) if convergence_curves else None,
                'std_curve': np.std(convergence_curves, axis=0) if convergence_curves else None,
                'final_mean': np.mean(final_values) if final_values else float('inf'),
                'final_std': np.std(final_values) if final_values else 0
            }
    
    return results

def plot_convergence_curves(results, save_dir):
    """绘制收敛曲线图"""
    
    functions = ['Branin', 'Hartmann6', 'Gramacy']
    algorithms = ['Add-GP-UCB', 'PP-GP-UCB', 'RPP-GP-UCB', 'TAS-BO']
    
    # 颜色和标记样式
    colors = {
        'Add-GP-UCB': '#1f77b4',
        'PP-GP-UCB': '#ff7f0e', 
        'RPP-GP-UCB': '#2ca02c',
        'TAS-BO': '#d62728'
    }
    
    markers = {
        'Add-GP-UCB': 'o',
        'PP-GP-UCB': 's',
        'RPP-GP-UCB': '^', 
        'TAS-BO': 'D'
    }
    
    # 为每个函数创建图
    for function in functions:
        plt.figure(figsize=(10, 6))
        
        for algorithm in algorithms:
            if (algorithm in results and 
                function in results[algorithm] and 
                results[algorithm][function]['mean_curve'] is not None):
                
                data = results[algorithm][function]
                x = range(1, len(data['mean_curve']) + 1)
                y_mean = data['mean_curve']
                y_std = data['std_curve']
                
                # 绘制均值曲线
                plt.plot(x, y_mean, 
                        color=colors[algorithm], 
                        marker=markers[algorithm],
                        label=algorithm, 
                        linewidth=2,
                        markersize=4,
                        markevery=max(1, len(x)//15))
                
                # 绘制标准差区间
                plt.fill_between(x, 
                               y_mean - y_std, 
                               y_mean + y_std,
                               color=colors[algorithm], 
                               alpha=0.2)
            else:
                print(f"警告：{algorithm} 在 {function} 上没有数据")
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Best Objective Value', fontsize=12)
        plt.title(f'{function} (D=100)', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plt.savefig(save_dir / f'{function}_convergence.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir / f'{function}_convergence.pdf', bbox_inches='tight')
        plt.show()

def create_summary_table(results, save_dir):
    """创建结果汇总表格"""
    
    functions = ['Branin', 'Hartmann6', 'Gramacy']
    algorithms = ['Add-GP-UCB', 'PP-GP-UCB', 'RPP-GP-UCB', 'TAS-BO']
    
    print("\n" + "="*80)
    print("Table: The best results of the synthetic data with D=100")
    print("="*80)
    
    # 创建DataFrame用于保存
    summary_data = []
    
    for function in functions:
        print(f"\n{function}:")
        print("-" * 60)
        print(f"{'算法':<15} {'函数':<10} {'最终最优值（均值±标准差）':<25}")
        print("-" * 60)
        
        # 收集当前函数的所有算法结果
        func_results = []
        
        for algorithm in algorithms:
            if (algorithm in results and 
                function in results[algorithm] and 
                results[algorithm][function]['final_values']):
                
                data = results[algorithm][function]
                final_mean = data['final_mean']
                final_std = data['final_std']
                
                func_results.append({
                    'algorithm': algorithm,
                    'function': function,
                    'mean': final_mean,
                    'std': final_std,
                    'mean_std_str': f"{final_mean:.4f} ± {final_std:.4f}"
                })
                
                summary_data.append({
                    'function': function,
                    'algorithm': algorithm,
                    'final_mean': final_mean,
                    'final_std': final_std,
                    'result_string': f"{final_mean:.4f} ± {final_std:.4f}"
                })
            else:
                func_results.append({
                    'algorithm': algorithm,
                    'function': function,
                    'mean': float('inf'),
                    'std': 0,
                    'mean_std_str': "N/A"
                })
                
                summary_data.append({
                    'function': function,
                    'algorithm': algorithm,
                    'final_mean': float('inf'),
                    'final_std': 0,
                    'result_string': "N/A"
                })
        
        # 按最终均值排序（越小越好）
        func_results.sort(key=lambda x: x['mean'])
        
        # 打印结果
        for result in func_results:
            print(f"{result['algorithm']:<15} {result['function']:<10} {result['mean_std_str']:<25}")
    
    # 保存为CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(save_dir / 'summary_results.csv', index=False, encoding='utf-8-sig')
    
    # 创建排名表
    print(f"\n{'='*80}")
    print("算法总体排名 (基于所有函数的平均排名)")
    print(f"{'='*80}")
    
    # 计算每个算法在每个函数上的排名
    algorithm_ranks = {alg: [] for alg in algorithms}
    
    for function in functions:
        func_data = [(alg, results[alg][function]['final_mean']) 
                    for alg in algorithms 
                    if (alg in results and 
                        function in results[alg] and 
                        results[alg][function]['final_values'])]
        
        # 按最终值排序（升序，越小越好）
        func_data.sort(key=lambda x: x[1])
        
        # 分配排名
        for rank, (alg, _) in enumerate(func_data, 1):
            algorithm_ranks[alg].append(rank)
    
    # 计算平均排名
    avg_ranks = {}
    for alg in algorithms:
        if algorithm_ranks[alg]:
            avg_ranks[alg] = np.mean(algorithm_ranks[alg])
        else:
            avg_ranks[alg] = float('inf')
    
    # 按平均排名排序
    sorted_algorithms = sorted(avg_ranks.items(), key=lambda x: x[1])
    
    print(f"{'排名':<6} {'算法':<15} {'平均排名':<10}")
    print("-" * 35)
    for i, (alg, avg_rank) in enumerate(sorted_algorithms, 1):
        if avg_rank != float('inf'):
            print(f"{i:<6} {alg:<15} {avg_rank:.2f}")
        else:
            print(f"{i:<6} {alg:<15} N/A")
    
    return df

def main():
    """主函数"""
    
    print("开始分析算法对比结果...")
    
    # 创建输出目录
    save_dir = Path("comparison_analysis")
    save_dir.mkdir(exist_ok=True)
    
    # 1. 收集结果
    print("\n1. 收集实验结果...")
    results = collect_results()
    
    # 2. 绘制收敛曲线
    print("\n2. 绘制收敛曲线...")
    plot_convergence_curves(results, save_dir)
    
    # 3. 创建汇总表格
    print("\n3. 创建结果汇总表格...")
    summary_df = create_summary_table(results, save_dir)
    
    # 4. 保存完整结果
    with open(save_dir / 'complete_results.json', 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        results_serializable = {}
        for alg in results:
            results_serializable[alg] = {}
            for func in results[alg]:
                results_serializable[alg][func] = {
                    'final_values': results[alg][func]['final_values'],
                    'final_mean': results[alg][func]['final_mean'],
                    'final_std': results[alg][func]['final_std']
                }
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 分析完成！所有结果已保存到 {save_dir} 目录")
    print(f"  - 收敛曲线图: {save_dir}/*_convergence.png/pdf")
    print(f"  - 结果汇总表: {save_dir}/summary_results.csv")
    print(f"  - 完整结果: {save_dir}/complete_results.json")

if __name__ == "__main__":
    main()