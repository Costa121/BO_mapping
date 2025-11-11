'''批量实验运行脚本: Add-GP-UCB, PP-GP-UCB, RPP-GP-UCB, TAS-BO'''
import os
import subprocess
import itertools
import time
from pathlib import Path

def run_comparison_experiments():
    """运行算法对比实验"""
    
    # 实验参数
    algorithms = [ 'TAS-BO','Add-GP-UCB', 'PP-GP-UCB', 'RPP-GP-UCB']
    functions = ['Branin', 'Hartmann6', 'Gramacy']
    seeds = [42, 123, 456]  # 3个不同的随机种子
    iterations = 100
    dimension = 100
    
    # 创建结果目录
    results_dir = Path("comparison_results")
    results_dir.mkdir(exist_ok=True)
    
    total_experiments = len(algorithms) * len(functions) * len(seeds)
    current_experiment = 0
    
    print(f"开始运行 {total_experiments} 个实验...")
    print(f"算法: {algorithms}")
    print(f"函数: {functions}")
    print(f"种子: {seeds}")
    print(f"迭代次数: {iterations}, 维度: {dimension}")
    print("-" * 60)
    
    # 记录实验结果
    experiment_log = []
    failed_experiments = []
    
    for algorithm, function, seed in itertools.product(algorithms, functions, seeds):
        current_experiment += 1
        
        print(f"[{current_experiment}/{total_experiments}] "
              f"运行: {algorithm} on {function} (seed={seed})")
        
        # 构建命令
        if algorithm == 'TAS-BO':
            cmd = [
                'python', 'math_test.py',
                '--opt', 'TAS-BO',
                '--compress', 'none',
                '--func', function,
                '--dim', str(dimension),
                '--iter_num', str(iterations),
                '--seed', str(seed),
                '--task', f"{algorithm}_{function}_seed{seed}"
            ]
        else:
            cmd = [
                'python', 'math_test.py',
                '--opt', 'SMAC',
                '--compress', algorithm,
                '--func', function,
                '--dim', str(dimension),
                '--iter_num', str(iterations),
                '--seed', str(seed),
                '--task', f"{algorithm}_{function}_seed{seed}"
            ]
        
        # 执行实验
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=3600  # 1小时超时
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"  ✓ 成功 (耗时: {duration:.1f}s)")
                
                # 验证结果文件是否生成
                results_dir_check = Path("results/test_complete/SMACrs")
                if results_dir_check.exists():
                    if algorithm == 'TAS-BO':
                        file_pattern = f"{algorithm}_{function}_seed{seed}_*TAS-BO*.json"
                    else:
                        file_pattern = f"{algorithm}_{function}_seed{seed}_*SMAC*{algorithm}*.json"
                    
                    result_files = list(results_dir_check.glob(file_pattern))
                    if result_files:
                        print(f"    ✓ 结果文件: {result_files[0].name}")
                    else:
                        print(f"    ⚠ 未找到结果文件 (模式: {file_pattern})")
                
                experiment_log.append({
                    'algorithm': algorithm,
                    'function': function,
                    'seed': seed,
                    'status': 'success',
                    'duration': duration
                })
            else:
                print(f"  ✗ 失败 (返回码: {result.returncode})")
                print(f"  错误信息: {result.stderr[:200]}...")
                failed_experiments.append({
                    'algorithm': algorithm,
                    'function': function,
                    'seed': seed,
                    'error': result.stderr
                })
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ 超时 (>1小时)")
            failed_experiments.append({
                'algorithm': algorithm,
                'function': function,
                'seed': seed,
                'error': 'Timeout'
            })
        except Exception as e:
            print(f"  ✗ 异常: {e}")
            failed_experiments.append({
                'algorithm': algorithm,
                'function': function,
                'seed': seed,
                'error': str(e)
            })
    
    print("\n" + "=" * 60)
    print("实验完成总结:")
    print(f"成功: {len(experiment_log)}/{total_experiments}")
    print(f"失败: {len(failed_experiments)}/{total_experiments}")
    
    if failed_experiments:
        print("\n失败的实验:")
        for exp in failed_experiments:
            print(f"  - {exp['algorithm']} on {exp['function']} (seed={exp['seed']}): {exp['error'][:100]}")
    
    # 保存实验记录
    import json
    with open(results_dir / "experiment_log.json", 'w') as f:
        json.dump({
            'successful': experiment_log,
            'failed': failed_experiments,
            'summary': {
                'total': total_experiments,
                'successful': len(experiment_log),
                'failed': len(failed_experiments)
            }
        }, f, indent=2)
    
    print(f"\n实验记录已保存到: {results_dir / 'experiment_log.json'}")
    print("请运行 analysis_comparison_results.py 来生成图表和表格")

if __name__ == "__main__":
    run_comparison_experiments()