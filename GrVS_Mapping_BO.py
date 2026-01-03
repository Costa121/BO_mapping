"""
GrVS-Mapping-BO: 使用 Gr-VS 的 ECI 评分指导嵌入矩阵映射

核心思想:
  1. rho_j := ECI_{j,max} = max_{x_j in [a_j,b_j]} EI(z(x_j))
     (Gr-VS 用最大 ECI 衡量变量有效性)
  2. 构建嵌入矩阵 A (D x d):
     - Top-k 显式轴: 最重要的 k 个维度各占一个轴
     - 其余维度轮询分配到剩余列，并用 rho 做权重
  3. 在低维 z 空间用 BO 优化，x = Az 回到高维评估

严格方案 A4: 外层循环分段跑 K 轮 BO，每段内 A 固定，段间更新 A

参考论文: Gr-VS (Greedy Variable Selection) 中 ECI 的定义
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from openbox import Optimizer, space as sp


class GrVSMappingBO:
    """
    GrVS-Mapping-BO 实现（严格方案 A4）
    
    流程:
    1. 初始采样 (Latin Hypercube)
    2. 用 GP + EI 计算每个维度的 ECI_{j,max} 作为重要性分数 rho
    3. 构建嵌入矩阵 A:
       - 前 k_explicit 个最重要维度：各占一个显式轴
       - 其余维度：轮询分配到剩余轴，按 rho 加权并归一化
    4. **固定 A 跑 K 轮 BO** (z-space surrogate 自洽)
    5. **段间更新 A** (用全局 X_history/y_history 重新计算 rho)
    6. 重复直到预算用完
    """
    
    def __init__(
        self,
        f,
        D: int,
        d: int,
        k_explicit: int,
        update_freq: int = 20,
        seed: int = 0,
        # Gr-VS (ECI) 评分参数
        eci_grid_size: int = 31,       # 每个维度的 1D 网格点数
        gp_alpha: float = 1e-6,        # GP 噪声/数值稳定
        gp_n_restarts: int = 0,        # GP 超参数优化重启次数
        rho_momentum: float = 0.5,     # ρ 平滑系数: rho <- m*rho_old + (1-m)*rho_new
        use_random_signs: bool = True, # 是否使用随机符号稳定化
        eps: float = 1e-12,
    ):
        """
        Args:
            f: 目标函数 f: R^D -> R (最小化)
            D: 原始维度
            d: 嵌入空间维度
            k_explicit: 显式轴数量（前 k 个最重要维度各占一轴）
            update_freq: 每段 BO 跑多少轮后更新 A (即 K)
            seed: 随机种子
            eci_grid_size: ECI 计算时每个维度的网格点数
            gp_alpha: GP 的噪声参数
            gp_n_restarts: GP 超参数优化重启次数
            rho_momentum: ρ 平滑系数
            use_random_signs: 是否使用随机符号避免权重全正
            eps: 数值稳定小常数
        """
        self.f = f
        self.D = int(D)
        self.d = int(d)
        self.k_explicit = int(min(k_explicit, d, D))
        self.update_freq = int(update_freq)  # 每段 BO 的轮数 K
        self.seed = int(seed)
        self.rng = np.random.RandomState(seed)
        
        # ECI 计算参数
        self.eci_grid_size = int(eci_grid_size)
        self.gp_alpha = float(gp_alpha)
        self.gp_n_restarts = int(gp_n_restarts)
        self.rho_momentum = float(rho_momentum)
        self.use_random_signs = use_random_signs
        self.eps = float(eps)
        
        # 历史记录（全局，跨所有段）
        self.X_history = []  # 高维点
        self.y_history = []  # 目标值
        
        # 嵌入矩阵和重要性
        self.A = None           # (D, d)
        self.rho = None         # (D,) 每个维度的重要性分数
        self.xj_max = None      # (D,) 每个维度 ECI 最大处的坐标值
        self.top_explicit_dims = None  # 显式轴对应的维度
        self.tail_dims = None   # 轮询分配的维度
        
        # 段内临时记录（每段 BO 的 z 和 y）
        self._segment_z_history = []
        self._segment_y_history = []
        
        # 段计数器
        self._segment_count = 0
        
        # 固定随机符号，避免权重全正导致结构压缩
        if self.use_random_signs:
            self.signs = self.rng.choice([-1.0, 1.0], size=self.D)
        else:
            self.signs = np.ones(self.D)
    
    # =========================
    # 初始采样
    # =========================
    def generate_init_samples(self, n_samples=20):
        """初始采样（Latin Hypercube），范围 [-1, 1]^D"""
        sampler = LatinHypercube(d=self.D, seed=self.seed)
        samples_01 = sampler.random(n=n_samples)  # [0, 1]^D
        return samples_01 * 2.0 - 1.0  # 转换为 [-1, 1]^D
    
    # =========================
    # GP 训练 + EI 计算
    # =========================
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """
        训练 GP 模型（在高维 X 空间）
        
        Args:
            X: (n, D) 训练输入
            y: (n,) 训练目标
        
        Returns:
            训练好的 GP 模型
        """
        y = y.reshape(-1)
        
        # 核函数：ConstantKernel * RBF
        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0, 
            length_scale_bounds=(1e-2, 1e2)
        )
        
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.gp_alpha,
            normalize_y=True,
            n_restarts_optimizer=self.gp_n_restarts,
            random_state=self.seed + len(self.X_history),
        )
        gp.fit(X, y)
        return gp
    
    @staticmethod
    def _expected_improvement_minimize(
        mu: np.ndarray, 
        sigma: np.ndarray, 
        f_min: float
    ) -> np.ndarray:
        """
        计算 EI（最小化问题）
        
        EI(x) = (f_min - mu) * Phi(Z) + sigma * phi(Z)
        其中 Z = (f_min - mu) / sigma
        
        Args:
            mu: 预测均值
            sigma: 预测标准差
            f_min: 当前最小值
        
        Returns:
            EI 值
        """
        sigma = np.maximum(sigma, 1e-15)
        imp = f_min - mu
        Z = imp / sigma
        return imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    def _compute_eci_scores(self):
        """
        计算每个维度的 ECI_{j,max} 分数
        
        对每个维度 j:
          在 incumbent x* 的 coordinate line 上做网格搜索:
            rho_j = max EI(z(x_j))
            xj_max = argmax EI
        
        这对应 Gr-VS 论文中的 ECI_{j,max} 定义
        """
        X = np.asarray(self.X_history, dtype=float)
        y = np.asarray(self.y_history, dtype=float)
        
        # 找到当前最优点 (incumbent)
        best_idx = int(np.argmin(y))
        x_star = X[best_idx].copy()
        f_min = float(y[best_idx])
        
        # 在高维 X 空间训练 GP
        gp = self._fit_gp(X, y)
        
        # 1D 网格
        grid = np.linspace(-1.0, 1.0, self.eci_grid_size)
        
        rho_new = np.zeros(self.D, dtype=float)
        xj_max = np.zeros(self.D, dtype=float)
        
        # 对每个维度计算 ECI_{j,max}
        for j in range(self.D):
            # 构造 coordinate line 上的点：只改变第 j 维
            Xj = np.tile(x_star, (self.eci_grid_size, 1))
            Xj[:, j] = grid
            
            # GP 预测
            mu, sigma = gp.predict(Xj, return_std=True)
            
            # 计算 EI
            ei = self._expected_improvement_minimize(mu, sigma, f_min)
            
            # 取最大值作为该维度的重要性分数
            idx = int(np.argmax(ei))
            rho_new[j] = float(ei[idx])
            xj_max[j] = float(grid[idx])
        
        # 数值稳定：避免全 0
        if np.all(rho_new <= 0):
            rho_new = np.ones_like(rho_new)
        
        # 动量平滑，减少 A 的抖动
        if self.rho is None:
            rho_smoothed = rho_new
        else:
            m = self.rho_momentum
            rho_smoothed = m * self.rho + (1.0 - m) * rho_new
        
        # 归一化到 [0, 1]（保持相对大小）
        rho_smoothed = np.maximum(rho_smoothed, 0.0)
        s = rho_smoothed.sum()
        if s > self.eps:
            rho_smoothed = rho_smoothed / s
        
        self.rho = rho_smoothed
        self.xj_max = xj_max
        
        # 打印调试信息
        top_5 = np.argsort(-self.rho)[:5]
        print(f"\n  📊 [GrVS] ECI 评分完成:")
        print(f"     Top-5 维度: {top_5.tolist()}")
        print(f"     对应 rho: {[f'{self.rho[i]:.4f}' for i in top_5]}")
    
    # =========================
    # 嵌入矩阵构建
    # =========================
    def _build_embedding_matrix(self):
        """
        基于 rho 构造 embedding 矩阵 A (D x d)
        
        策略（复用 RF-Mapping-BO 的框架）:
        1. 前 k_explicit 维显式成轴: A[dim_i, i] = 1
        2. 其余维度轮询分配到剩余轴，按 rho 加权
        3. 稳定化: 混合列乘随机符号 + 每列 L2 归一化
        """
        D, d, k = self.D, self.d, self.k_explicit
        rho = self.rho
        
        # 1. 按重要性降序排序
        idx_sorted = np.argsort(-rho)
        self.top_explicit_dims = idx_sorted[:k].tolist()
        self.tail_dims = idx_sorted[k:].tolist()
        
        # 2. 初始化 A
        A = np.zeros((D, d), dtype=float)
        
        # 3. 显式轴: Top-k 各占一个轴
        for col, dim in enumerate(self.top_explicit_dims):
            A[dim, col] = 1.0
        
        # 4. 混合轴: 轮询分配 tail 维度
        n_mix_axes = d - k
        
        if n_mix_axes > 0 and len(self.tail_dims) > 0:
            # 分组：轮询分配
            groups = [[] for _ in range(n_mix_axes)]
            for t_idx, dim in enumerate(self.tail_dims):
                groups[t_idx % n_mix_axes].append(dim)
            
            # 每组内按 rho 归一化权重，并乘符号
            for g_i, dims in enumerate(groups):
                col = k + g_i
                if len(dims) == 0:
                    continue
                
                # 组内 rho 归一化
                group_rho_sum = float(np.sum(rho[dims]) + self.eps)
                for dim in dims:
                    w = float(rho[dim] / group_rho_sum)
                    A[dim, col] = self.signs[dim] * w
        
        # 5. 每列 L2 归一化（保证尺度统一）
        for col in range(d):
            norm2 = float(np.linalg.norm(A[:, col], ord=2))
            if norm2 > self.eps:
                A[:, col] /= norm2
        
        self.A = A
        
        # 打印调试信息
        print(f"\n  🔧 [GrVS-Mapping] 嵌入矩阵更新 (总样本数={len(self.X_history)}):")
        print(f"     显式轴 (k={k}): dims={self.top_explicit_dims[:min(5, k)]}...")
        if n_mix_axes > 0:
            print(f"     混合轴: {n_mix_axes} 个, 容纳 {len(self.tail_dims)} 个 tail 维度")
    
    def update_importance_and_embedding(self):
        """更新 ECI 分数和嵌入矩阵（段间调用）"""
        self._segment_count += 1
        print(f"\n{'='*60}")
        print(f"🔄 [GrVS-Mapping] 段 {self._segment_count}: 更新重要性和嵌入矩阵")
        print(f"   当前总样本数: {len(self.X_history)}")
        print(f"{'='*60}")
        self._compute_eci_scores()
        self._build_embedding_matrix()
    
    # =========================
    # 解码
    # =========================
    def decode(self, z: np.ndarray) -> np.ndarray:
        """
        从低维 z 映射到高维 x
        
        Args:
            z: (d,) 低维向量，范围 [-1, 1]
        
        Returns:
            x: (D,) 高维向量, x = Az (裁剪到 [-1, 1])
        """
        x = self.A @ z
        return np.clip(x, -1.0, 1.0)
    
    # =========================
    # 段内目标函数包装（A 固定）
    # =========================
    def _create_segment_objective(self):
        """
        创建当前段的目标函数包装器
        
        注意：这个函数在段内被调用时，A 是固定的
        """
        # 捕获当前的 A（固定）
        A_fixed = self.A.copy()
        
        def objective_wrapper(config):
            # 提取 z
            z = np.array([config[f'z{i}'] for i in range(self.d)], dtype=float)
            
            # 使用固定的 A 解码到 x
            x = A_fixed @ z
            x = np.clip(x, -1.0, 1.0)
            
            # 评估
            y = float(self.f(x))
            
            # 记录到全局历史（用于段间更新）
            self.X_history.append(x.copy())
            self.y_history.append(y)
            
            # 记录到段内历史（用于调试）
            self._segment_z_history.append(z.copy())
            self._segment_y_history.append(y)
            
            return {'objectives': [y]}
        
        return objective_wrapper
    
    # =========================
    # 运行单段 BO
    # =========================
    def _run_segment_bo(self, n_runs: int, segment_id: int):
        """
        运行单段 BO（A 固定）
        
        Args:
            n_runs: 本段跑多少轮
            segment_id: 段编号（用于 task_id）
        """
        if n_runs <= 0:
            return
        
        # 清空段内历史
        self._segment_z_history = []
        self._segment_y_history = []
        
        # 创建 z 空间
        space = sp.Space()
        for i in range(self.d):
            space.add_variable(sp.Real(f'z{i}', -1.0, 1.0, default_value=0.0))
        
        # 创建段内目标函数（A 固定）
        objective = self._create_segment_objective()
        
        # 段内初始采样数（取较小值）
        init_runs = min(3, max(1, n_runs // 4))
        
        print(f"\n  🚀 [段 {segment_id}] 开始 BO: {n_runs} 轮 (init={init_runs})")
        
        opt = Optimizer(
            objective,
            space,
            max_runs=n_runs,
            surrogate_type='gp',
            acq_type='ei',
            acq_optimizer_type='random_scipy',
            initial_runs=init_runs,
            init_strategy='latin_hypercube',
            random_state=self.seed + segment_id * 1000,
            task_id=f'grvs_seg{segment_id}_D{self.D}_d{self.d}'
        )
        
        opt.run()
        
        # 段结束统计
        if self._segment_y_history:
            seg_best = min(self._segment_y_history)
            global_best = min(self.y_history)
            print(f"  ✅ [段 {segment_id}] 完成: 段内最优={seg_best:.6f}, 全局最优={global_best:.6f}")
    
    # =========================
    # 运行优化（严格方案 A4）
    # =========================
    def run_optimization(self, init_n=20, max_iters=100):
        """
        运行优化（严格方案 A4）
        
        流程:
        1. 初始采样 init_n 个点
        2. 首次计算 rho 和 A
        3. 外层循环：
           - 固定 A，跑 K 轮 BO (K = update_freq)
           - 段间更新 A
           - 重复直到预算用完
        """
        print(f"\n{'='*80}")
        print(f"GrVS-Mapping-BO (严格方案 A4: 段间更新 A)")
        print(f"D={self.D}, d={self.d}, k_explicit={self.k_explicit}, seed={self.seed}")
        print(f"update_freq(K)={self.update_freq}, eci_grid_size={self.eci_grid_size}")
        print(f"{'='*80}\n")
        
        # 阶段1: 初始采样
        print(f"📊 阶段 1: 初始采样 ({init_n} 个点)")
        X_init = self.generate_init_samples(init_n)
        
        for x in X_init:
            y = float(self.f(x))
            self.X_history.append(x)
            self.y_history.append(y)
        
        print(f"   初始最优: {min(self.y_history):.6f}\n")
        
        # 阶段2: 首次更新 rho 和 A
        print(f"🔍 阶段 2: 首次计算 ECI 分数和嵌入矩阵")
        self.update_importance_and_embedding()
        
        # 阶段3: 分段 BO
        remaining_budget = max_iters - init_n
        K = self.update_freq  # 每段轮数
        segment_id = 1
        
        print(f"\n🚀 阶段 3: 分段 BO (总预算={remaining_budget}, 每段 K={K})")
        
        while remaining_budget > 0:
            # 本段跑多少轮
            n_runs_this_segment = min(K, remaining_budget)
            
            # 跑本段 BO（A 固定）
            self._run_segment_bo(n_runs_this_segment, segment_id)
            
            remaining_budget -= n_runs_this_segment
            
            # 段间更新 A（如果还有预算）
            if remaining_budget > 0:
                self.update_importance_and_embedding()
            
            segment_id += 1
        
        X = np.asarray(self.X_history)
        y = np.asarray(self.y_history)
        
        print(f"\n{'='*80}")
        print(f"✅ 优化完成!")
        print(f"   最终最优值: {float(y.min()):.6f}")
        print(f"   总评估次数: {len(y)}")
        print(f"   总段数: {self._segment_count}")
        print(f"{'='*80}")
        
        return X, y


def run_grvs_mapping_bo(
    f, D, d, k_explicit,
    init_n=20, max_iters=100,
    update_freq=20,
    eci_grid_size=31,
    rho_momentum=0.5,
    use_random_signs=True,
    seed=42,
    save_results=True,
    task_id_prefix="grvs_mapping"
):
    """
    GrVS-Mapping-BO 便捷接口（严格方案 A4）
    
    Args:
        f: 目标函数 (最小化)
        D: 原始维度
        d: 嵌入维度
        k_explicit: 显式轴数量
        init_n: 初始采样数
        max_iters: 总迭代次数
        update_freq: 每段 BO 轮数 K（段间更新 A）
        eci_grid_size: ECI 网格点数
        rho_momentum: ρ 平滑系数
        use_random_signs: 是否使用随机符号
        seed: 随机种子
        save_results: 是否保存结果
        task_id_prefix: 任务 ID 前缀
    
    Returns:
        X: (n, D) 历史输入
        y: (n,) 历史输出
    """
    optimizer = GrVSMappingBO(
        f=f,
        D=D,
        d=d,
        k_explicit=k_explicit,
        update_freq=update_freq,
        eci_grid_size=eci_grid_size,
        rho_momentum=rho_momentum,
        use_random_signs=use_random_signs,
        seed=seed,
    )
    X, y = optimizer.run_optimization(init_n=init_n, max_iters=max_iters)
    
    if save_results:
        results_dir = Path("results_grvs_mapping")
        results_dir.mkdir(exist_ok=True)
        
        result_data = {
            'X': X.tolist(),
            'y': y.tolist(),
            'best_y': float(y.min()),
            'best_x': X[int(np.argmin(y))].tolist(),
            'convergence': np.minimum.accumulate(y).tolist(),
            'top_explicit_dims': optimizer.top_explicit_dims,
            'tail_dims': optimizer.tail_dims,
            'final_rho': optimizer.rho.tolist() if optimizer.rho is not None else None,
            'xj_max': optimizer.xj_max.tolist() if optimizer.xj_max is not None else None,
            'embedding_matrix_shape': list(optimizer.A.shape) if optimizer.A is not None else None,
            'config': {
                'method': 'GrVS-Mapping-BO (Scheme A4)',
                'D': int(D),
                'd': int(d),
                'k_explicit': int(k_explicit),
                'update_freq': int(update_freq),
                'eci_grid_size': int(eci_grid_size),
                'rho_momentum': float(rho_momentum),
                'use_random_signs': bool(use_random_signs),
                'seed': int(seed),
                'init_n': int(init_n),
                'max_iters': int(max_iters),
                'actual_iters': int(len(y)),
                'n_segments': int(optimizer._segment_count),
            }
        }
        
        save_path = results_dir / f"{task_id_prefix}_D{D}_d{d}_k{k_explicit}_s{seed}.json"
        with open(save_path, 'w') as f_out:
            json.dump(result_data, f_out, indent=2)
        
        print(f"\n💾 结果已保存: {save_path}")
    
    return X, y


# =========================
# 测试代码
# =========================
if __name__ == "__main__":
    # 简单测试：Sphere 函数
    def sphere(x):
        return float(np.sum(x**2))
    
    D, d, k = 20, 5, 2
    X, y = run_grvs_mapping_bo(
        f=sphere,
        D=D,
        d=d,
        k_explicit=k,
        init_n=10,
        max_iters=50,
        update_freq=10,  # 每 10 轮更新一次 A
        seed=42,
        save_results=False
    )
    
    print(f"\n测试完成: best_y = {y.min():.6f}")