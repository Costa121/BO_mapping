"""
GrVS-TopK-TailFilter-BO: top-k 单轴尾部轮询映射 + TailFilter + 方向符号

核心思想:
  1. 使用 Gr-VS 的 ECI 评分计算变量重要性 rho
  2. **Top-k 单轴**: 前 k 个最重要维度各占一个显式轴 (A[dim_i, i] = 1)
  3. **尾部轮询**: 剩余维度轮询分配到后 (d-k) 个混合轴
  4. TailFilter: 对尾部维度做过滤，只保留重要性最高的 Top-M 个
  5. 方向符号: 使用 sign(x_{j,max} - x_j*) 确定方向，避免随机符号的不稳定性
  6. 每列 L2 归一化保证尺度统一

与原版 GrVS-Mapping-BO 的区别:
  - 原版: top-k 显式轴 + 全部尾部轮询 + 随机符号
  - 本版: top-k 显式轴 + TailFilter 尾部轮询 + 方向符号
"""

import json
from pathlib import Path
import numpy as np
from scipy.stats.qmc import LatinHypercube
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from openbox import Optimizer, space as sp


class GrVSTopKTailFilterBO:
    """
    GrVS-TopK-TailFilter-BO 实现
    
    特点:
    - Top-k 显式轴: 前 k 个最重要维度各占一个独立轴
    - TailFilter: 对尾部维度过滤，只保留重要性最高的 M 个
    - 方向符号: sign(x_{j,max} - x_j*) 替代随机符号
    - 尾部轮询: 过滤后的尾部维度轮询分配到剩余 (d-k) 个轴
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
        eci_grid_size: int = 31,
        gp_alpha: float = 1e-6,
        gp_n_restarts: int = 0,
        rho_momentum: float = 0.5,
        tail_filter: bool = True,       # 是否启用 TailFilter
        tail_filter_ratio: float = 1.5, # 保留 M = (d-k) * ratio 个尾部维度
        eps: float = 1e-12,
    ):
        """
        Args:
            f: 目标函数 f: R^D -> R (最小化)
            D: 原始维度
            d: 嵌入空间维度
            k_explicit: 显式轴数量（前 k 个最重要维度各占一轴）
            update_freq: 每段 BO 跑多少轮后更新 A
            seed: 随机种子
            eci_grid_size: ECI 计算时每个维度的网格点数
            gp_alpha: GP 的噪声参数
            gp_n_restarts: GP 超参数优化重启次数
            rho_momentum: ρ 平滑系数
            tail_filter: 是否启用 TailFilter
            tail_filter_ratio: TailFilter 保留的尾部维度数 M = min(D-k, int((d-k) * ratio))
            eps: 数值稳定小常数
        """
        self.f = f
        self.D = int(D)
        self.d = int(d)
        self.k_explicit = int(min(k_explicit, d, D))
        self.update_freq = int(update_freq)
        self.seed = int(seed)
        self.rng = np.random.RandomState(seed)
        
        # ECI 计算参数
        self.eci_grid_size = int(eci_grid_size)
        self.gp_alpha = float(gp_alpha)
        self.gp_n_restarts = int(gp_n_restarts)
        self.rho_momentum = float(rho_momentum)
        self.tail_filter = tail_filter
        self.tail_filter_ratio = float(tail_filter_ratio)
        self.eps = float(eps)
        
        # 历史记录
        self.X_history = []
        self.y_history = []
        
        # 嵌入矩阵和重要性
        self.A = None
        self.rho = None               # (D,) 每个维度的重要性分数
        self.xj_max = None            # (D,) 每个维度 ECI 最大处的坐标值
        self.x_star = None            # (D,) 当前最优点
        self.top_explicit_dims = None # 显式轴对应的维度
        self.tail_dims = None         # 轮询分配的尾部维度
        self.active_tail_dims = None  # TailFilter 后保留的尾部维度
        
        # 段内临时记录
        self._segment_z_history = []
        self._segment_y_history = []
        
        # 段计数器
        self._segment_count = 0
    
    # =========================
    # 初始采样
    # =========================
    def generate_init_samples(self, n_samples=20):
        """初始采样（Latin Hypercube），范围 [-1, 1]^D"""
        sampler = LatinHypercube(d=self.D, seed=self.seed)
        samples_01 = sampler.random(n=n_samples)
        return samples_01 * 2.0 - 1.0
    
    # =========================
    # GP 训练 + EI 计算
    # =========================
    def _fit_gp(self, X: np.ndarray, y: np.ndarray) -> GaussianProcessRegressor:
        """训练 GP 模型（在高维 X 空间）"""
        y = y.reshape(-1)
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
        """计算 EI（最小化问题）"""
        sigma = np.maximum(sigma, 1e-15)
        imp = f_min - mu
        Z = imp / sigma
        return imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    def _compute_eci_scores(self):
        """
        计算每个维度的 ECI_{j,max} 分数
        
        同时记录:
        - rho: 重要性分数
        - xj_max: 每个维度 ECI 最大处的坐标
        - x_star: 当前最优点
        """
        X = np.asarray(self.X_history, dtype=float)
        y = np.asarray(self.y_history, dtype=float)
        
        # 找到当前最优点 (incumbent)
        best_idx = int(np.argmin(y))
        x_star = X[best_idx].copy()
        f_min = float(y[best_idx])
        self.x_star = x_star
        
        # 在高维 X 空间训练 GP
        gp = self._fit_gp(X, y)
        
        # 1D 网格
        grid = np.linspace(-1.0, 1.0, self.eci_grid_size)
        
        rho_new = np.zeros(self.D, dtype=float)
        xj_max = np.zeros(self.D, dtype=float)
        
        # 对每个维度计算 ECI_{j,max}
        for j in range(self.D):
            Xj = np.tile(x_star, (self.eci_grid_size, 1))
            Xj[:, j] = grid
            
            mu, sigma = gp.predict(Xj, return_std=True)
            ei = self._expected_improvement_minimize(mu, sigma, f_min)
            
            idx = int(np.argmax(ei))
            rho_new[j] = float(ei[idx])
            xj_max[j] = float(grid[idx])
        
        # 数值稳定
        if np.all(rho_new <= 0):
            rho_new = np.ones_like(rho_new)
        
        # 动量平滑
        if self.rho is None:
            rho_smoothed = rho_new
        else:
            m = self.rho_momentum
            rho_smoothed = m * self.rho + (1.0 - m) * rho_new
        
        # 归一化
        rho_smoothed = np.maximum(rho_smoothed, 0.0)
        s = rho_smoothed.sum()
        if s > self.eps:
            rho_smoothed = rho_smoothed / s
        
        self.rho = rho_smoothed
        self.xj_max = xj_max
        
        # 打印调试信息
        top_5 = np.argsort(-self.rho)[:5]
        print(f"\n  📊 [GrVS-TopK-TailFilter] ECI 评分完成:")
        print(f"     Top-5 维度: {top_5.tolist()}")
        print(f"     对应 rho: {[f'{self.rho[i]:.4f}' for i in top_5]}")
    
    # =========================
    # 嵌入矩阵构建 (Top-k + TailFilter + 方向符号)
    # =========================
    def _build_embedding_matrix(self):
        """
        Top-k 单轴尾部轮询映射 + TailFilter + 方向符号
        
        步骤:
        1. Top-k 显式轴: 前 k 个最重要维度各占一个独立轴
        2. TailFilter: 对剩余尾部维度过滤，只保留 Top-M 个
        3. 方向符号: sign(x_{j,max} - x_j*) 确定方向
        4. 尾部轮询: 过滤后的尾部维度按 rho 加权轮询分配到剩余 (d-k) 个轴
        5. 每列 L2 归一化
        """
        D, d, k = self.D, self.d, self.k_explicit
        rho = self.rho.copy()
        x_star = self.x_star.copy()
        xj_max = self.xj_max.copy()
        
        # 1. 按重要性降序排序，确定 top-k 和 tail
        idx_sorted = np.argsort(-rho)
        self.top_explicit_dims = idx_sorted[:k].tolist()
        all_tail_dims = idx_sorted[k:].tolist()
        
        # 2. TailFilter: 只保留尾部中重要性最高的 M 个维度
        n_mix_axes = d - k  # 混合轴数量
        
        if self.tail_filter and n_mix_axes > 0:
            # M = min(|tail|, int((d-k) * ratio))
            M = min(len(all_tail_dims), max(n_mix_axes, int(n_mix_axes * self.tail_filter_ratio)))
            # 在 tail 中按 rho 排序（已经是降序排列，直接取前 M 个）
            self.active_tail_dims = all_tail_dims[:M]
        else:
            self.active_tail_dims = all_tail_dims
        
        self.tail_dims = all_tail_dims
        
        # 3. 计算方向符号: sign(x_{j,max} - x_j*)
        direction = xj_max - x_star
        sgn = np.sign(direction)
        sgn[sgn == 0] = 1.0  # 防止符号为零
        
        # 4. 初始化嵌入矩阵 A
        A = np.zeros((D, d), dtype=float)
        
        # 5. Top-k 显式轴: 每个 top 维度独占一个轴
        for col, dim in enumerate(self.top_explicit_dims):
            A[dim, col] = 1.0
        
        # 6. 尾部轮询: 过滤后的尾部维度轮询分配到混合轴
        if n_mix_axes > 0 and len(self.active_tail_dims) > 0:
            # 分组：轮询分配
            groups = [[] for _ in range(n_mix_axes)]
            for t_idx, dim in enumerate(self.active_tail_dims):
                groups[t_idx % n_mix_axes].append(dim)
            
            # 每组内按 rho 归一化权重，并乘方向符号
            for g_i, dims in enumerate(groups):
                col = k + g_i
                if len(dims) == 0:
                    continue
                
                # 组内 rho 归一化
                group_rho_sum = float(np.sum(rho[dims]) + self.eps)
                for dim in dims:
                    w = float(rho[dim] / group_rho_sum)
                    A[dim, col] = sgn[dim] * w
        
        # 7. 每列 L2 归一化
        for col in range(d):
            norm2 = float(np.linalg.norm(A[:, col], ord=2))
            if norm2 > self.eps:
                A[:, col] /= norm2
        
        self.A = A
        
        # 打印调试信息
        print(f"\n  🔧 [TopK-TailFilter] 嵌入矩阵更新 (总样本数={len(self.X_history)}):")
        print(f"     Top-k 显式轴 (k={k}): dims={self.top_explicit_dims[:min(5, k)]}...")
        if n_mix_axes > 0:
            print(f"     TailFilter: 保留 {len(self.active_tail_dims)}/{len(self.tail_dims)} 个尾部维度")
            print(f"     活跃尾部维度 (前5): {self.active_tail_dims[:min(5, len(self.active_tail_dims))]}...")
        print(f"     使用方向符号: sign(x_max - x*)")
    
    def update_importance_and_embedding(self):
        """更新 ECI 分数和嵌入矩阵"""
        self._segment_count += 1
        print(f"\n{'='*60}")
        print(f"🔄 [TopK-TailFilter] 段 {self._segment_count}: 更新重要性和嵌入矩阵")
        print(f"   当前总样本数: {len(self.X_history)}")
        print(f"{'='*60}")
        self._compute_eci_scores()
        self._build_embedding_matrix()
    
    # =========================
    # 解码
    # =========================
    def decode(self, z: np.ndarray) -> np.ndarray:
        """从低维 z 映射到高维 x: x = Az, 裁剪到 [-1, 1]"""
        x = self.A @ z
        return np.clip(x, -1.0, 1.0)
    
    # =========================
    # 段内目标函数包装
    # =========================
    def _create_segment_objective(self):
        """创建当前段的目标函数包装器（A 固定）"""
        A_fixed = self.A.copy()
        
        def objective_wrapper(config):
            z = np.array([config[f'z{i}'] for i in range(self.d)], dtype=float)
            x = A_fixed @ z
            x = np.clip(x, -1.0, 1.0)
            y = float(self.f(x))
            
            self.X_history.append(x.copy())
            self.y_history.append(y)
            self._segment_z_history.append(z.copy())
            self._segment_y_history.append(y)
            
            return {'objectives': [y]}
        
        return objective_wrapper
    
    # =========================
    # 运行单段 BO
    # =========================
    def _run_segment_bo(self, n_runs: int, segment_id: int):
        """运行单段 BO（A 固定）"""
        if n_runs <= 0:
            return
        
        self._segment_z_history = []
        self._segment_y_history = []
        
        space = sp.Space()
        for i in range(self.d):
            space.add_variable(sp.Real(f'z{i}', -1.0, 1.0, default_value=0.0))
        
        objective = self._create_segment_objective()
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
            task_id=f'topk_tailfilter_seg{segment_id}_D{self.D}_d{self.d}'
        )
        
        opt.run()
        
        if self._segment_y_history:
            seg_best = min(self._segment_y_history)
            global_best = min(self.y_history)
            print(f"  ✅ [段 {segment_id}] 完成: 段内最优={seg_best:.6f}, 全局最优={global_best:.6f}")
    
    # =========================
    # 运行优化
    # =========================
    def run_optimization(self, init_n=20, max_iters=100):
        """运行优化（方案 A4: 段间更新 A）"""
        print(f"\n{'='*80}")
        print(f"GrVS-TopK-TailFilter-BO (Top-k + TailFilter + 方向符号)")
        print(f"D={self.D}, d={self.d}, k_explicit={self.k_explicit}, seed={self.seed}")
        print(f"update_freq(K)={self.update_freq}, tail_filter={self.tail_filter}")
        print(f"tail_filter_ratio={self.tail_filter_ratio}")
        print(f"{'='*80}\n")
        
        # 阶段1: 初始采样
        print(f"📊 阶段 1: 初始采样 ({init_n} 个点)")
        X_init = self.generate_init_samples(init_n)
        
        for x in X_init:
            y = float(self.f(x))
            self.X_history.append(x)
            self.y_history.append(y)
        
        print(f"   初始最优: {min(self.y_history):.6f}\n")
        
        # 阶段2: 首次更新
        print(f"🔍 阶段 2: 首次计算 ECI 分数和嵌入矩阵")
        self.update_importance_and_embedding()
        
        # 阶段3: 分段 BO
        remaining_budget = max_iters - init_n
        K = self.update_freq
        segment_id = 1
        
        print(f"\n🚀 阶段 3: 分段 BO (总预算={remaining_budget}, 每段 K={K})")
        
        while remaining_budget > 0:
            n_runs_this_segment = min(K, remaining_budget)
            self._run_segment_bo(n_runs_this_segment, segment_id)
            remaining_budget -= n_runs_this_segment
            
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


def run_grvs_topk_tailfilter_bo(
    f, D, d, k_explicit,
    init_n=20, max_iters=100,
    update_freq=20,
    eci_grid_size=31,
    rho_momentum=0.5,
    tail_filter=True,
    tail_filter_ratio=1.5,
    seed=42,
    save_results=True,
    task_id_prefix="grvs_topk_tailfilter"
):
    """
    GrVS-TopK-TailFilter-BO 便捷接口
    
    Args:
        f: 目标函数 (最小化)
        D: 原始维度
        d: 嵌入维度
        k_explicit: 显式轴数量
        init_n: 初始采样数
        max_iters: 总迭代次数
        update_freq: 每段 BO 轮数 K
        eci_grid_size: ECI 网格点数
        rho_momentum: ρ 平滑系数
        tail_filter: 是否启用 TailFilter
        tail_filter_ratio: 保留尾部维度数 M = min(D-k, int((d-k) * ratio))
        seed: 随机种子
        save_results: 是否保存结果
        task_id_prefix: 任务 ID 前缀
    """
    optimizer = GrVSTopKTailFilterBO(
        f=f,
        D=D,
        d=d,
        k_explicit=k_explicit,
        update_freq=update_freq,
        eci_grid_size=eci_grid_size,
        rho_momentum=rho_momentum,
        tail_filter=tail_filter,
        tail_filter_ratio=tail_filter_ratio,
        seed=seed,
    )
    X, y = optimizer.run_optimization(init_n=init_n, max_iters=max_iters)
    
    if save_results:
        results_dir = Path("results_grvs_topk_tailfilter")
        results_dir.mkdir(exist_ok=True)
        
        result_data = {
            'X': X.tolist(),
            'y': y.tolist(),
            'best_y': float(y.min()),
            'best_x': X[int(np.argmin(y))].tolist(),
            'convergence': np.minimum.accumulate(y).tolist(),
            'top_explicit_dims': optimizer.top_explicit_dims,
            'tail_dims': optimizer.tail_dims,
            'active_tail_dims': optimizer.active_tail_dims,
            'final_rho': optimizer.rho.tolist() if optimizer.rho is not None else None,
            'xj_max': optimizer.xj_max.tolist() if optimizer.xj_max is not None else None,
            'config': {
                'method': 'GrVS-TopK-TailFilter-BO',
                'D': int(D),
                'd': int(d),
                'k_explicit': int(k_explicit),
                'tail_filter': bool(tail_filter),
                'tail_filter_ratio': float(tail_filter_ratio),
                'update_freq': int(update_freq),
                'eci_grid_size': int(eci_grid_size),
                'rho_momentum': float(rho_momentum),
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
    def sphere(x):
        return float(np.sum(x**2))
    
    D, d, k = 20, 5, 2
    X, y = run_grvs_topk_tailfilter_bo(
        f=sphere,
        D=D,
        d=d,
        k_explicit=k,
        init_n=10,
        max_iters=50,
        update_freq=10,
        tail_filter=True,
        tail_filter_ratio=1.5,
        seed=42,
        save_results=False
    )
    
    print(f"\n测试完成: best_y = {y.min():.6f}")
