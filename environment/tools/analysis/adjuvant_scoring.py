#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
确定性、版本化的佐剂筛选打分与统计工具库。

设计目标：把核心的"打分 / 选择性 / 归一化 / 多重比较校正 / 伪重复聚合"逻辑
固化为确定性的库函数，避免每次运行由 LLM 即兴重写算法导致结果不可复现，
并修复历史实现中的关键缺陷：
- selectivity 比值在分母接近 0 时爆炸（除零伪影）；
- 各打分分量量纲不一致直接加权，被单一无界分量主导；
- 把同一化合物的剂量×时间×重复当作独立样本（伪重复）做检验；
- 缺失多重比较校正。

所有函数均为纯函数、无副作用，可被生成代码 import，也可独立调用。
"""

from typing import Dict, List, Sequence, Optional, Tuple
import numpy as np
import pandas as pd

# 版本号：写入结果可追溯所用打分逻辑；逻辑变更时递增
SCORING_VERSION = "1.0.0"


def robust_zscore(values: Sequence[float], epsilon: float = 1.0) -> np.ndarray:
    """
    基于 median / MAD 的稳健 Z 分数。

    epsilon 作为 MAD 的下限，防止低方差特征的 Z 分数爆量；默认 1.0 比历史值
    （0.1）更保守，减少后续被 clip 截断饱和的情况。
    """
    arr = np.asarray(values, dtype=float)
    median = np.nanmedian(arr)
    mad = np.nanmedian(np.abs(arr - median))
    scale = max(mad * 1.4826, epsilon)
    return (arr - median) / scale


def minmax_normalize(values: Sequence[float],
                     p_low: float = 5.0,
                     p_high: float = 95.0) -> np.ndarray:
    """
    基于分位数的稳健 min-max 归一化到 [0, 1]，对离群值不敏感。

    使用 p_low / p_high 分位数（默认 5/95）作为范围，超出部分裁剪到 [0, 1]。
    """
    arr = np.asarray(values, dtype=float)
    lo = np.nanpercentile(arr, p_low)
    hi = np.nanpercentile(arr, p_high)
    if not np.isfinite(hi - lo) or (hi - lo) <= 0:
        # 退化情形：全部相等或无有效范围，返回中性 0.5
        return np.full_like(arr, 0.5, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def selectivity_margin(module_scores: Sequence[float],
                       normalize: bool = True) -> float:
    """
    计算选择性（最高模块相对次高模块的优势），修复历史除零缺陷。

    历史实现用 (max - second) / (max + second + 1e-3)，当两者正负相消时分母
    趋近 0，分数爆炸到数百，使噪声化合物排到榜首。

    本实现：
    - 先将模块分映射到非负区间（默认按本化合物自身 min-max 到 [0,1]）；
    - 选择性 = 最高值 - 次高值（margin），天然非负、有界，不做比值。

    Args:
        module_scores: 单个化合物在各佐剂模块上的得分
        normalize: 是否先把模块分归一到 [0,1]（推荐 True，保证跨化合物可比且非负）

    Returns:
        选择性 margin，范围 [0, 1]（normalize=True 时）；模块数 < 2 时返回 0.0
    """
    arr = np.asarray(module_scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    if normalize:
        lo, hi = np.min(arr), np.max(arr)
        if (hi - lo) <= 0:
            return 0.0
        arr = (arr - lo) / (hi - lo)
    ordered = np.sort(arr)[::-1]
    return float(ordered[0] - ordered[1])


def composite_score(components: Dict[str, Sequence[float]],
                    weights: Dict[str, float],
                    higher_is_better: Optional[Dict[str, bool]] = None) -> np.ndarray:
    """
    将多个打分分量先各自归一化到 [0,1]，再做加权求和，结果落在 [0,1]。

    这样可避免不同量纲 / 无界分量（如历史的 selectivity 比值）主导总分。

    Args:
        components: {分量名: 该分量在所有候选上的数组}
        weights: {分量名: 权重}，内部会归一化到和为 1
        higher_is_better: {分量名: 是否越大越好}，False 的分量会取 (1 - 归一值)
                          （例如毒性分量越低越好）

    Returns:
        每个候选的综合得分数组，范围 [0, 1]
    """
    if not components:
        raise ValueError("components 不能为空")
    higher_is_better = higher_is_better or {}

    length = len(next(iter(components.values())))
    total_weight = sum(weights.get(name, 0.0) for name in components)
    if total_weight <= 0:
        raise ValueError("权重之和必须为正")

    score = np.zeros(length, dtype=float)
    for name, raw in components.items():
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        normed = minmax_normalize(raw)
        if not higher_is_better.get(name, True):
            normed = 1.0 - normed
        score += (w / total_weight) * normed
    return score


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正，返回与输入等长的校正后 q 值（NaN 原样保留）。
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    qvals = np.full(n, np.nan, dtype=float)
    valid_mask = np.isfinite(p)
    valid_idx = np.where(valid_mask)[0]
    m = valid_idx.size
    if m == 0:
        return qvals

    valid_p = p[valid_idx]
    order = np.argsort(valid_p)
    ranked = valid_p[order]
    ranks = np.arange(1, m + 1)
    adjusted = ranked * m / ranks
    # 保证单调不减（从大到小取累计最小）
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    q_sorted = np.empty(m, dtype=float)
    q_sorted[order] = adjusted
    qvals[valid_idx] = q_sorted
    return qvals


def aggregate_pseudoreplicates(df: pd.DataFrame,
                               unit_cols: Sequence[str],
                               value_cols: Sequence[str],
                               agg: str = "mean") -> pd.DataFrame:
    """
    将伪重复折叠为独立实验单元，修复用 n=样本数 夸大显著性的统计错误。

    例如把同一化合物在所有剂量×时间×生物重复的样本，按 (compound, time) 等
    实验单元聚合（默认取均值），使后续统计以独立单元为单位。

    Args:
        df: 原始样本级数据
        unit_cols: 定义独立实验单元的列（如 ['COMPOUND_NAME', 'SACRIFICE_PERIOD']）
        value_cols: 需要聚合的数值列
        agg: 聚合方式（'mean' / 'median'）

    Returns:
        以实验单元为行的聚合后 DataFrame
    """
    missing = [c for c in list(unit_cols) + list(value_cols) if c not in df.columns]
    if missing:
        raise KeyError(f"数据缺少列: {missing}")
    grouped = df.groupby(list(unit_cols))[list(value_cols)]
    aggregated = getattr(grouped, agg)()
    return aggregated.reset_index()
