# ==============================================================================
# 文件: sampler.py (V3 - 修正排行榜胜率计算逻辑)
# 描述: 实现基于Chatbot Arena论文的自适应模型对战抽样器
# ==============================================================================

import pandas as pd
import numpy as np
import random
import os
from itertools import combinations

class ModelSampler:
    """
    实现一个自适应抽样策略，用于选择模型对进行比较。

    该策略基于 "Chatbot Arena" 论文的思想，优先选择具有以下特征的模型对：
    1.  **高不确定性**: 两个模型的历史胜率接近50%。
    2.  **探索性**: 两个模型之前的对战次数较少。
    
    这样做可以更高效地收敛模型排名，用更少的总票数获得更稳定的结果。
    抽样权重的计算遵循论文中公式(9)的精神。
    """

    def __init__(self, model_list: list, ratings_file_path: str):
        """
        初始化抽样器。

        Args:
            model_list (list): 所有可用模型名称的列表。
            ratings_file_path (str): 存储历史投票记录的CSV文件路径。
        """
        if len(model_list) < 2:
            raise ValueError("模型抽样器至少需要两个模型才能工作。")
        self.model_list = sorted(model_list)
        self.ratings_file = ratings_file_path
        self.all_pairs = list(combinations(self.model_list, 2))
        self.battle_counts = {pair: 0 for pair in self.all_pairs}
        self.win_counts = {model: {other_model: 0 for other_model in self.model_list if other_model != model} for model in self.model_list}
        self.leaderboard_printed = False

    def _load_and_process_ratings(self):
        """
        从CSV文件中加载历史投票数据，并计算每个模型对的对战次数和胜利次数。
        """
        # 每次调用时重置统计数据，以反映最新状态
        self.battle_counts = {pair: 0 for pair in self.all_pairs}
        self.win_counts = {model: {other_model: 0 for other_model in self.model_list if other_model != model} for model in self.model_list}

        if not os.path.exists(self.ratings_file):
            print("ℹ️ [抽样器] 未找到历史投票文件，将从零开始。")
            return

        try:
            ratings_df = pd.read_csv(self.ratings_file)
            if ratings_df.empty:
                return

            required_cols = ['winner', 'model_a', 'model_b']
            if not all(col in ratings_df.columns for col in required_cols):
                print(f"⚠️ [抽样器] 投票文件 '{self.ratings_file}' 缺少必需列，已跳过历史数据处理。")
                return

            # [关键修复] 迭代数据行并正确解析胜者
            for _, row in ratings_df.iterrows():
                model_a_name = row['model_a']
                model_b_name = row['model_b']
                winner_identifier = row['winner'] # winner列的值是 'model_a' 或 'model_b'

                if model_a_name not in self.model_list or model_b_name not in self.model_list:
                    continue

                # 为对战次数计数
                pair = tuple(sorted((model_a_name, model_b_name)))
                if pair in self.battle_counts:
                    self.battle_counts[pair] += 1
                
                # 根据 'winner' 列的标识符，确定胜者和败者的真实模型名称
                if winner_identifier == 'model_a':
                    winning_model, losing_model = model_a_name, model_b_name
                    self.win_counts[winning_model][losing_model] += 1
                elif winner_identifier == 'model_b':
                    winning_model, losing_model = model_b_name, model_a_name
                    self.win_counts[winning_model][losing_model] += 1
                # 如果是 'tie' 或其他值，则不计入任何一方的胜场，此逻辑保持不变
        
        except Exception as e:
            print(f"❌ [抽样器] 处理投票文件时发生错误: {e}")
            self.battle_counts = {pair: 0 for pair in self.all_pairs}
            self.win_counts = {model: {other_model: 0 for other_model in self.model_list if other_model != model} for model in self.model_list}


    def _calculate_sampling_weights(self) -> (list, list):
        """
        根据历史数据为每个模型对计算抽样权重。
        """
        pairs, weights = [], []
        max_weight_for_unseen = 0.0

        for pair in self.all_pairs:
            m1, m2 = pair
            n = self.battle_counts.get(pair, 0)
            
            if n > 0:
                # 注意：p_hat是m1相对m2的胜率，即使交换m1,m2，p_hat会变为1-p_hat，但p(1-p)不变
                wins_m1_vs_m2 = self.win_counts[m1].get(m2, 0)
                p_hat = wins_m1_vs_m2 / n
                variance_proxy = p_hat * (1 - p_hat) + 1e-6
                weight = np.sqrt(variance_proxy) * (1/np.sqrt(n) - 1/np.sqrt(n + 1))
                pairs.append(pair)
                weights.append(weight)
                if weight > max_weight_for_unseen:
                    max_weight_for_unseen = weight

        if max_weight_for_unseen == 0.0:
            max_weight_for_unseen = 1.0

        for pair in self.all_pairs:
            if self.battle_counts.get(pair, 0) == 0:
                pairs.append(pair)
                weights.append(max_weight_for_unseen * 1.1)
        
        return pairs, weights

    def _display_leaderboard(self):
        """
        计算并显示当前排行榜。
        """
        print("\n" + "="*70)
        print("📊 当前模型排行榜 (基于历史投票数据)".center(70))
        print("="*70)

        model_stats = []
        for model in self.model_list:
            total_wins = sum(self.win_counts[model].values())
            total_battles = sum(self.battle_counts[pair] for pair in self.all_pairs if model in pair)
            
            win_rate = (total_wins / total_battles) * 100 if total_battles > 0 else 0
            model_stats.append({
                'model': model,
                'win_rate': win_rate,
                'wins': total_wins,
                'battles': total_battles
            })

        sorted_stats = sorted(model_stats, key=lambda x: x['win_rate'], reverse=True)

        if not any(s['battles'] > 0 for s in sorted_stats):
            print("无有效的对战数据，无法生成排行榜。")
            print("="*70 + "\n")
            return

        print(f"{'排名':<5}{'模型名称':<45}{'胜率':<10}{'胜场/总对战':<15}")
        print(f"{'-'*4:<5}{'-'*44:<45}{'-'*9:<10}{'-'*14:<15}")

        for rank, stats in enumerate(sorted_stats, 1):
            battle_summary = f"{stats['wins']}/{stats['battles']}"
            print(f"{rank:<5}{stats['model']:<45}{stats['win_rate']:.2f}%{'':<4}{battle_summary:<15}")
        
        print("="*70 + "\n")

    def _get_selection_reason(self, pair: tuple) -> str:
        """
        根据模型对的情况，生成选择它的技术理由。
        """
        canonical_pair = tuple(sorted(pair))
        n = self.battle_counts.get(canonical_pair, 0)

        if n == 0:
            return "全新对决：这对模型组合是首次被抽中，优先进行探索。"

        # 确保 p_hat 计算基于范式对的顺序
        m1, m2 = canonical_pair
        wins_m1_vs_m2 = self.win_counts[m1].get(m2, 0)
        p_hat = wins_m1_vs_m2 / n

        reason = ""
        # 分析不确定性
        if 0.4 <= p_hat <= 0.6:
            reason += f"激烈对决 (胜率 ≈ {p_hat:.0%})：双方势均力敌，需更多数据来判断优劣。"
        elif 0.2 <= p_hat < 0.4 or 0.6 < p_hat <= 0.8:
            reason += f"竞争性对决 (胜率 ≈ {p_hat:.0%})：胜负有一定倾向，但仍存在不确定性。"
        else:
            reason += f"低不确定性对决 (胜率 ≈ {p_hat:.0%})：为确保公平，仍需偶尔抽样。"

        # 分析探索程度
        if n <= 30:
            reason += f" (探索不足：对战次数仅 {n} 次)"

        return reason

    def select_pair(self) -> tuple:
        """
        公开方法，用于选择下一场对战的模型对。
        """
        self._load_and_process_ratings()

        if not self.leaderboard_printed:
            self._display_leaderboard()
            self.leaderboard_printed = True
        
        print("🔄 [抽样器] 正在使用自适应策略选择模型对...")
        
        pairs, weights = self._calculate_sampling_weights()

        if not pairs:
            print("⚠️ [抽样器] 未能计算权重，已回退至随机抽样。")
            return tuple(random.sample(self.model_list, 2))
        
        try:
            selected_pair = random.choices(population=pairs, weights=weights, k=1)[0]
            reason = self._get_selection_reason(selected_pair)
            print(f"✅ [抽样器] 策略选定对战: {selected_pair}")
            print(f"    👉 理由: {reason}")
            
            return tuple(random.sample(list(selected_pair), 2))
        
        except Exception as e:
            print(f"❌ [抽样器] 加权抽样时发生错误: {e}。已回退至随机抽用。")
            return tuple(random.sample(self.model_list, 2))