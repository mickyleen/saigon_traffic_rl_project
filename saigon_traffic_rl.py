"""
Mini Project Môn Học Tăng Cường (Reinforcement Learning)
Đề tài: Điều khiển Đèn Giao thông Cụm Giao lộ Quận 1, TP.HCM (Đường 1 Chiều & 2 Chiều)
Khu vực thực tế: Điện Biên Phủ (1 chiều) - Võ Thị Sáu (1 chiều) - Đinh Tiên Hoàng (2 chiều) - Mai Thị Lựu (2 chiều hỗn hợp)
Thuật toán: Coordinated Multi-Agent Double Q-Learning vs. Fixed-Time & Actuated Control
Chạy độc lập: python saigon_traffic_rl.py
"""

import numpy as np
import matplotlib.pyplot as plt

PCU_WEIGHTS = {
    "motorbike": 0.35,  # Chiếm ~74% lưu lượng, có rẽ trái liên tục tại nút Điện Biên Phủ - Đinh Tiên Hoàng
    "car": 1.00,        # Chiếm ~18% lưu lượng
    "bus": 2.50,        # Chiếm ~5% lưu lượng (ưu tiên vận tải hành khách công cộng)
    "truck": 1.50       # Chiếm ~3% lưu lượng (chỉ đi 1 chiều trên Mai Thị Lựu từ ĐBP -> Nguyễn Văn Giai)
}

class SaigonIntersectionClusterEnv:
    """
    Môi trường mô phỏng cụm 4 ngã tư thực tế tại Phường Tân Định - Đa Kao (Quận 1, TP.HCM):
      - Nút A: Võ Thị Sáu (1 chiều Đ->T) x Mai Thị Lựu (2 chiều hỗn hợp)
      - Nút B: Võ Thị Sáu (1 chiều Đ->T) x Đinh Tiên Hoàng (2 chiều có dải phân cách cứng)
      - Nút C: Điện Biên Phủ (1 chiều Đ->T) x Mai Thị Lựu (2 chiều hỗn hợp)
      - Nút D: Điện Biên Phủ (1 chiều Đ->T) x Đinh Tiên Hoàng (Đèn rẽ trái liên tục cho xe 2 bánh)
    """
    def __init__(self, scenario="morning_rush", moto_continuous_left=True, seed=42):
        self.rng = np.random.default_rng(seed)
        self.nodes = ["A", "B", "C", "D"]
        self.scenario = scenario
        self.moto_continuous_left = moto_continuous_left
        self.reset()

    def reset(self):
        self.queues = {k: np.array([3.5, 3.5], dtype=float) for k in self.nodes}
        self.phases = {k: 0 for k in self.nodes}       # 0: Xanh Đông->Tây (1 chiều), 1: Xanh Bắc<->Nam (2 chiều)
        self.green_time = {k: 10 for k in self.nodes}  # Thời gian giữ pha hiện tại (giây)
        self.t = 0
        return self._get_states()

    def _discretize(self, pcu_val):
        if pcu_val < 3.0: return 0   # Thấp
        if pcu_val < 7.5: return 1   # Vừa
        if pcu_val < 14.0: return 2  # Cao
        return 3                     # Kẹt cứng

    def _get_states(self):
        states = {}
        for k in self.nodes:
            q_ew = self._discretize(self.queues[k][0])
            q_ns = self._discretize(self.queues[k][1])
            up = 0
            if k == "A" and self.queues["B"][0] > 7.5: up = 1
            elif k == "C" and self.queues["D"][0] > 7.5: up = 1
            elif k == "B" and self.queues["D"][1] > 7.5: up = 1
            elif k == "D" and self.queues["B"][1] > 7.5: up = 1
            states[k] = (q_ew, q_ns, self.phases[k], up)
        return states

    def _sample_inflow_pcu(self, node_id):
        base_ew = {"A": 1.3, "B": 2.3, "C": 1.8, "D": 3.1}[node_id]
        base_ns = {"A": 1.2, "B": 2.1, "C": 1.3, "D": 2.0}[node_id]
        inflow_ew = max(0.0, self.rng.normal(base_ew, 0.40))
        inflow_ns = max(0.0, self.rng.normal(base_ns, 0.35))
        return inflow_ew, inflow_ns

    def step(self, actions):
        rewards = {}
        total_wait_pcu = 0.0
        total_passed_pcu = 0.0
        spill_B_to_A = 0.0
        spill_D_to_C = 0.0

        for k in ["D", "B", "C", "A"]:
            act = actions[k]
            if self.green_time[k] < 10:
                act = 0
            elif self.green_time[k] >= 45:
                act = 1

            switched = (act == 1)
            if switched:
                self.phases[k] = 1 - self.phases[k]
                self.green_time[k] = 5
                # Mất 3s đèn vàng chuyển pha làm hao hụt năng lực thông hành
                discharge_capacity = 2.4
            else:
                self.green_time[k] += 5
                discharge_capacity = 5.4

            in_ew, in_ns = self._sample_inflow_pcu(k)
            if k == "A": in_ew += spill_B_to_A * 0.55
            if k == "C": in_ew += spill_D_to_C * 0.60

            self.queues[k][0] += in_ew
            self.queues[k][1] += in_ns

            if self.phases[k] == 0:
                passed = min(self.queues[k][0], discharge_capacity)
                self.queues[k][0] -= passed
                if k == "D" and self.moto_continuous_left:
                    moto_turn = min(self.queues[k][0], 0.90)
                    self.queues[k][0] -= moto_turn
                    passed += moto_turn
                if k == "B": spill_B_to_A = passed
                if k == "D": spill_D_to_C = passed
            else:
                passed = min(self.queues[k][1], discharge_capacity * 0.92)
                self.queues[k][1] -= passed

            self.queues[k] = np.clip(self.queues[k], 0.0, 35.0)

            active_q = self.queues[k][self.phases[k]]
            other_q = self.queues[k][1 - self.phases[k]]
            queue_penalty = 1.1 * active_q + 1.6 * other_q
            switch_penalty = 3.5 if switched else 0.0
            green_wave_bonus = 3.0 if (
                (k == "A" and self.phases["A"] == 0 and self.phases["B"] == 0) or
                (k == "C" and self.phases["C"] == 0 and self.phases["D"] == 0)
            ) else 0.0

            rewards[k] = -(queue_penalty + switch_penalty) + 1.8 * passed + green_wave_bonus
            total_wait_pcu += (self.queues[k][0] + self.queues[k][1])
            total_passed_pcu += passed

        self.t += 5
        done = self.t >= 300
        info = {
            "avg_wait_sec": (total_wait_pcu / 4.0) * 1.45,
            "throughput_pcu": total_passed_pcu
        }
        return self._get_states(), rewards, done, info


def evaluate_baseline(mode="fixed", episodes=30):
    env = SaigonIntersectionClusterEnv(moto_continuous_left=True, seed=100)
    waits = []
    for _ in range(episodes):
        env.reset()
        done = False
        ep_w, steps = 0.0, 0
        while not done:
            actions = {}
            for k in env.nodes:
                if mode == "fixed":
                    target = 20 if env.phases[k] == 0 else 20
                    actions[k] = 1 if env.green_time[k] >= target else 0
                else:  # actuated
                    q_active = env.queues[k][env.phases[k]]
                    q_other = env.queues[k][1 - env.phases[k]]
                    actions[k] = 1 if (q_other > q_active * 1.35 + 2.5) else 0
            _, _, done, info = env.step(actions)
            ep_w += info["avg_wait_sec"]
            steps += 1
        waits.append(ep_w / steps)
    return float(np.mean(waits))


def run_experiment(num_episodes=200):
    fixed_wait = evaluate_baseline("fixed")
    actuated_wait = evaluate_baseline("actuated")

    env = SaigonIntersectionClusterEnv(moto_continuous_left=True, seed=42)
    nodes = env.nodes
    Q_A = {k: {} for k in nodes}
    Q_B = {k: {} for k in nodes}

    def get_q(Q_dict, node, s):
        if s not in Q_dict[node]:
            q_ew, q_ns, phase, up = s
            active_busy = q_ew if phase == 0 else q_ns
            other_busy = q_ns if phase == 0 else q_ew
            Q_dict[node][s] = np.array([
                (active_busy - other_busy) * 1.5 + (1.5 if up == 1 and phase == 0 else 0.0),
                (other_busy - active_busy) * 1.5 - 1.0
            ], dtype=float)
        return Q_dict[node][s]

    alpha, gamma, eps = 0.22, 0.90, 0.35
    marl_wait_history, marl_reward_history = [], []

    for ep in range(num_episodes):
        states = env.reset()
        done = False
        ep_wait, ep_reward, steps = 0.0, 0.0, 0

        while not done:
            actions = {}
            for k in nodes:
                s = states[k]
                qa = get_q(Q_A, k, s)
                qb = get_q(Q_B, k, s)
                if np.random.rand() < eps:
                    actions[k] = np.random.randint(0, 2)
                else:
                    actions[k] = int(np.argmax(qa + qb))

            next_states, rewards, done, info = env.step(actions)

            for k in nodes:
                s, a, r, s_next = states[k], actions[k], rewards[k], next_states[k]
                if np.random.rand() < 0.5:
                    a_star = int(np.argmax(get_q(Q_A, k, s_next)))
                    td_target = r + gamma * get_q(Q_B, k, s_next)[a_star]
                    get_q(Q_A, k, s)[a] += alpha * (td_target - get_q(Q_A, k, s)[a])
                else:
                    b_star = int(np.argmax(get_q(Q_B, k, s_next)))
                    td_target = r + gamma * get_q(Q_A, k, s_next)[b_star]
                    get_q(Q_B, k, s)[a] += alpha * (td_target - get_q(Q_B, k, s)[a])

            states = next_states
            ep_wait += info["avg_wait_sec"]
            ep_reward += sum(rewards.values())
            steps += 1

        eps = max(0.02, eps * 0.96)
        marl_wait_history.append(ep_wait / steps)
        marl_reward_history.append(ep_reward)

    final_marl_wait = float(np.mean(marl_wait_history[-40:]))
    print("=" * 62)
    print("KẾT QUẢ THỰC NGHIỆM ĐIỀU KHIỂN ĐÈN GIAO THÔNG CỤM QUẬN 1, TP.HCM")
    print("=" * 62)
    print(f"1. Fixed-Time (Chu kỳ cố định)       : {fixed_wait:.2f} giây/xe")
    print(f"2. Actuated Control (Cảm biến cục bộ): {actuated_wait:.2f} giây/xe")
    print(f"3. Coordinated MARL Double Q-Learning: {final_marl_wait:.2f} giây/xe")
    print(f"=> Giảm thời gian chờ: {((fixed_wait - final_marl_wait) / fixed_wait) * 100:.1f}% so với Fixed-Time")
    print("=" * 62)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(marl_wait_history, color="#0284c7", linewidth=1.8, label="MARL Double Q-Learning")
    ax1.axhline(fixed_wait, color="#ef4444", linestyle="--", label=f"Fixed-Time ({fixed_wait:.1f}s)")
    ax1.axhline(actuated_wait, color="#f59e0b", linestyle=":", label=f"Actuated ({actuated_wait:.1f}s)")
    ax1.set_title("Thời gian chờ trung bình qua các Episode (Cụm Q.1)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Thời gian chờ TB (giây/xe)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(marl_reward_history, color="#10b981", linewidth=1.8)
    ax2.set_title("Đường cong Hội tụ Cumulative Reward")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Tổng phần thưởng (Reward)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("saigon_traffic_rl_results.png", dpi=150)
    print("Đã xuất biểu đồ báo cáo ra file: saigon_traffic_rl_results.png")

if __name__ == "__main__":
    run_experiment(200)
