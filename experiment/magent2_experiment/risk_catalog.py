from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RiskScenario:
    identifier: str
    name: str
    source: str
    activation: str
    oracle: str


OFFICIAL_SOURCE = "MAgent2 Battlefield公开环境说明与运行规则（v0.3.4）"

RISK_SCENARIOS: tuple[RiskScenario, ...] = (
    RiskScenario("S01", "无接敌搜索", OFFICIAL_SOURCE, "回合开始后连续3步红方观测中无蓝方", "动作及观测输出合法，回合状态可继续推进"),
    RiskScenario("S02", "发现敌方", OFFICIAL_SOURCE, "至少一个红方智能体在局部观测中发现蓝方", "观测维度和数值有效"),
    RiskScenario("S03", "有效攻击机会", OFFICIAL_SOURCE, "至少一个蓝方进入任一红方的8方向攻击范围", "动作属于0至20且奖励为有限数"),
    RiskScenario("S04", "协同攻击机会", OFFICIAL_SOURCE, "至少两个红方同时具备攻击同一蓝方的条件", "多代理动作、奖励和代理标识保持一致"),
    RiskScenario("S05", "障碍邻近交战", OFFICIAL_SOURCE, "接敌时至少一个红方邻近地图障碍或边界", "位置、动作和观测不越界"),
    RiskScenario("S06", "局部数量劣势", OFFICIAL_SOURCE, "以任一红方为中心的切比雪夫距离6格内，蓝方数量至少比红方多2", "环境仍返回全部存活代理的合法数据"),
    RiskScenario("S07", "低生命状态", OFFICIAL_SOURCE, "至少25%的存活红方生命比例低于0.4", "生命值有限且位于合法范围"),
    RiskScenario("S08", "生命恢复", OFFICIAL_SOURCE, "代理数不变时红方总生命值在连续步中增加", "生命恢复与公开规则一致且不超过上限"),
    RiskScenario("S09", "成员死亡后继续交战", OFFICIAL_SOURCE, "红方成员死亡后仍有红方与蓝方存活并继续执行", "死亡代理被移除，存活代理和终止标志同步"),
    RiskScenario("S10", "红方胜利", OFFICIAL_SOURCE, "蓝方被清除且红方仍有存活", "回合正确终止并给出有限奖励"),
    RiskScenario("S11", "红方失败", OFFICIAL_SOURCE, "红方被清除且蓝方仍有存活", "回合正确终止并给出有限奖励"),
    RiskScenario("S12", "周期上限终止", OFFICIAL_SOURCE, "达到max_cycles时双方仍有存活或结果为平局", "truncation与周期上限同步且无残留执行"),
)

SCENARIO_IDS = tuple(item.identifier for item in RISK_SCENARIOS)


@dataclass(frozen=True)
class RiskTestCase:
    environment_seed: int
    blue_policy: str
    map_size: int
    max_cycles: int
    deterministic: bool

    def as_tuple(self) -> tuple[object, ...]:
        return (self.environment_seed, self.blue_policy, self.map_size, self.max_cycles, self.deterministic)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RiskParameterDomain:
    environment_seeds: tuple[int, ...] = tuple(range(1000))
    blue_policies: tuple[str, ...] = ("random", "aggressive", "shared_model")
    map_sizes: tuple[int, ...] = (60, 80, 100)
    max_cycles_values: tuple[int, ...] = (20, 40, 80, 200)
    deterministic_values: tuple[bool, ...] = (True, False)

    def contains(self, case: RiskTestCase) -> bool:
        return (
            case.environment_seed in self.environment_seeds
            and case.blue_policy in self.blue_policies
            and case.map_size in self.map_sizes
            and case.max_cycles in self.max_cycles_values
            and case.deterministic in self.deterministic_values
        )


COMMON_RISK_CASES: tuple[RiskTestCase, ...] = (
    RiskTestCase(17, "random", 80, 80, True),
    RiskTestCase(193, "random", 80, 80, True),
    RiskTestCase(389, "random", 80, 80, True),
    RiskTestCase(617, "random", 80, 80, True),
    RiskTestCase(881, "random", 80, 80, True),
)


def coverage_at(traces: Sequence[Iterable[str]], k: int) -> int:
    covered: set[str] = set()
    for trace in traces[:k]:
        covered.update(trace)
    return len(covered.intersection(SCENARIO_IDS))


def protocol_episode_count(policy_count: int, method_count: int, repetitions: int, budget: int) -> int:
    return policy_count * method_count * repetitions * budget
