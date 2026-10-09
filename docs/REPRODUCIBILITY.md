# 复核与正式实验重跑

本文档对应v1.2.0。模型、冻结用例、执行记录和标定统计输入自v1.1.0公开；v1.2.0增加严格P4判定及100种固定排列的重排复核。旧版v1.0.0只支持汇总复核。

## 公开材料能够完成什么

在只有本仓库文件的情况下，可运行`python scripts/reproduce.py --no-plots`生成表5、覆盖曲线数值、首次全覆盖分布和最终12项统计检验。数值计算仅用Python标准库；图像输出另需`requirements-analysis.txt`。本仓库附带数据是既有执行结果的汇总，没有重新运行或补选测试用例。

## 正式重跑材料

三个模型权重、设计阶段2352回合统计输入、最终用例清单及所有配置、奖励几何修正版清单、逐回合执行记录和压缩状态轨迹见`artifacts/`。`plan.json`仅包含终稿四种方法，配置、顺序和种子不变。

使用公开入口`scripts/replay_experiment.py`，不要直接运行旧的研究入口。原始冻结清单原样保留在`artifacts/provenance/`，包括未发布的内部文件和历史批次的哈希。公开入口依据单独的`plan.json`与`data/release_manifest.json`，不假装旧冻结清单中的全部文件均已公开。

原环境为Windows、Python 3.11.9、PyTorch 2.5.1 CPU、NumPy 1.26.4及MAgent2 0.3.4。创建Python 3.11虚拟环境后安装`requirements-replay.txt`，用于设计核验与执行；`requirements-experiment.txt`保留旧环境完整清单，其中包含文稿处理等非执行依赖。allpairspy 2.5.1的原副本及许可位于`experiment/vendor/`，脚本自动加入路径。跨平台或浮点后端差异应报告，不得换种子直到结果一致。

原始冻结生成过程还检查过更早实验种子集合；该检查所涉及的旧实验目录未作为公开材料提供。不声称重新建立全部历史过程，也不重新训练模型或执行全部标定。

## 从状态记录重算覆盖

```bash
python scripts/replay_experiment.py verify-records
```

检查完整任务网格、冻结参数和原状态轨迹文件哈希；先验证历史缓存与历史判定器一致，再依据`state_flags`重新统计SC，依据`state_trace`、终局标签及包括终止步在内的无M2约束重新匹配KPC。按冻结顺序还原全部1200组合，与已公开汇总逐项核对，并复核表5及12项统一Holm比较。

同一命令读取公开的100种排列，计算重排AUC，并与发布结果精确比较。输出写入`generated/verify-records/`：`recounted_combinations.json`为原顺序严格口径结果，`p4_corrections.json`为125条修正明细，`reordering_summary.json`为重排结果，`record_verification.json`为检查报告。此过程不执行仿真。原始缓存不能直接当作修正后的逐例路径集合。

状态轨迹主要为标签、阶段序列和迁移证据，不是完整逐步位置、生命值、观测张量和动作日志。本命令验证记录到覆盖的计算，不单独证明原始观测到每个状态谓词的全部判定。压缩包中的每个gzip轨迹文件与原归档逐字节一致；JSONL是从原执行文件按最终用例标识选择的原始行。

## 重建正式用例的配置与顺序

```bash
python scripts/replay_experiment.py verify-design
```

以候选目录、screen/refine两阶段2352回合统计输入调用原选择算法，对20套设计逐位置核对有效参数与目标；奖励组经过原几何重复修正后再比较。该过程重现基于既有统计输入的用例选择，不重跑标定。历史统计中`trace_file`指向标定时旧轨迹，该部分不随本包发布，选择算法不需要读取它。

本论文方法允许用户直接提供“目标路径—配置”关联及优先顺序，但本仓库公开材料所对应的这组定量实验采用既有离线统计输入。`verify-design`只验证实际采用的设计分支，不将用户直接给参数的分支描述为已单独验证。P4修正属于正式记录的评价修正，不反向重选用例、不重新排序，也不修改设计阶段已冻结的历史输入。

## 重新执行模型与冻结用例

```bash
python scripts/replay_experiment.py rerun --workers 4
python scripts/replay_experiment.py rerun --full --workers 4 --output generated/full-run
```

默认范围预先确定为四种方法设计0的前10例、所有3模型、首个环境种子，重复配置去重。`--full`运行公开计划的全部唯一配置、3模型、20种子，再重算1200组合。每套逻辑预算仍为30例，每例一次，未命中不补跑或换例。

模型使用`weights_only=True`加载，红方仍采用最大概率动作。新建环境并实际执行，不读取既有结果来跳过任务。输出必须是全新的`generated`子目录，存在则拒绝，不覆盖原始数据。旧记录仅在新回合结束后比较，不能影响策略或配置。失败保留已完成的新记录，不自动筛选重跑。

新结果经与存档复核相同的严格P4函数评价后，再检查覆盖集合、步数、终局和阶段序列，差异写入`execution_verification.json`并以错误退出。输出中`archived_covered_paths`保留历史阶段判定器的结果，`covered_paths`为当前严格口径；参考记录先从轨迹复核，不能用旧缓存直接比较。完整模式还核对全部组合与论文统计。实际完成的验收范围见`RELEASE_VALIDATION.md`，不能把少量执行检查表述为全量重跑。

## 版本优先级

1. 论文表5和公开`data/table5_reference.json`采用四种最终方法。
2. 主实验的random、combination、risk取自`results/analysis/suite_metrics.json`中batch=main的记录。
3. reward取自`results/reward_correction/suite_metrics.json`，不能替换为原主实验中的reward组。
4. 最终显著性结论以公开`scripts/reproduce.py`及`data/reference_statistics.json`的12项统一Holm口径为准。
5. `experiment/scripts/analyze_design.py`等原代码保留较早的分析分组，供方法溯源，不作为终稿统计复核命令。
6. 当前逐例P4判定与重排结果以v1.2.0公开入口及`results/reordering_summary.json`为准。旧版本只用于历史追溯。

## 许可与数据可用性

完整训练过程、标定全部逐步轨迹、未纳入终稿的早期实验以及论文和内部审批材料不在Git下载范围内。公开可访问不等于无限制再分发，第三方许可保留，研究材料的具体再分发许可由权利人确认。没有代为承诺单位审批完成，也未公开实际装备任务数据。
