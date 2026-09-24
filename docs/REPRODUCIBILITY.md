# 复核与完整重跑

## 公开材料能够完成什么

在只有本仓库文件的情况下，可运行`python scripts/reproduce.py --no-plots`生成表5、覆盖曲线数值、首次全覆盖分布和最终12项统计检验。数值计算仅用Python标准库；图像输出另需`requirements-analysis.txt`。本仓库附带数据是既有执行结果的汇总，没有重新运行或补选测试用例。

## 完整重跑需要什么

完整重跑还需要：三个模型权重、设计阶段历史库、冻结清单及所有配置、奖励几何修正版清单。逐轨迹复核还需要原始执行记录和压缩轨迹。未公开材料须向作者申请，不能由公开汇总逆向还原。

取得原始完整项目后，以该完整项目的冻结文件及目录布局为准；不要把公开目录当成已具备原始数据的运行环境。`experiment/scripts/run_frozen.py`检查冻结输入哈希，`reward_correction.py`处理额外登记的修正版任务。原始程序没有授权因覆盖不足而补跑或换例。

`requirements-experiment.txt`记录原实验依赖；allpairspy 2.5.1的原副本及许可位于`experiment/vendor/`。原环境为Windows、Python 3.11.9。依赖清单不保证跨平台逐位一致的仿真行为，也不包含Python解释器本身。

原始冻结生成过程还检查过更早实验种子集合；该检查所涉及的旧实验目录未作为公开材料提供。当前资料保留最终冻结及来源，不声称可仅从公开代码重新建立全部历史过程。

## 版本优先级

1. 论文表5和公开`data/table5_reference.json`采用四种最终方法。
2. 主实验的random、combination、risk取自`results/analysis/suite_metrics.json`中batch=main的记录。
3. reward取自`results/reward_correction/suite_metrics.json`，不能替换为原主实验中的reward组。
4. 最终显著性结论以公开`scripts/reproduce.py`及`data/reference_statistics.json`的12项统一Holm口径为准。
5. `experiment/scripts/analyze_design.py`等原代码保留较早的分析分组，供方法溯源，不作为终稿统计复核命令。

## 许可与数据可用性

未公开材料不在Git下载范围内。取得文件的渠道和使用授权须另行确认，不将“可以查看仓库”等同于取得全部数据或无限制再分发许可。作者信息和正式许可尚未在本包中代填。
