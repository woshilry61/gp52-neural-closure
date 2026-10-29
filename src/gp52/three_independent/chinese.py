"""Chinese reading guide generated from the final numerical tables."""
import csv,json
from pathlib import Path
from .report import OUT,render_markdown
def main():
 rows=list(csv.DictReader((OUT/'summary.csv').open()))
 def pick(scenario,method,layout=0):
  return next(r for r in rows if r['blood']=='cfd' and r['scenario']==scenario and r['method']==method and int(r['layout'])==layout and r['noise']==('known_inlet' if scenario=='forward' else 'noisy'))
 a=pick('forward','C0');b=pick('forward','neural')
 ft='| 方法 | Th 误差 | Fg 误差 | Fn 误差 |\n| --- | --- | --- | --- |\n'
 for name,r in [('固定 C0',a),('冻结神经模型（三个种子平均）',b)]:ft+='| '+name+' | '+' | '.join(f'{100*float(r[k]):.2f}%' for k in ['Th','Fg','Fn'])+' |\n'
 it='| 测点数 | C0 入口误差 | 神经模型入口误差 | C0 Fn 误差 | 神经模型 Fn 误差 |\n| --- | --- | --- | --- | --- |\n'
 for n in [1,2,3]:
  c=pick('inverse','C0',n);d=pick('inverse','neural',n)
  it+=f'| {n} | '+' | '.join(f'{100*float(r[k]):.2f}%' for r,k in [(c,'inlet'),(d,'inlet'),(c,'Fn'),(d,'Fn')])+' |\n'
 conv=list(csv.DictReader((OUT/'reference-convergence.csv').open()));maxchange=max(float(r[k]) for r in conv for k in ['Th','Fg','Fn'])
 text=f"""# 三血管独立 CFD 验证：中文说明

## 结论

本轮验证已完成。已知入口时，冻结神经模型的浓度预测优于固定 C0，但一、二、三测点下的平均入口恢复误差都高于 C0。因此，目前不能把它称为全面超过数值法的最终模型。

## 这次到底做了什么

先用原始数值算例的边界和初始条件，独立求解三段轴对称血流 CFD，并在分叉处耦合流量与压力作用。血流参考不使用 Sen 的预测结果。再根据求出的速度场，计算 Th、Fg、Fn 的生化参考数据。

神经模型与 C0 共用同一套独立血流，接受同样的观测、噪声和评分规则。之后将两者的输入都换回 Sen 血流，保持参考数据不变，测量替换血流的影响。

模型没有重新训练，原有结果没有覆盖。本轮共评估 16 个测试入口，包含无噪声和三个噪声重复，以及一、二、三测点布局。

## 已知入口时，浓度预测怎么样

下面均为全血管、全时间窗口的相对误差，越低越好。

{ft}

## 不知道入口，只给少量测点时怎么样

反演使用 Th 和 Fg 的观测，不使用 Fn 观测。下表是三个噪声重复的平均结果。

{it}

入口误差评估 0–0.82 秒；浓度误差评估 0–1.23 秒。这里的 Fn 更准，不意味着入口曲线恢复得更准。无噪声情况下也出现同样的入口恢复劣势，不能只归因于观测噪声。

完整的 Th、Fg、Fn 反演误差及各条血管的误差见英文报告、summary.csv 和 per-vessel-errors.csv。曲线页面可切换全部测试算例、血流来源和测点数。

## 为什么比旧报告误差高

旧三血管训练参考先使用 Sen 的平均血流，再规定血管中心快、靠壁慢的抛物线速度分布，然后计算三种物质。

新 CFD 没有强制抛物线。它可以出现血管中心向前流、靠壁区域向后流。这时平均流量很小，内部物质输运仍然明显。当前输运闭合的强度与平均流速绝对值相乘，在平均流量为零时修正也会消失，这是现有结构的限制之一。

控制诊断中，只把独立 CFD 的径向速度重新换成同平均流量的抛物线，冻结网络的 Fn 误差就降到约 1.64%。这说明旧训练假设对原来的好成绩有明显影响。该诊断没有被当成新的真实参考。

## Sen 血流误差影响有多大

对照同一份独立生化参考，神经模型的已知入口 Fn 误差由独立血流下的 6.84% 变为 Sen 血流下的 7.02%。因此，本轮暴露的问题不能主要归咎于 Sen 血流。

还发现一个上游问题：旧守恒修正对较小的第二条子血管改动较大，使该血管的流量误差由修正前约 4.40% 变为修正后约 23.24%。本轮保留了这套原有处理，没有改权重或处理规则。两种生化方法始终使用相同的处理后血流。

## 参考数据可靠吗

血流网格、靠壁网格、化学网格、时间步和速度采样都做了加密检查。预先指定的三个核查算例中，三种浓度的最大相对变化约为 {100*maxchange:.2f}%。分叉守恒、化学守恒及 GPU/CPU 求解一致性也做了核对。

这仍是数值参考：上游仓库提供数值血流档案，Th、Fg、Fn 由我们根据设定的方程计算，没有真实生化测量。三段轴对称血管用理想分叉耦合，不包含完整三维分叉角度和涡流结构；初始径向速度也有明确设定。

## 训练与本轮验证的区别

旧三血管生化模型使用 24 个训练算例、8 个验证算例和 16 个测试算例。固定血流，先训练输运闭合，再训练反应闭合，各约 3,000 次主要迭代。

本轮所有网络权重均冻结。反演只调整入口浓度曲线的七个参数。新增独立 CFD 数据没有用于训练。

原有 269 个受保护文件核对一致，结果保存在新的独立验证目录中。
"""
 (OUT/'SUMMARY_ZH.md').write_text(text)
 html='<!doctype html><meta charset="utf-8"><title>三血管独立 CFD 验证</title><style>body{font:17px system-ui;line-height:1.7;max-width:1050px;margin:32px auto;padding:20px;color:#163342}table{width:100%;border-collapse:collapse;font-size:16px}td,th{padding:10px;border-bottom:1px solid #d7e3e8;text-align:left}th{background:#eaf3f6}h2{margin-top:32px}a{color:#087eaa}</style><p><a href="report.html">详细英文报告</a> · <a href="curves.html">切换查看全部浓度曲线</a></p>'+render_markdown(text)
 (OUT/'summary-zh.html').write_text(html)
if __name__=='__main__':main()
