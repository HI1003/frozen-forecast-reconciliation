"""Compact result floats and add sMASE to Table 2 from locked full-precision metrics."""

# Resolve the approved project layout while retaining sealed source-path keys.
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[2]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
from src.support.paths.project import project_file

from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/tables/accuracy'
WORK=ROOT/'tmp/compact_results_presentation_20261023'
METRICS=ROOT/'results/metrics/all_level_metrics.csv'
METHODS=['base','OLS','WLS-struct','WLS-var','MinT-shrink','MO-L1','MO-L2','RF','LightGBM','Elasso_adapted',
         'BU','BUL','BUN','TD','TDL','TDN','HYB','HYBL','HYBN']

def sha(p):return hashlib.sha256(project_file(Path(p)).read_bytes()).hexdigest()


def source():
    m=json.loads((project_file(ROOT/'results/SHA256.json')).read_text(encoding='utf-8'))
    assert sha(METRICS)==m[METRICS.relative_to(ROOT).as_posix()]
    f=pd.read_csv(project_file(METRICS),float_precision='round_trip')
    assert len(f)==228 and not f.duplicated(['dataset','base','step','method']).any()
    assert set(f.method)==set(METHODS)
    assert f.groupby(['dataset','base','step']).size().eq(19).all()
    assert round(f.loc[f.method=='HYBN','coh_nmae'].mean(),3)==.345
    return f


def table(f,lang):
    zh=lang=='zh'
    caption=('十二个预测设置下的全层级测试 WAPE 与 sMASE。' if zh else
             'All-level test WAPE and sMASE across twelve forecasting settings.')
    note=(r'线性与 MLP 调和头、RF 和 LightGBM 的指标在种子 42、43、44 下分别计算后取算术平均；不平均预测。'
          r'Elasso 报告所选单次拟合，基础预测与确定性方法报告对应预测的指标。最终显示值保留小数点后三位。'
          r'每列最佳显示值以浅橙底加粗，次佳不同显示值以浅灰底加下划线，并列值统一标记。'
          r'Walmart 与 GEFCom2017F 的步长单位分别为天和小时。$\dagger$HYBN 使用 $\lambda=0$，平均相干 nMAE 为 0.345\%。' if zh else
          r'For linear and MLP heads, RF and LightGBM, metrics are computed separately for seeds 42, 43 and 44, then averaged arithmetically; predictions are not averaged. '
          r'Elasso reports its selected single fit; base forecasts and deterministic methods report their corresponding metrics. '
          r'Displayed values are rounded to three decimal places. The best displayed value in each column is bold on pale orange; '
          r'the second-best distinct displayed value is underlined on pale gray; displayed ties share a mark. Leads are days for Walmart and hours for GEFCom2017F. '
          r'$\dagger$HYBN uses $\lambda=0$; mean coherence nMAE is 0.345\%.')
    lines=[r'% Generated from locked original-precision metrics; no experiment or test changes.',
           r'\begin{table}[!htbp]',r'\centering',r'\begingroup',
           r'\renewcommand{\baselinestretch}{1}\small',
           r'\caption{'+caption+'}',r'\label{tab:main_wape}',
           r'\fontsize{8.5}{10}\selectfont',r'\renewcommand{\arraystretch}{0.95}',
           r'\setlength{\tabcolsep}{2pt}',
           r'\definecolor{accuracybest}{RGB}{252,232,207}',r'\definecolor{accuracysecond}{RGB}{237,237,237}',
           r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l*{12}{r}@{}}',r'\toprule']
    audit=[]
    for panel,(dataset,leads) in enumerate([('walmart',[1,7,14]),('gefcom2017f',[1,6,24])]):
        if panel:lines.append(r'\midrule')
        name='Walmart' if dataset=='walmart' else 'GEFCom2017F'
        lines.extend([
            r'\multicolumn{13}{l}{\textit{('+('a' if panel==0 else 'b')+') '+name+r'}} \\',
            ('方法' if zh else 'Method')+r' & \multicolumn{6}{c}{Chronos-2} & \multicolumn{6}{c}{TimesFM 2.5} \\',
            r'\cmidrule(lr){2-7}\cmidrule(l){8-13}',
            r' & \multicolumn{3}{c}{WAPE (\%) $\downarrow$} & \multicolumn{3}{c}{sMASE $\downarrow$}'
            r' & \multicolumn{3}{c}{WAPE (\%) $\downarrow$} & \multicolumn{3}{c}{sMASE $\downarrow$} \\',
            r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(l){11-13}',
            ' & '+' & '.join(f'$h={h}$' for h in leads*4)+r' \\',r'\midrule'])
        cols=[(base,metric,h) for base in ['chronos2','timesfm25'] for metric in ['wape','mase'] for h in leads]
        values={}
        for base,metric,h in cols:
            q=f[(f.dataset==dataset)&(f.base==base)&(f.step==h)].set_index('method')[metric].loc[METHODS]
            display={method:f'{value:.3f}' for method,value in q.items()}
            distinct=sorted(set(display.values()),key=float)
            values[(base,metric,h)]=(q,display,distinct[:2])
        for i,method in enumerate(METHODS):
            if i==10:lines.append(r'\midrule')
            label={'base':'未调和预测' if zh else 'Unreconciled',
                   'Elasso_adapted':'Elasso（适配）' if zh else 'Elasso (adapted)',
                   'HYBN':r'HYBN$\dagger$'}.get(method,method)
            cells=[]
            for base,metric,h in cols:
                q,display,top=values[(base,metric,h)]
                value=display[method];mark='none'
                if value==top[0]:
                    cell=r'\cellcolor{accuracybest}\textbf{'+value+'}';mark='best'
                elif len(top)>1 and value==top[1]:
                    cell=r'\cellcolor{accuracysecond}\underline{'+value+'}';mark='second'
                else:cell=value
                cells.append(cell)
                audit.append(dict(dataset=dataset,base=base,step=h,method=method,metric=metric,
                                  original_value=float(q[method]),display_value=value,highlight=mark))
            lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular*}',
                  r'\par\smallskip\begin{minipage}{\textwidth}\footnotesize '+note+r'\end{minipage}',
                  r'\endgroup',r'\end{table}'])
    return '\n'.join(lines)+'\n',pd.DataFrame(audit)


def main():
    import argparse
    parser=argparse.ArgumentParser(description="Render or check the final accuracy table from locked metrics.")
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output',type=Path,help='New directory; --check reads retained paper tables')
    args=parser.parse_args()
    frame=source()
    if not args.check and args.output is None: parser.error('rendering requires --output NEW_DIRECTORY')
    output=project_file(OUT) if args.check else args.output.resolve()
    if not args.check: output.mkdir(parents=True,exist_ok=False)
    for lang in ('en','zh'):
        text, cells=table(frame,lang)
        path=output/f'main_accuracy_{lang}.tex'
        if args.check:
            assert path.read_text(encoding='utf-8')==text, str(path)
        else:
            output.mkdir(parents=True,exist_ok=True)
            path.write_text(text,encoding='utf-8')
    print('Final bilingual accuracy tables verified.' if args.check else 'Final bilingual accuracy tables written.')

if __name__=='__main__':
    main()
