"""Render current Section 5 evidence without training, selection or new tests."""

# Resolve the approved project layout while retaining sealed source-path keys.
from pathlib import Path as _ProjectPath
import sys as _project_sys
_PROJECT_ROOT = _ProjectPath(__file__).resolve().parents[3]
_project_sys.path.insert(0, str(_PROJECT_ROOT))
from src.support.paths.project import project_file

from pathlib import Path
import argparse
import hashlib
import json
import platform

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'results/analysis/results_presentation'
METHODS = ['BU','BUL','BUN','TD','TDL','TDN','HYB','HYBL','HYBN']
DATASETS = [('walmart','Walmart',[1,7,14]),('gefcom2017f','GEFCom2017F',[1,6,24])]
METRICS = [('wape','WAPE'),('mase','sMASE')]
BLUE, TEAL, ORANGE = '#2868A2', '#338F86', '#CC7838'
INK, GRAY = '#263645', '#82919E'
TRADEOFF, SELECTION = '#CC3311', '#49454F'
KEY = ['dataset','base','step']
SOURCES = {}


def sha(path):
    return hashlib.sha256(project_file(path).read_bytes()).hexdigest()


def locked_csv(relative, expected):
    path = ROOT/relative
    assert sha(path) == expected, relative
    SOURCES[relative] = expected
    return pd.read_csv(project_file(path), float_precision='round_trip')


def export(fig, name):
    fig.savefig(project_file(OUT/f'{name}.pdf'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(project_file(OUT/f'{name}.svg'), bbox_inches='tight', pad_inches=.07)
    fig.savefig(project_file(OUT/f'{name}.png'), dpi=320, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)


def contrast_color(rgba):
    rgb = np.asarray(rgba[:3])
    linear = np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    luminance = float(linear @ [.2126,.7152,.0722])
    return '#111111' if luminance>.18 else 'white'


def style(ax, title, ylabel=None):
    ax.set_title(title,loc='left',fontsize=9.4,fontweight='bold',pad=9,color=INK)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#C1CBD3')
    ax.tick_params(length=0,pad=4)
    ax.grid(axis='y',color='#E4EAF0',lw=.55,zorder=0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    if ylabel:
        ax.set_ylabel(ylabel,labelpad=5)


def minimum_span(ax, metric):
    low,high = ax.get_ylim()
    span = {'wape':.10,'mase':.02}[metric]
    if high-low < span:
        mid = (low+high)/2
        ax.set_ylim(mid-span/2,mid+span/2)
    if metric == 'mase':
        ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))


def strategy_profiles(metrics):
    rows = []
    for metric,_ in METRICS:
        wide = metrics.pivot(index=KEY,columns='method',values=metric)
        for method in METHODS:
            for key,value in (100*(1-wide[method]/wide['base'])).items():
                rows.append(dict(zip(KEY,key),metric=metric,method=method,reduction=float(value)))
    effects = pd.DataFrame(rows)
    assert len(effects) == 216
    effects.to_csv(project_file(OUT/'strategy_setting_effects.csv'),index=False)
    prior = pd.read_csv(project_file(ROOT/'results/evidence/anchor/setting_relative_reductions.csv'))
    keys = KEY+['metric','method']
    a = effects.set_index(keys).reduction.sort_index()
    b = prior.set_index(keys).relative_reduction.sort_index()
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-12)
    means = effects.groupby(['dataset','metric','method'],sort=True).reduction.mean().reset_index()
    means.to_csv(project_file(OUT/'strategy_anchor_means.csv'),index=False)
    groups = [('BU / BUL / BUN',['BU','BUL','BUN'],TEAL,'-'),
              ('TD / TDL / TDN',['TD','TDL','TDN'],ORANGE,'--'),
              ('HYB / HYBL / HYBN'+chr(8224),['HYB','HYBL','HYBN'],BLUE,'-.')]
    fig,axes = plt.subplots(2,2,figsize=(7.2,4.7))
    fig.subplots_adjust(left=.12,right=.99,top=.89,bottom=.16,wspace=.24,hspace=.50)
    fig.text(.55,.985,'Means across Chronos-2 and TimesFM 2.5, three leads each',
             ha='center',va='top',fontsize=8.3,color=INK)
    ticks = [-30,-10,-3,-1,0,1,3,10]
    for row,(dataset,title,_) in enumerate(DATASETS):
        for col,(metric,name) in enumerate(METRICS):
            ax = axes[row,col]
            style(ax,f'({"abcd"[row*2+col]}) {title}: {name}')
            ax.set_yscale('symlog',linthresh=1,linscale=.7,base=10)
            ax.set_ylim(-35,15)
            ax.set_yticks(ticks,[str(v) for v in ticks])
            ax.minorticks_off()
            ax.axhline(0,color=GRAY,lw=.8,zorder=1)
            for group,(_,methods,color,linestyle) in enumerate(groups):
                x = np.arange(3)+(group-1)*.14
                averages = []
                for j,method in enumerate(methods):
                    q = effects[(effects.dataset==dataset)&(effects.metric==metric)&
                                (effects.method==method)].sort_values(['base','step'])
                    assert len(q)==6 and q.reduction.between(-35,15).all()
                    averages.append(q.reduction.mean())
                ax.plot(x,averages,color=color,ls=linestyle,lw=1.35,zorder=3)
                ax.scatter(x,averages,s=39,marker='D',facecolor='white',
                           edgecolor=color,lw=1.25,zorder=4)
            ax.set_xlim(-.38,2.38)
            ax.set_xticks([0,1,2],['No learning','Linear','Nonlinear'])
    fig.text(.025,.55,'Error reduction vs unreconciled (%)',rotation=90,
             va='center',ha='center',fontsize=8.4)
    handles = [Line2D([],[],color=color,ls=ls,lw=1.4,label=label,
                      marker='D',markerfacecolor='white',markersize=4)
               for label,_,color,ls in groups]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),
               ncol=3,frameon=False,fontsize=8,handlelength=2.5,columnspacing=1.4)
    export(fig,'strategy_anchor_profiles')
    return {'setting_effects_in_source_csv':216,'displayed_setting_points':0,'dataset_means':36,
            'y_scale':'symmetric logarithmic, linear within +/-1%, limits -35 to 15%',
            'x_categories':['direct reconciliation (no learning)','linear correction','nonlinear correction'],
            'means':'Arithmetic means of six setting-specific relative error reductions',
            'reference':'Original node-wise forecasts from each frozen base model before reconciliation',
            'direction':'Higher is better; zero is no change; predictions are not averaged across base models',
            'horizontal_offsets':[-.14,0,.14],
            'lines':'Connect categorical means within an anchor design; not fitted trends or intervals'}



def direct_residual(cells):
    zero = cells[(cells.level=='all')&(cells.lambda_value==0)]
    pairs = [('BUN (BU anchor)','bottom_mlp_residual','bottom_mlp_plain',TEAL),
             ('TDN (TD anchor)','td_features_mlp_residual','td_features_mlp_plain',ORANGE),
             ('HYBN (HYB anchor)','allnode_hyb_residual','allnode_mlp_plain',BLUE)]
    rows = []
    for metric,_ in METRICS:
        wide = zero.pivot(index=KEY,columns='arm',values=metric)
        for label,residual,direct,color in pairs:
            values = wide[[residual,direct]].dropna()
            assert len(values)==12
            for key,p in values.iterrows():
                value = 100*(1-p[residual]/p[direct])
                assert value > 0
                rows.append(dict(zip(KEY,key),metric=metric,label=label,direct_arm=direct,
                                 residual_arm=residual,reduction=float(value)))
    effects = pd.DataFrame(rows)
    effects.to_csv(project_file(OUT/'direct_residual_effects.csv'),index=False)
    old = pd.read_csv(project_file(ROOT/'results/evidence/selected/mapping_summary.csv')).set_index('family')
    for _,residual,_,_ in pairs:
        for metric,_ in METRICS:
            values = effects[(effects.residual_arm==residual)&(effects.metric==metric)].reduction
            np.testing.assert_allclose(values.mean(),old.loc[residual,metric+'_reduction'],rtol=0,atol=1e-12)
    means = effects.groupby(['dataset','metric','label'],sort=True).reduction.mean().reset_index()
    means.to_csv(project_file(OUT/'direct_residual_dataset_means.csv'),index=False)
    fig,axes = plt.subplots(2,2,figsize=(7.2,4.8))
    fig.subplots_adjust(left=.12,right=.99,top=.92,bottom=.19,wspace=.24,hspace=.72)
    styles = [('D','-'),('D','--'),('D','-.')]
    plotted = []
    for row,(dataset,title,leads) in enumerate(DATASETS):
        for col,(metric,name) in enumerate(METRICS):
            ax = axes[row,col]
            style(ax,f'({"abcd"[row*2+col]}) {title}: {name}')
            for group,(base,base_name) in enumerate([('chronos2','Chronos-2'),('timesfm25','TimesFM 2.5')]):
                x = np.arange(3)+group*4
                for k,(label,_,_,color) in enumerate(pairs):
                    q = effects[(effects.dataset==dataset)&(effects.metric==metric)&
                                (effects.base==base)&(effects.label==label)].set_index('step').loc[leads]
                    assert len(q)==3 and q.reduction.between(0,65).all()
                    ax.plot(x+(k-1)*.14,q.reduction.to_numpy(),color=color,ls=styles[k][1],
                            marker=styles[k][0],ms=4,mfc='white',mew=.95,lw=1.25,zorder=3)
                    plotted.extend((dataset,metric,base,int(h),label) for h in leads)
                ax.text(group*4+1,-.22,base_name,transform=ax.get_xaxis_transform(),
                        ha='center',va='top',fontsize=7.8)
            ax.set_xlim(-.40,6.40)
            ax.set_ylim(0,65)
            ax.set_yticks([0,20,40,60])
            ax.set_xticks([0,1,2,4,5,6],[str(h) for h in leads]*2)
            ax.axvline(3,color='#CFD8E0',lw=.6,ls=(0,(2,3)))
            ax.axhline(0,color=GRAY,lw=.75)
    assert len(plotted)==len(set(plotted))==len(effects)==72
    fig.text(.025,.57,'Error reduction vs direct MLP (%)',rotation=90,
             va='center',ha='center',fontsize=8.4)
    handles = [Line2D([],[],color=color,ls=styles[k][1],marker=styles[k][0],
                      markerfacecolor='white',lw=1.25,ms=4,label=label)
               for k,(label,_,_,color) in enumerate(pairs)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),
               ncol=3,frameon=False,fontsize=8,handlelength=2.3,columnspacing=1.3)
    export(fig,'direct_residual_effects')
    return {'setting_effects_in_source_csv':len(effects),'dataset_means_in_csv':12,
            'panels':4,'displayed_setting_points':len(plotted),'points_per_panel':18,
            'layout':'Rows are datasets; columns are metrics; base-model groups contain three leads each',
            'horizontal_offsets':[-.14,0,.14],
            'markers':'Hollow diamonds, matching Figure 3',
            'lines':'Connect only the three evaluated leads within the same base model and strategy',
            'intervals_or_ranges':False}



def lambda_curves(cells, selected):
    lam = cells[(cells.level=='all')&(cells.arm=='allnode_hyb_residual')]
    assert len(lam)==120
    summary = lam.groupby(['dataset','lambda_value'],sort=True).agg(wape=('wape','mean'),mase=('mase','mean'),coh_nmae=('coh_nmae','mean')).reset_index()
    summary.to_csv(project_file(OUT/'lambda_dataset_means.csv'),index=False)
    selected = selected[selected.arm=='allnode_hyb_residual'].set_index('dataset')
    assert selected['lambda'].to_dict()=={'walmart':.3,'gefcom2017f':.01}
    chosen = []
    fig,axes = plt.subplots(2,2,figsize=(7.2,4.5))
    fig.subplots_adjust(left=.09,right=.98,top=.87,bottom=.19,wspace=.30,hspace=.70)
    ticks = [0,.001,.003,.01,.03,.1,.3,1,3,10]
    labels = ['0','.001','.003','.01','.03','.1','.3','1','3','10']
    for row,(dataset,title,_) in enumerate(DATASETS):
        q = summary[summary.dataset==dataset].sort_values('lambda_value')
        value = float(selected.loc[dataset,'lambda'])
        pick = q[q.lambda_value==value].iloc[0]
        chosen.append(dict(dataset=dataset,lambda_value=value,wape=float(pick.wape),mase=float(pick.mase),coh_nmae=float(pick.coh_nmae)))
        for col,(metric,name) in enumerate(METRICS):
            ax = axes[row,col]
            style(ax,f'({"abcd"[row*2+col]}) {title}: {name}',name+' (%)' if metric=='wape' else name)
            ax.set_xscale('symlog',linthresh=.001,linscale=.55)
            ax.axhline(q.iloc[0][metric],color=GRAY,lw=.8,ls=(0,(3,3)))
            ax.plot(q.lambda_value,q[metric],color=TRADEOFF,lw=1.4,marker='o',ms=3,zorder=3)
            ax.axvline(value,color=SELECTION,ls=(0,(2,3)),lw=.8,alpha=.8)
            ax.scatter(value,pick[metric],marker='D',s=53,facecolor='white',edgecolor=SELECTION,lw=1.5,zorder=5)
            ax.text(.02,.94,rf'$\lambda_{{\mathrm{{val}}}}={value:g}$',transform=ax.transAxes,ha='left',va='top',fontsize=8,color=SELECTION)
            ax.set_xticks(ticks,labels,rotation=45,ha='right',fontsize=7.2)
            ax.set_xlabel(r'Fixed penalty weight, $\lambda$',fontsize=8,labelpad=3)
            ax.margins(x=.04,y=.18)
            minimum_span(ax,metric)
    handles = [Line2D([],[],ls='--',color=GRAY,label=r'$\lambda=0$ reference',lw=.8),
               Line2D([],[],marker='D',ls='',markerfacecolor='white',markeredgecolor=SELECTION,label='Validation choice',ms=5)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=2,frameon=False,fontsize=8)
    export(fig,'hybn_penalty_response')
    pd.DataFrame(chosen).to_csv(project_file(OUT/'validation_selected_test_metrics.csv'),index=False)
    return {'configurations':120,'points':40,'panels':4,'curve_colour':TRADEOFF,'validation_colour':SELECTION,
            'layout':'Rows are datasets; columns are metrics, matching Figures 3 and 4',
            'multiplier':'Fixed throughout each separate training run',
            'selection':'Previously frozen validation choices; no test selection','selected_test_metrics':chosen}


def budget_curves(budget):
    budget = budget.copy()
    assert len(budget)==108 and budget.level.eq('all').all()
    budget['epsilon_pct'] = budget.method.str.extract(r'HYBN-eps([\d.]+)',expand=False).astype(float)
    budget['validation_pass'] = budget.val_coh_nmae.le(budget.epsilon_pct+1e-8)
    budget['test_pass'] = budget.coh_nmae.le(budget.epsilon_pct+1e-8)
    summary = budget.groupby(['dataset','epsilon_pct'],sort=True).agg(wape=('wape','mean'),mase=('mase','mean'),coh_nmae=('coh_nmae','mean'),validation_pass=('validation_pass','sum'),test_pass=('test_pass','sum'),fits=('seed','size')).reset_index()
    summary.to_csv(project_file(OUT/'budget_dataset_summary.csv'),index=False)
    half = summary[summary.epsilon_pct==.5]
    assert half.validation_pass.sum()==36 and half.test_pass.sum()==32
    wide = budget.pivot(index=KEY+['seed'],columns='epsilon_pct',values=['wape','mase','coh_nmae'])
    for metric in ['wape','mase','coh_nmae']:
        np.testing.assert_array_equal(wide[metric][1],wide[metric][2])
    fig = plt.figure(figsize=(7.2,4.1))
    gs = fig.add_gridspec(2,2,left=.09,right=.98,top=.86,bottom=.13,wspace=.30,hspace=.65)
    for row,(dataset,title,_) in enumerate(DATASETS):
        q = summary[summary.dataset==dataset].sort_values('epsilon_pct')
        for col,(metric,name) in enumerate(METRICS):
            ax = fig.add_subplot(gs[row,col])
            style(ax,f'({"abcd"[row*2+col]}) {title}: {name}',name+' (%)' if metric=='wape' else name)
            ax.plot(q.epsilon_pct,q[metric],color=TRADEOFF,lw=1.4,marker='o',ms=4,zorder=3)
            ax.set_xticks([.5,1,2],['0.5','1','2'])
            ax.set_xlabel(r'Allowed coherence nMAE, $100\epsilon$ (%)',fontsize=8,labelpad=4)
            ax.set_xlim(.40,2.1)
            ax.margins(y=.22)
            minimum_span(ax,metric)
    export(fig,'hybn_budget_response')
    return {'fits':108,'panels':4,'feasibility_panels_displayed':False,'curve_colour':TRADEOFF,
            'layout':'Rows are datasets; columns are metrics, matching Figures 3-5',
            'multiplier':'Updated after each training epoch within a fixed-budget run',
            'validation_half_percent':36,'test_half_percent':32,'wider_budget_predictions_identical':'Retained locked evidence; matched metrics rechecked'}


def main():
    global OUT
    source_output=OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',nargs='+',type=int,choices=[3,4,5,6],default=[3,4,5,6])
    parser.add_argument('--output',type=Path,required=True,help='New directory for rendered figures')
    args = parser.parse_args()
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'Arial','font.size':8.3,'axes.labelsize':8.4,
                         'xtick.labelsize':7.6,'ytick.labelsize':7.6,'text.color':INK,
                         'axes.labelcolor':INK,'pdf.fonttype':42,'ps.fonttype':42,
                         'svg.fonttype':'none','axes.linewidth':.6})
    digests = json.loads((ROOT/'results/SHA256.json').read_text(encoding='utf-8'))
    def read(relative):
        return locked_csv(relative, digests[relative])
    metrics = read('results/metrics/all_level_metrics.csv')
    assert len(metrics) == 228
    cells = read('results/metrics/all_configurations.csv')
    budget = read('results/evidence/compact/budget_fit_rows.csv')
    audit = 'results/evidence/compact/record_audit.json'
    assert sha(ROOT/audit) == digests[audit]
    inactive = json.loads((ROOT/audit).read_text())['inactive_budgets']
    assert len(inactive)==24 and all(r['identical_test_predictions'] for r in inactive)
    relative = 'results/figures/section5/validation_selected_lambdas.csv'
    selected = read(relative)
    selection_receipt = {'source':relative,'sha256':digests[relative]}
    checks = {}
    renderers = {3:('strategy',lambda:strategy_profiles(metrics)),
                 4:('direct_residual',lambda:direct_residual(cells)),
                 5:('penalty',lambda:lambda_curves(cells,selected)),
                 6:('budget',lambda:budget_curves(budget))}
    for number in args.figures:
        key,render = renderers[number]
        checks[key] = render()

    for relative,digest in SOURCES.items():
        assert sha(ROOT/relative)==digest
    manifest = {'scope':'Section 5 presentation of previously verified results',
                'source_chain':'retained original-precision paper metrics -> these figures',
                'active_figures':{'3':'strategy_anchor_profiles.pdf','4':'direct_residual_effects.pdf',
                                  '5':'hybn_penalty_response.pdf','6':'hybn_budget_response.pdf'},
                'last_rendered_figures':args.figures,
                'training':False,'inference':False,'selection':False,'statistical_tests_recomputed':False,
                'input_sha256':SOURCES,'validation_selection_source':selection_receipt,
                'checks':checks,'script':'scripts/analysis/figures/render_paper_figures.py','script_sha256':sha(Path(__file__)),
                'render_environment':{'python':platform.python_version(),'matplotlib':matplotlib.__version__,
                                      'numpy':np.__version__,'pandas':pd.__version__},
                'output_sha256':{p.name:sha(p) for p in project_file(OUT).iterdir() if p.suffix in ['.pdf','.png','.svg','.csv']}}
    (project_file(OUT/'source_manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)


if __name__=='__main__':
    main()
