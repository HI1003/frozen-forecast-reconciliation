"""Render the paper's mean ranks and primary non-rejection groups."""
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd

SOURCE = Path(__file__).resolve().parents[3]/'results/statistics/rank_groups'

def plot():
    ranks = pd.read_csv(SOURCE/'mean_ranks.csv').set_index('method')
    groups = pd.read_csv(SOURCE/'non_rejection_groups.csv')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig, axes = plt.subplots(2,1,figsize=(10.2,6.6))
    fig.subplots_adjust(left=.02,right=.98,top=.97,bottom=.03,hspace=.14)
    labels = {'base':'Unreconciled','Elasso_adapted':'Elasso (adapted)','HYBN':'HYBN†'}
    for ax,metric,title in zip(axes,['wape','mase'],['(a) WAPE','(b) sMASE']):
        order = ranks[metric].sort_values(kind='stable').index.tolist()
        nleft = 10
        ax.set(xlim=(-7.7,27.7),ylim=(3.5,-.47));ax.axis('off')
        ax.plot([1,19],[0,0],color='#222222',lw=1.0)
        for tick in range(1,20):
            ax.plot([tick,tick],[0,-.055],color='#222222',lw=.7)
            ax.text(tick,-.13,str(tick),ha='center',va='bottom',fontsize=8)
        ax.text(-7.3,-.27,title,fontsize=12,fontweight='bold')
        for i,m in enumerate(order):
            r = ranks.loc[m,metric]
            color = '#111111'
            if i<nleft:
                y=1.02+i*.25
                ax.plot([r,r,-.4],[0,y,y],lw=.8,color=color,zorder=1)
                ax.text(-.65,y,f'{labels.get(m,m)}  ({r:.2f})',ha='right',va='center',color=color,fontsize=10.5)
            else:
                y=1.02+(len(order)-1-i)*.25
                ax.plot([r,r,20.4],[0,y,y],lw=.8,color=color,zorder=1)
                ax.text(20.65,y,f'{labels.get(m,m)}  ({r:.2f})',ha='left',va='center',color=color,fontsize=10.5)
        selected=groups[(groups['mode']=='primary')&(groups.metric==metric)].sort_values('first_rank')
        for j,row in enumerate(selected.itertuples()):
            members=row.members.split('|')
            x=ranks.loc[members,metric].to_numpy()
            y=.17+j*.075
            ax.plot([x.min()-.045,x.max()+.045],[y,y],color='#111111',lw=3.6,solid_capstyle='butt',zorder=5)
        assert .17+len(selected)*.075 < .96, 'Too many groups for label clearance'
    for ext in ['png','pdf','svg']:
        fig.savefig(OUT/f'mean_rank_groups.{ext}',dpi=220,bbox_inches='tight',pad_inches=.08)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    OUT = parser.parse_args().output.resolve()
    OUT.mkdir(parents=True,exist_ok=False)
    plot()
