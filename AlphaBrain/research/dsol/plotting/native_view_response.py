"""Measured yaw/pitch surfaces; interpolation is visibly separate from samples."""
from pathlib import Path
import numpy as np


def baseline_figure(summary, output):
    """Two measured-point maps for the complete panel; no spatial interpolation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);rows=summary['per_view']
    x=[r['yaw_deg'] for r in rows];y=[r['pitch_deg'] for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,values,title,cmap in [
        (axes[0],[100*r['success_rate'] for r in rows],'Task success (%)','viridis'),
        (axes[1],[r['mean_accel'] for r in rows],'Mean Accel (selection noises)','magma')]:
        dots=ax.scatter(x,y,c=values,s=150,cmap=cmap,edgecolors='white',linewidths=.5)
        ax.scatter([0],[0],marker='*',s=240,facecolors='none',edgecolors='red',linewidths=1.5)
        ax.set(xlabel='Yaw (degrees)',ylabel='Pitch (degrees)',title=title,xlim=(-35,35),ylim=(-24,24))
        ax.grid(alpha=.2);fig.colorbar(dots,ax=ax,shrink=.85)
    fig.suptitle('Native model: 40 tasks, 400 states; 27 measured views, no interpolation\nRed star: nominal view; these are panel averages, not within-state causal relations',fontsize=10)
    fig.savefig(output/'baseline-view-maps.png',dpi=160)
    fig.savefig(output/'baseline-view-maps.pdf')
    plt.close(fig)


def figures(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with PdfPages(output/'native-view-surfaces.pdf') as pdf:
        for index,state in enumerate(summary['states']):
            rows=state['views'];xy=np.array([r['angles_deg'] for r in rows]);a=np.array([r['primary_accel'] for r in rows]);s=np.array([r['success_rate'] for r in rows])
            fig=plt.figure(figsize=(12,5),layout='constrained')
            for position,z,title in [(1,a,'Accel (two fixed selection noises)'),(2,s,'Task success (four execution seeds)')]:
                ax=fig.add_subplot(1,2,position,projection='3d')
                ax.plot_trisurf(xy[:,0],xy[:,1],z,alpha=.3,color='#78a9c9',linewidth=.3)
                points=ax.scatter(xy[:,0],xy[:,1],z,c=s,cmap='viridis',vmin=0,vmax=1,s=35,depthshade=False)
                ax.set(xlabel='Yaw (deg)',ylabel='Pitch (deg)',zlabel=title,title=title)
                ax.scatter([0],[0],[z[0]],marker='*',s=150,color='red')
            fig.colorbar(points,ax=fig.axes,shrink=.65,label='Measured success fraction')
            fig.suptitle(f"{state['task']} / initial {state['initial']}\n27 measured poses; translucent surface is interpolation, not extra evidence",fontsize=10)
            pdf.savefig(fig);fig.savefig(output/f'state-{index:02d}.png',dpi=150);plt.close(fig)
