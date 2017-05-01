import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab, cm


def display_components(n_components, decomp, folder, cols, BES_decomp):

    n_comps = min(n_components,20)
    comp_labels = {}

    for comp_no in range(0,n_comps):

        fig, axes = plt.subplots(ncols=2)

        ax = axes[1]
        comp = pd.DataFrame( decomp.components_[comp_no], index = cols, columns = ["components_"] )
        comp["comp_absmag"] = comp["components_"].abs()
        comp = comp.sort_values(by="comp_absmag",ascending=True)
        ax.set_xlabel("abs. variable coeffs")
        ax.set_title("Histogram of abs. variable coeffs")
        comp["comp_absmag"].hist( bins=30, ax=ax, figsize=(10,6) )

        # set top abs_mag variable to label
        comp_labels[comp_no] = comp.index[-1:][0] # last label (= highest magnitude)
        # if top abs_mag variable is negative
        if comp[-1:]["components_"].values[0] < 0:
            comp["components_"]         = -comp["components_"]
            decomp.components_[comp_no] = -decomp.components_[comp_no]
            BES_decomp[comp_no]         = -BES_decomp[comp_no]

        ax = axes[0]
        title = "Comp. "+str(comp_no)+" (" + comp.index[-1:][0] + ")"
        comp_labels[comp_no] = title
        ax.set_title( title )
        ax.set_xlabel("variable coeffs")
        xlim = (min(comp["components_"].min(),-1) , max(comp["components_"].max(),1) )
        comp["components_"].tail(30).plot( kind='barh', ax=ax,figsize=(10,6), xlim=xlim )

        fname = folder + title.replace("/","_") + ".png"

        fig.savefig( fname, bbox_inches='tight' )

        if comp_no >4:
            plt.close()
    return (BES_decomp, comp_labels)
    

def display_pca_data(n_components, decomp, BES_std):    
    
    figsz = (3,3)

    if hasattr(decomp, 'explained_variance_ratio_'):
        print('explained variance ratio (first 30): %s'
              % str(decomp.explained_variance_ratio_[0:30]) )
        
    if hasattr(decomp, 'explained_variance_'):
        print('explained variance (first 30): %s'
              % str(decomp.explained_variance_[0:30]) )
        plt.figure(figsize = figsz)
        plt.plot( range(1,n_components+1), decomp.explained_variance_, linewidth=2)
        plt.xlabel('n_components')
        plt.ylabel('explained_variance_') 
        
    if hasattr(decomp, 'noise_variance_'): 
        if isinstance(decomp.noise_variance_, float):
            print('noise variance: %s'
                  % str(decomp.noise_variance_) )
        
    if hasattr(decomp, 'score'):
        print('average log-likelihood of all samples: %s'
              % str(decomp.score(BES_std)) )
        
    if hasattr(decomp, 'score_samples') and not np.isinf( decomp.score(BES_std) ):
        pd.DataFrame( decomp.score_samples(BES_std) ).hist(bins=100,figsize = figsz)

    if hasattr(decomp, 'n_iter_'):
        print('number of iterations: %s'
              % str(decomp.n_iter_) )
        
    if hasattr(decomp, 'loglike_'):
        plt.figure(figsize = figsz)
        plt.plot( decomp.loglike_, linewidth=2 )
        plt.xlabel('n_iter')
        plt.ylabel('log likelihood') 

    if hasattr(decomp, 'error_'):
        plt.figure(figsize = figsz)
        plt.plot( decomp.error_, linewidth=2)
        plt.xlabel('n_iter')
        plt.ylabel('error')     
    
    
# xlim, ylim, samples, weights
def weighted_kde(xlim, ylim, samples, weights):

    #create mesh grid
    x      = np.linspace(xlim[0], xlim[1], 100)
    y      = np.linspace(ylim[0], ylim[1], 100)
    xx, yy = np.meshgrid(x, y)

    #Evaluate the kde on a grid
    pdf    = gaussian_kde(samples.values, weights=weights.values)
    zz     = pdf((np.ravel(xx), np.ravel(yy)))
    zz     = np.reshape(zz, xx.shape)

    # produce kdeplot
    vmax=abs(zz).max()
    vmin=-abs(zz).max()
    levels = np.arange(vmin*1.1, vmax*1.1, (vmax-vmin)/30)  # Boost the upper limit to avoid truncation errors.
    norm   = cm.colors.Normalize(vmax=vmax, vmin=vmin)
    cmap   = cm.PRGn
    cset1  = plt.contourf(xx, yy, zz, levels,
                     cmap=cm.get_cmap(cmap, len(levels) - 1),
                     norm=norm)
    plt.xlabel( comp_labels[x_axis] )
    plt.ylabel( comp_labels[y_axis] ) 
    plt.title('Decomposition of BES dataset; Overview')    
