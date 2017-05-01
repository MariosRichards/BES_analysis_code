import matplotlib.pyplot as plt


def display_components(n_components, decomp, folder):
    import matplotlib.pyplot as plt
    n_comps = min(n_components,20)
    comp_labels = {}

    for comp_no in range(0,n_comps):

        fig, axes = plt.subplots(ncols=2)

        ax = axes[1]
        comp = pd.DataFrame( decomp.components_[comp_no], index = BES_reduced.columns, columns = ["components_"] )
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