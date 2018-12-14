import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab, cm
import pickle, os
from gaussian_kde import gaussian_kde

import re

def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 

def amalgamate_waves(df, pattern, forward_fill=True):
    # euref_imm = amalgamate_waves(BES_reduced_with_na,"euRefVoteW",forward_fill=False)
    # assumes simple wave structure, give a pattern that works!
    df_cols_dict = {int(re.search("W(\d+)", x).groups()[0]):x for x in df.columns if re.match(pattern, x)}
    # sort columns
    df_cols = [df_cols_dict[x] for x in sorted(df_cols_dict.keys())]
    # forward fill and and pick last column - or backward fill and pick first column
    if forward_fill:
        latest_series = df[df_cols].fillna(method="ffill",axis=1)[df_cols[-1]]
    else:
        latest_series = df[df_cols].fillna(method="bfill",axis=1)[df_cols[0]]
    # if it's a category, retain category type/options/order
    if df[df_cols[0]].dtype.name == "category":
        latest_series = latest_series.astype(
                    pd.api.types.CategoricalDtype(categories = df[df_cols[0]].cat.categories) )
    # update name
    re.match("(.*?)W\d+","climateChangeW11").groups()[0]
    print("Amalgamating variables: ")
    print(df_cols_dict)
    name_stub = re.match("(.*?)W\d+",  list(df_cols_dict.values())[0]).groups()[0]
    latest_series.name = name_stub+"W"+"&".join([str(x) for x in sorted(df_cols_dict.keys())])
    
    return latest_series

import unicodedata
import string

valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
char_limit = 30

def clean_filename(filename, whitelist=valid_filename_chars, replace=' '):
    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')
    
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:
        print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]    

def create_subdir(base_dir,subdir):
    output_subfolder = base_dir + os.sep + clean_filename(subdir) + os.sep
    if not os.path.exists( output_subfolder ):
        os.makedirs( output_subfolder )
    return output_subfolder






def get_manifest(dataset_name, BES_file_manifest):
    manifest = BES_file_manifest[ BES_file_manifest["Name"] == dataset_name ]
    dataset_filename = manifest["Stata_Filename"].values[0]
    dataset_description = manifest["Friendlier_Description"].values[0]
    dataset_citation = manifest["Citation"].values[0]
    dataset_start = manifest["Date_Start"].values[0]
    dataset_stop = manifest["Date_Stop"].values[0]
    dataset_wave = manifest["Wave No"].values[0]
    return (manifest, dataset_filename, dataset_description, dataset_citation, dataset_start, dataset_stop, dataset_wave)


def get_small_files(data_subfolder, encoding):
    try:
        var_type    = pd.read_msgpack( data_subfolder + "var_type.msgpack")
    except:
        var_type    = pd.read_csv( data_subfolder + "var_type.csv", encoding=encoding)
        var_type.set_index("Unnamed: 0", inplace=True)
    print("var_type",  var_type.shape )

    fname = data_subfolder + "cat_dictionary.pkl"
    with open(fname, "rb") as f:
        cat_dictionary = pickle.load( f )

    fname = data_subfolder + "new_old_col_names.pkl"
    with open(fname, "rb") as f:
        new_old_col_names = pickle.load(f)
    old_new_col_names = {v: k for k, v in new_old_col_names.items()}
    return (var_type, cat_dictionary, new_old_col_names, old_new_col_names)

def hdf_shrink_to_msgpack(hdf_file):
    df = pd.read_hdf( data_subfolder + hdf_file+".hdf" )
    df = df.apply(pd.to_numeric,downcast='float')
    df.to_msgpack(data_subfolder + hdf_file+".hdf".replace('.hdf','.msgpack'))
    del df
    gc.collect()
    


def setup_directories():
    if os.getcwd().split(os.sep)[-1] != 'BES_analysis_code':
        raise Exception("Stop! You're in the wrong directory - should be in 'BES_analysis_code'")

    BES_code_folder   = "../BES_analysis_code/" # we should be here!
    BES_small_data_files = BES_code_folder + "small data files" + os.sep
    if not os.path.exists( BES_small_data_files ):
        os.makedirs( BES_small_data_files )

    # we should create these if they don't already exist
    BES_data_folder   = "../BES_analysis_data/"
    if not os.path.exists( BES_data_folder ):
        os.makedirs( BES_data_folder )

    BES_output_folder = "../BES_analysis_output/"
    if not os.path.exists( BES_output_folder ):
        os.makedirs( BES_output_folder )
        
    BES_file_manifest = pd.read_csv( BES_small_data_files + "BES_file_manifest.csv" )

    BES_R_data_files = BES_data_folder + "R_data" + os.sep
    if not os.path.exists( BES_R_data_files ):
        os.makedirs( BES_R_data_files )
    return (BES_code_folder, BES_small_data_files, BES_data_folder, BES_output_folder, BES_file_manifest, BES_R_data_files)



def display_components(n_components, decomp, cols, BES_decomp, manifest, 
                       save_folder = False, show_first_x_comps=4,
                       show_histogram=True, flip_axes=True):
    
    if hasattr(decomp, 'coef_'):
        decomp_components = decomp.coef_
    elif hasattr(decomp, 'components_'):
        decomp_components = decomp.components_
    else:
        raise ValueError('no component attribute in decomp')    

    # hardcoded at 20?    
    n_comps = min(n_components,20)
    comp_labels = {}
    comp_dict = {}

    for comp_no in range(0,n_comps):

        fig, axes = plt.subplots(ncols=1+show_histogram)
        
        comp = pd.DataFrame( decomp_components[comp_no], index = cols, columns = ["components_"] )
        comp["comp_absmag"] = comp["components_"].abs()
        comp = comp.sort_values(by="comp_absmag",ascending=True)        
        
        if show_histogram:
            comp_ax = axes[0]
            
            hist_ax = axes[1]
            hist_ax.set_xlabel("abs. variable coeffs")
            hist_ax.set_title("Histogram of abs. variable coeffs")
            comp["comp_absmag"].hist( bins=30, ax=hist_ax, figsize=(10,6) )
            
        else:
            comp_ax = axes
            
        # set top abs_mag variable to label
        comp_labels[comp_no] = comp.index[-1:][0] # last label (= highest magnitude)
        # if top abs_mag variable is negative
     
        if flip_axes & (comp[-1:]["components_"].values[0] < 0):

            comp["components_"]         = -comp["components_"]
            decomp_components[comp_no]  = -decomp_components[comp_no]
            BES_decomp[comp_no]         = -BES_decomp[comp_no]

        dataset_description = manifest["Friendlier_Description"].values[0]
        title = "Comp. "+str(comp_no)+" (" + comp.index[-1:][0] + ")"
        comp_labels[comp_no] = title
        comp_ax.set_title( dataset_description + "\n" + title )
        comp_ax.set_xlabel("variable coeffs")
        xlim = (min(comp["components_"].min(),-1) , max(comp["components_"].max(),1) )
        comp["components_"].tail(30).plot( kind='barh', ax=comp_ax, figsize=(10,6), xlim=xlim )
        dataset_citation = "Source: " + manifest["Citation"].values[0]

        if (save_folder != False):
            comp_ax.annotate(dataset_citation, (0,0), (0, -40),
                             xycoords='axes fraction', textcoords='offset points', va='top', fontsize = 7)            
            fname = save_folder + os.sep + title.replace("/","_").replace(":","_") + ".png"
            fig.savefig( fname, bbox_inches='tight' )

        comp_dict[comp_no] = comp
        # show first x components
        if (comp_no >= min(show_first_x_comps,n_components)):
            plt.close()

        
    return (BES_decomp, comp_labels, comp_dict)
    

def display_pca_data(n_components, decomp, BES_std):    
    
    figsz = (16,3)
    
    f, axs = plt.subplots( 1, 4, figsize=figsz )

    axno = 0
    
    if hasattr(decomp, 'explained_variance_ratio_'):
        print('explained variance ratio (first 30): %s'
              % str(decomp.explained_variance_ratio_[0:30]) )
        
    if hasattr(decomp, 'explained_variance_'):
        print('explained variance (first 30): %s'
              % str(decomp.explained_variance_[0:30]) )

        axs[axno].plot( range(1,n_components+1), decomp.explained_variance_, linewidth=2)
        # ,figsize = figsz)
        axs[axno].set_xlabel('n_components')
        axs[axno].set_ylabel('explained_variance_')
        axs[axno].set_title('explained variance by n_components')
        axno = axno + 1
        
    if hasattr(decomp, 'noise_variance_'): 
        if isinstance(decomp.noise_variance_, float):
            print('noise variance: %s'
                  % str(decomp.noise_variance_) )
        
    if hasattr(decomp, 'score'):
        print('average log-likelihood of all samples: %s'
              % str(decomp.score(BES_std)) )
        
    if hasattr(decomp, 'score_samples') and not np.isinf( decomp.score(BES_std) ):
        pd.DataFrame( decomp.score_samples(BES_std) ).hist(bins=100,figsize = figsz, ax=axs[axno])
        axs[axno].set_xlabel('log likelihood')
        axs[axno].set_ylabel('frequency')
        axs[axno].set_title('LL of samples')
        axno = axno + 1

    if hasattr(decomp, 'n_iter_'):
        print('number of iterations: %s'
              % str(decomp.n_iter_) )
        
    if hasattr(decomp, 'loglike_'):
        axs[axno].plot( decomp.loglike_, linewidth=2) # ,figsize = figsz)
        axs[axno].set_xlabel('n_iter')
        axs[axno].set_ylabel('log likelihood')
        axs[axno].set_title('LL by iter')
        axno = axno + 1

    if hasattr(decomp, 'error_'):

        axs[axno].plot( decomp.error_, linewidth=2, figsize = figsz)
        axs[axno].set_xlabel('n_iter')
        axs[axno].set_ylabel('error')
        axs[axno].set_title('LL by iter')
        axno = axno + 1
    
    
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
