import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import mlab, cm
import pickle, os
from gaussian_kde import gaussian_kde
from IPython.display import display, display_html, HTML
import re

import sys, gc

global BES_code_folder, BES_small_data_files, BES_data_folder, BES_output_folder, BES_file_manifest, BES_R_data_files
global BES_Panel

def pretty_print(df):
    return display( HTML( df.to_html().replace("\\n","<br>") ) )

def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def memory_use(locs = locals().items()):
    gc.collect()
    # locals().items()
    for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locs),
                             key= lambda x: -x[1])[:10]:
        print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))

# helper functions
#global best_weight_series
#def weighted_mean(series):
#    return (series*best_weight_series.loc[series.index]).sum()/(best_weight_series.loc[series.index]).sum()


# quietly drops row with np.nan weights!    
def weighted_mean(x, **kws):
    val, weight = map(np.asarray, zip(*x))
    mask = (~np.isnan(val))&(~np.isnan(weight))
    val, weight = val[mask],weight[mask]
    return (val * weight).sum() / weight.sum()
        

def datetime_weighted_mean(x, **kws):
    val, weight = map(np.asarray, zip(*x))
    val = pd.Series(val).apply(lambda x: x.timestamp() if pd.notnull(x) else np.nan)
    mask = (~np.isnan(val))
    val, weight = val[mask],weight[mask]
    result = (val * weight).sum() / np.sum(weight)
#     raise Exception
    result = datetime.fromtimestamp(result,tz=pytz.timezone('GMT')) if pd.notnull(result) else np.nan  # turn back from timestamp
    return result    
    

from pandas._libs.lib import is_integer




def weighted_qcut(values, weights, q, **kwargs):
    if is_integer(q):
        quantiles = np.linspace(0, 1, q + 1)
    else:
        quantiles = q
        
    if values.isnull().sum()>0:
        raise Exception("nans in values")
        
    if weights.isnull().sum()>0:
        raise Exception("nans in weights")
        
    order = weights.loc[weights.index[values.argsort()]].cumsum()
    bins = pd.cut(order / order.iloc[-1], quantiles, **kwargs)
    return bins.sort_index()

def weighted_value_counts(x, wts, *args, **kwargs):
    normalize = kwargs.get('normalize', False)
    perc = kwargs.get('perc', False)
    decimal_places = kwargs.get('decimal_places', 2)
    suppress_raw_samplesize = kwargs.get('suppress_raw_samplesize', False)
    
    ascending = kwargs.get('ascending', True)
    if not x.name:
        x.name = "value"
    c0 = x.name 
    c1 = wts.name
    df = pd.concat([x,wts],axis=1)
    xtmp = df.groupby(c0).agg({c1:'sum'}).sort_values(c1,ascending=False)
    s = pd.Series(index=xtmp.index, data=xtmp[c1], name=c0)
    s.name = "weighted_sample_size"
    if normalize:
        s = s / df[c1].sum()
        s.name = "weighted_sample_fraction"
    if normalize and perc:
        s = s*100
        s.name = "weighted_sample_percentage"
    s = s.round(decimal_places)
    if decimal_places==0:
        s=s.astype('int')
        
    if not suppress_raw_samplesize:
        output = pd.DataFrame([s,x[wts.notnull()].value_counts()]).T
        output.columns = [s.name,"raw_sample_size"]
        output.index.name = x.name
        output.sort_values(by=s.name,inplace=True, ascending=ascending)
    else:
        output = s
    return output
    
    
# CHEATY FIX FOR WEIGHTING SEABORN KDES BEFORE THEY FIX SEABORN TO PASS WEIGHTS

# so we take in a series of weights - either assumes/force it to be non-null floats [0-inf)
# flip coins for the fractional parts of the weights to round up/down proportionately
# then replicate rows on the basis of the resulting weights

def lazy_weighted_indices(weights):
    x = weights.apply(lambda x: np.floor(x) if (np.random.rand() > x%1) else np.ceil(x)).astype('int')
    return flatten( [[weights.index[ind]]*x.values[ind] for ind in range(weights.shape[0])] )




def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in set(lst1) if value in temp] 
    return lst3 

def amalgamate_waves(df, pattern, forward_fill=True, specify_wave_order = None, low_priority_values = [], match=True):
    # euref_imm = amalgamate_waves(BES_reduced_with_na,"euRefVoteW",forward_fill=False)
    # assumes simple wave structure, give a pattern that works!
    
    ## BUG - we need to check that categories are the same for all columns!!!
    
    
    if match:
        df_cols_dict = {int(re.search("W(\d+)", x).groups()[0]):x for x in df.columns if re.match(pattern, x)}
    else:
        df_cols_dict = {int(re.search("W(\d+)", x).groups()[0]):x for x in df.columns if re.search(pattern, x)}
    # sort columns
    if specify_wave_order is not None:
        df_cols = [df_cols_dict[x] for x in specify_wave_order]
    else:
        df_cols = [df_cols_dict[x] for x in sorted(df_cols_dict.keys())]
    
    # forward fill and and pick last column - or backward fill and pick first column
    if len(df_cols)<=1:
        raise Exception("Can't amalgamate less than two variables!")
    if forward_fill:
        pick_col = -1
        method = "ffill"        
    else:
        pick_col = 0
        method = "bfill"
    latest_series = df[df_cols]\
        .replace(low_priority_values,[np.nan]*len(low_priority_values))\
        .fillna(method=method,axis=1)[df_cols[pick_col]]
    # stop values in low_priority from cascading, overwrite where possible, otherwise reinsert where there are only nans 
    categories = df[df_cols[0]].cat.categories
    if low_priority_values:
        high_priority_values = [x for x in categories if x not in low_priority_values]
        low_priority_series = df[df_cols]\
                .replace(high_priority_values,[np.nan]*len(high_priority_values))\
                .fillna(method=method,axis=1)[df_cols[pick_col]]
        low_priority_series = low_priority_series.cat.set_categories(latest_series.cat.categories)
        # apply doesn't trigger on NaN values!
        low_priority_mask = low_priority_series.apply(lambda x: x in low_priority_values).astype('bool').fillna(False) & latest_series.isnull()
        latest_series.loc[low_priority_mask] = low_priority_series.loc[low_priority_mask]
    # if it's a category, retain category type/options/order
    if df[df_cols[0]].dtype.name == "category":
        latest_series = latest_series.astype(
                    pd.api.types.CategoricalDtype(categories) )
    # update name
    # re.match("(.*?)W\d+","climateChangeW11").groups()[0]
    print("Amalgamating variables: ")
    print(df_cols_dict,df_cols)
    name_stub = re.match("(.*?)W\d+",  list(df_cols_dict.values())[0]).groups()[0]
    latest_series.name = name_stub+"W"+"&".join([str(x) for x in sorted(df_cols_dict.keys())])
    
    return latest_series

import unicodedata
import string



valid_filename_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
def clean_filename(filename, whitelist=valid_filename_chars, replace=' ', char_limit = 30):
    import warnings
    # replace spaces
    for r in replace:
        filename = filename.replace(r,'_')
    
    # keep only valid ascii chars
    cleaned_filename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore').decode()
    
    # keep only whitelisted chars
    cleaned_filename = ''.join(c for c in cleaned_filename if c in whitelist)
    if len(cleaned_filename)>char_limit:

        warnings.warn("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
        # print("Warning, filename truncated because it was over {}. Filenames may no longer be unique".format(char_limit))
    return cleaned_filename[:char_limit]    

def create_subdir(base_dir, subdir, char_limit=50):
    output_subfolder = base_dir + os.sep + clean_filename(subdir, char_limit=char_limit) + os.sep
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


def get_small_files(data_subfolder, encoding ):
    try:
        var_type    = pd.read_msgpack( data_subfolder + "var_type.msgpack")
    except:
        var_type    = pd.read_csv( data_subfolder + "var_type.csv", encoding=encoding)
        var_type.set_index("Unnamed: 0", inplace=True)
    print("var_type",  var_type.shape )

    fname = data_subfolder + "cat_dictionary.pkl"
    with open(fname, "rb") as f:
        cat_dictionary = pickle.load( f )

    try:
        fname = data_subfolder + "new_old_col_names.pkl"
        with open(fname, "rb") as f:
            new_old_col_names = pickle.load(f)
        old_new_col_names = {v: k for k, v in new_old_col_names.items()}
    except:
        print("old_new_col_names not loaded")
        old_new_col_names = None
        new_old_col_names = None
        
    return (var_type, cat_dictionary, new_old_col_names, old_new_col_names)

def hdf_shrink_to_msgpack(hdf_file):
    df = pd.read_hdf( data_subfolder + hdf_file+".hdf" )
    df = df.apply(pd.to_numeric,downcast='float')
    df.to_msgpack(data_subfolder + hdf_file+".hdf".replace('.hdf','.msgpack'))
    del df
    gc.collect()
    


def setup_directories():
   # if os.getcwd().split(os.sep)[-1] != 'BES_analysis_code':
   #     raise Exception("Stop! You're in the wrong directory - should be in 'BES_analysis_code'")
    if 'BES_analysis_code' not in os.getcwd().split(os.sep):
        raise Exception("Stop! You're in the wrong directory - should be in 'BES_analysis_code' *or a child directory*")   

    depth = len(os.getcwd().split(os.sep)) - os.getcwd().split(os.sep).index('BES_analysis_code')
    base = "".join(([os.pardir+os.sep]*depth))

    BES_code_folder   = base+"BES_analysis_code"+os.sep # we should be here!
    BES_small_data_files = BES_code_folder + "small data files" + os.sep
    if not os.path.exists( BES_small_data_files ):
        os.makedirs( BES_small_data_files )

    # we should create these if they don't already exist
    BES_data_folder   = base+"BES_analysis_data"+os.sep
    if not os.path.exists( BES_data_folder ):
        os.makedirs( BES_data_folder )

    BES_output_folder = base+"BES_analysis_output"+os.sep
    if not os.path.exists( BES_output_folder ):
        os.makedirs( BES_output_folder )
        
    BES_file_manifest = pd.read_csv( BES_small_data_files + "BES_file_manifest.csv",encoding = "ISO-8859-1" )

    BES_R_data_files = BES_data_folder + "R_data" + os.sep
    if not os.path.exists( BES_R_data_files ):
        os.makedirs( BES_R_data_files )
    return (BES_code_folder, BES_small_data_files, BES_data_folder, BES_output_folder, BES_file_manifest, BES_R_data_files)



def display_components(n_components, decomp, cols, BES_decomp, manifest=None, 
                       save_folder = False, show_first_x_comps=4,
                       show_histogram=True, flip_axes=True, max_comp=20, max_var_per_comp = 30):
    
    if hasattr(decomp, 'coef_'):
        decomp_components = decomp.coef_
    elif hasattr(decomp, 'components_'):
        decomp_components = decomp.components_
    else:
        raise ValueError('no component attribute in decomp')    

    # hardcoded at 20?    
    n_comps = min(n_components,max_comp)
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

        title = "Comp. "+str(comp_no)+" (" + comp.index[-1:][0] + ")"
        comp_labels[comp_no] = title            
        if manifest is not None:
            dataset_description = manifest["Friendlier_Description"].values[0]
            comp_ax.set_title( dataset_description + "\n" + title )
        else:
            comp_ax.set_title( title )
        comp_ax.set_xlabel("variable coeffs")
        xlim = (min(comp["components_"].min(),-1) , max(comp["components_"].max(),1) )
        comp["components_"].tail(max_var_per_comp).plot( kind='barh', ax=comp_ax, figsize=(10,6), xlim=xlim )
        if manifest is not None:
            dataset_citation = "Source: " + manifest["Citation"].values[0]
            comp_ax.annotate(dataset_citation, (0,0), (0, -40),
                             xycoords='axes fraction', textcoords='offset points', va='top', fontsize = 7)            

        if (save_folder != False):
            fname = save_folder + clean_filename(title) + ".png"
            fig.savefig( fname, bbox_inches='tight' )

        comp_dict[comp_no] = comp
        # show first x components
        if (comp_no >= min(show_first_x_comps,n_components)):
            plt.close()

        
    return (BES_decomp, comp_labels, comp_dict)
    

def display_pca_data(n_components, decomp, BES_std, y=[]):    
    
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
        if len(y)==0:
            print('average log-likelihood of all samples: %s'
                  % str(decomp.score(BES_std)) )
        else:
            print('mean classification accuracy (harsh if many cats.): %s'
                  % str(decomp.score(BES_std, y)) )
        
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

    
from scipy.stats import pearsonr, spearmanr
def corr_simple_pearsonr(df1,df2, mask=1, round_places=2):
    mask = df1.notnull()&df2.notnull()&mask
    (r,p) = pearsonr(df1[mask],df2[mask])
    return [round(r,round_places), round(p,round_places), mask.sum()]

def corr_simple_spearmanr(df1,df2, mask=1, round_places=2):
    mask = df1.notnull()&df2.notnull()&mask
    (r,p) = spearmanr(df1[mask],df2[mask])
    return [round(r,round_places), round(p,round_places), mask.sum()]

def get_pruned(x):
    if x in new_old_col_names.keys():
        x = new_old_col_names[x]
    if x in var_type.index:
        x = var_type.loc[x,"pruned"]
    return x

# case sensitive search/match
# return with notnull count
# accept a mask to filter notnulls by

flatten = lambda l: [item for sublist in l for item in sublist]

def match(df, pattern, case_sensitive=False, mask=None):
    if mask is None:
           mask = pd.Series(np.ones( (df.shape[0]) ) , index=df.index).astype('bool')
    if case_sensitive:
        return df.loc[mask,[x for x in df.columns if re.match(pattern, x)]].notnull().sum()
    else:
        return df.loc[mask,[x for x in df.columns if re.match(pattern, x, re.IGNORECASE)]].notnull().sum()

def search(df, pattern, case_sensitive=False, mask=None):
    if mask is None:
        mask = pd.Series(np.ones( (df.shape[0]) ), index=df.index).astype('bool')
    if case_sensitive:
        return df.loc[mask,[x for x in df.columns if re.search(pattern, x)]].notnull().sum()
    else:
        return df.loc[mask,[x for x in df.columns if re.search(pattern, x, re.IGNORECASE)]].notnull().sum()

def remove_wave(x):
    return re.sub("(W\d+)+","",x)    

#cat_col_mar_df = pd.read_csv(BES_small_data_files+"legend_colour_marker_dict.csv",index_col=0)
#cat_col_mar_df

#col_str = 'rbmkgcy'
#mar_str = ".,ov^<>8spP*hH+xXDd|_1234"
#global colours, markers
#colours = cycle(col_str)
#markers = cycle(mar_str)


from itertools import cycle
global colours, markers
def get_cat_col_mar(label, BES_small_data_files):
   # global colours, markers
#    global BES_code_folder, BES_small_data_files, BES_data_folder, BES_output_folder, BES_file_manifest, BES_R_data_files
    col_str = 'rbmkgcy'
    mar_str = ".,ov^<>8spP*hH+xXDd|_1234"    
# first use
    #print("GOT HERE!!!!")
    if 'cat_col_mar_df' not in globals():
        #print("INSIDE IF STATEMENT!!!")
        global cat_col_mar_df
        cat_col_mar_df = pd.read_csv(BES_small_data_files+"legend_colour_marker_dict.csv",index_col=0)
      #  global colours, markers
        colours = cycle(col_str)
        markers = cycle(mar_str)        
        
        
    if label in cat_col_mar_df.index:
        row = cat_col_mar_df.loc[label]
        return (row["colour"],row["marker"])
    else:
        (col, mar) = get_next_col_mar()
        count = 0
        while col_mar_comb_already_exists(col+mar):
            (col, mar) = get_next_col_mar()
            count = count+1
            if count>=(len(col_str)-1)*(len(mar_str)-1):
                raise Exception("stuck hunting for next col mar combinations!")
        cat_col_mar_df.loc[label] = [col, mar]
        cat_col_mar_df.to_csv(BES_small_data_files+"legend_colour_marker_dict.csv")
        return (col, mar)
           
def get_next_col_mar():
    return (next(colours),next(markers))
    
    
def col_mar_comb_already_exists(colmar):
    return colmar in (cat_col_mar_df["colour"] + cat_col_mar_df["marker"]).values
    
    
def drop_zero_var(df):
    # remove zero variance columns
    df = df.drop( df.columns[df.var()==0] , axis=1)
    return df
    
global num_to_weight
def get_weights(dataset_name, BES_Panel):
    max_wave = int(re.match("W(\d+)_?",dataset_name).groups()[0])
    num_to_wave = {x:"W"+str(x) for x in range(1,max_wave+1)}
    ## problem here if it's *not* a combined panel!
    num_to_weight = { y:[x for x in BES_Panel.columns.sort_values(ascending=False) if re.match("wt_(new|full)_W"+str(y)+"($|_result$)",x)][0] for y in range(1,max_wave+1) }
    weights = BES_Panel[list(num_to_weight.values())].copy()
    return max_wave, num_to_wave, num_to_weight, weights
    
from sklearn.preprocessing import StandardScaler
def standard_scale(df):
    return pd.DataFrame( StandardScaler().fit_transform(df.values ),
                         columns = df.columns,
                         index   = df.index      )

                         
                         
def trim_strings(x):
    if len( x.split("\n") )>1:
        return x.split("\n")[0] + "[...]"
    else:
        return x                         
                         
def display_corr(df, name, corr_type, top_num = 20, round_places = 2,
                 correlation_text = "r", p_value_text = "p", sample_size_text = "N",
                 text_wrap_length=50):
#     df.index = [x[0:60] for x in df.index]
    df.index =  [trim_strings(x) for x in df.index.str.wrap(width = text_wrap_length)]
    
    df[correlation_text] = df[correlation_text].round(round_places)
    
    df1 = df.sort_values(by=correlation_text, ascending=False)[0:top_num][[correlation_text,p_value_text,sample_size_text]]
    df2 = df.sort_values(by=correlation_text)[0:top_num][[correlation_text,p_value_text,sample_size_text]]
    
    df1[p_value_text]     = df1[p_value_text].apply(lambda x: "{0:0.2f}".format(x))
    df2[p_value_text]     = df2[p_value_text].apply(lambda x: "{0:0.2f}".format(x))

    df1_caption = "Top "+str(top_num)+ " positive "+"("+corr_type+")"+" correlations for "+name
    df2_caption = "Top "+str(top_num)+ " negative "+"("+corr_type+")"+" correlations for "+name

    df1_styler = df1.style.set_table_attributes("style='display:inline'").set_caption(df1_caption)
    df2_styler = df2.style.set_table_attributes("style='display:inline'").set_caption(df2_caption)

    display_html(df1_styler._repr_html_().replace("\\n","<br />")+df2_styler._repr_html_().replace("\\n","<br />"), raw=True)


def make_corr_summary(input_df, name,  corr_type = "spearman", pattern=None, sample_size_text = "N", correlation_text = "r",
                      abs_correlation_text = "abs_r", p_value_text = "p",
                      min_p_value = 0.01, min_variance = 0.0, min_sample_size = 500):

    if pattern is None:
        pattern=name
    #df1 = input_df
    focal_var = input_df[name]
    focal_mask = focal_var.notnull()


    pattern_list = [x for x in input_df.columns if re.search(pattern,x)]

    variances = input_df[focal_mask].astype('float32').var()
    low_var_list = list(variances[variances<min_variance].index)
    sample_sizes = input_df[focal_mask].notnull().sum()
    low_sample_size_list = list(sample_sizes[sample_sizes<min_sample_size].index)

    drop_list = pattern_list+low_var_list+low_sample_size_list


    if corr_type == "pearson":
        df = input_df.drop(drop_list,axis=1).astype('float32').apply(lambda x: corr_simple_pearsonr(x,focal_var)).apply(pd.Series)
    elif corr_type == "spearman":
        df = input_df.drop(drop_list,axis=1).astype('float32').apply(lambda x: corr_simple_spearmanr(x,focal_var)).apply(pd.Series)

    if len(df.columns)!=3:
        df=df.T
    df.columns = [correlation_text,p_value_text,sample_size_text]
 
    df[sample_size_text] = df[sample_size_text].astype('int')
    df[abs_correlation_text] = df[correlation_text].abs()

    zero_var_other_way_around_list = list(df[df[correlation_text].isnull()].index)
    df.dropna(inplace=True)

    insignificant_list = df[df[p_value_text]>min_p_value].index
    df.drop(insignificant_list,inplace=True)

    df.sort_values(by=abs_correlation_text,ascending=False,inplace=True)


    stub_dict = {}
    drop_list = []
    # drop repeated references to same variable in different waves???
    # so, what about different categories??? eg. blahWX_subcat
    # how about, just replace wave match as "X"
    # create a dictionary keyed on the top corr variable with all the drops inside
    for ind in df.index:
        waveless = remove_wave(ind)
        if waveless in stub_dict.keys():
            drop_list.append(ind)
            stub_dict[waveless].append(ind)
        else:
            stub_dict[waveless] = [ind]
    df.drop(drop_list,inplace=True)
    return df, corr_type    
    
    
def get_all_weights(mask, BES_Panel, specific_wave = None):
    #global BES_Panel

    if mask is None:
        if specific_wave is None:
            wts = BES_Panel[list(num_to_weight.values())]
        else:
            wts = BES_Panel[specific_wave]    
    else:   
        if specific_wave is None:
            wts = BES_Panel[list(num_to_weight.values())][mask]
        else:
            wts = BES_Panel[specific_wave][mask]

    wts = wts/wts.mean()

    wts = wts.mean(axis=1)
    wts =wts/wts.mean()
    return wts

    
def nice_bar_plot(ser1, ser2, output_folder, BES_Panel, normalize = 'columns', sort_labels=False,
                  text_width=8, text_fontsize=14, min_sample_size=100, title=None, drop_insig=True, fuckErrors=True,
                  mask=1, title_fontsize=14):
    var1 = ser1.name
    var2 = ser2.name
    
    mask = ser1.notnull() & ser2.notnull() & mask
    ct = pd.crosstab( ser1, ser2,
                      values= get_all_weights(mask, BES_Panel), aggfunc=sum, normalize=normalize)*100
    if sort_labels:
        sorted_labels = list(ser2.value_counts().index)
    else:
        sorted_labels = list(ser2.cat.categories)
           

    unweighted = pd.crosstab( ser1, ser2 )
    errors = 100 * np.sqrt(unweighted)/unweighted     

    labels_by_sample_size = {unweighted.sum().values[x]:sorted_labels[x]+" (N="+str(unweighted.sum().values[x])+")" for x in range(0,len(sorted_labels))}    
    labels_by_sample_size = {sorted_labels[x]+" (N="+str(unweighted.sum().values[x])+")":unweighted.sum().values[x] for x in range(0,len(sorted_labels))}    
    labels_restricted = [x for x in labels_by_sample_size.keys() if labels_by_sample_size[x] > min_sample_size] 
#     return labels_by_sample_size, labels_restricted
    if drop_insig:
        rubbish_entries = ct<errors
        ct[rubbish_entries]=np.nan
        errors[rubbish_entries]=np.nan
#     return(sorted_labels, errors, labels_by_sample_size)
    all_nan_rows = ~errors.isnull().any(axis=1)
    errors.columns = list( labels_by_sample_size.keys() )
   
    ct.columns = list( labels_by_sample_size.keys() )
    ct = ct.loc[all_nan_rows, labels_restricted]
#     return errors, labels_restricted
    errors = errors.loc[all_nan_rows, labels_restricted]
#     errors=errors.T
#     return errors
#     return errors, ct
    treatment = var2 +" by " + var1
    output_subfolder = create_subdir(output_folder, treatment)
    
    import textwrap 

    wrapper = textwrap.TextWrapper(width=text_width) 

    stacked = ct.stack().reset_index().rename(columns={0:'%',"level_1":var2})
    err_stacked = errors.stack().reset_index().rename(columns={0:'%',"level_1":var2})
    fig = plt.figure(figsize=(20, 8))
    ax = fig.subplots()

#     a = [np.ones(16),np.ones(16)]
#     a = errors.values
#     return a
#     iter(a)    
    
    stacked[var1] = stacked[var1].apply( lambda x: x +" (N="+str(unweighted.sum(axis=1).loc[x])+")" )
    stacked[var1].cat.set_categories(stacked[var1].cat.categories[all_nan_rows],inplace=True)
#     return stacked
#     return stacked['%'].shape,err_stacked["%"].values.reshape(len(stacked),1).shape
    if fuckErrors:
        sns.barplot(x = stacked[var2],
                    y = stacked['%'],
                    hue = stacked[var1],
                    ax = ax, order = labels_restricted);
    else:
        sns.barplot(x = stacked[var2],
                    y = stacked['%'],
                    hue = stacked[var1],
                    yerr = errors.values)        
#                    ax = ax, order = labels_restricted,
                    # err_stacked["%"].values );
# .reshape(len(stacked),1)
    if title is None:
        title = var2 +" by " + var1
    plt.title(title, fontsize=title_fontsize)
    sorted_labels = [sorted_labels[x]+" (N="+str(unweighted.sum().values[x])+")" for x in range(0,len(sorted_labels))]
    ax.set_xticklabels([ wrapper.fill(text=x) for x in labels_restricted], rotation=0, fontsize=text_fontsize);

    ax.annotate(dataset_citation, (0,0), (0, -140),
                     xycoords='axes fraction', textcoords='offset points', va='top', fontsize = 7) ;           
    fname = output_subfolder + clean_filename(title) + ".png"
    fig.savefig( fname, bbox_inches='tight' )    

    
def sort_by_wave(lst):
    dict_by_wave = {int(x.split("W")[-1]):x for x in lst}
    return [dict_by_wave[x] for x in sorted(dict_by_wave.keys())]



       
# transform a column of data until it's as approximately normally distributed as can be
# because most Machine Learning/Statistical methods assume data is ~normally distributed
# basically, what people normally do randomly logging/square-rooting data, only automatically

from scipy import stats
def box_cox_normalise(ser, offset = 3, bw='scott'):
    
    
    # box cox lr_scale
    fig = plt.figure()
    ax1 = fig.add_subplot(311)
    x = ser.values +ser.values.min()+offset
    prob = stats.probplot(x, dist=stats.norm, plot=ax1)
    ax1.set_xlabel('')
    ax1.set_title('Probplot against normal distribution')
    ax2 = fig.add_subplot(312)
    xt, _ = stats.boxcox(x)
    prob = stats.probplot(xt, dist=stats.norm, plot=ax2)
    ax2.set_title('Probplot after Box-Cox transformation')
    ax3 = fig.add_subplot(313)
    xt_std = (xt-xt.mean())/xt.std()
    sns.kdeplot(xt_std, ax=ax3, bw=bw, cut=0);
    sns.kdeplot(np.random.normal(size=len(xt_std)), ax=ax3, cut=0);
    plt.suptitle(ser.name)
    return xt_std
    

def corrank(X):
    import itertools
    df = pd.DataFrame([[(i,j),X.loc[i,j]] for i,j in list(itertools.combinations(X.corr(), 2))],columns=['pairs','corr'])    
    print(df.sort_values(by='corr',ascending=False).dropna())
    
    
# messy but time saver
    

import shap
import xgboost as xgb
# from sklearn.preprocessing import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.linear_model import ElasticNet
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import accuracy_score, log_loss,f1_score,roc_auc_score
import pickle

def shap_outputs(shap_values, train, target_var, output_subfolder,
                 dependence_plots = False, threshold = .1, min_features = 30,
                 title=None,save_shap_values=False,skip_bar_plot=False, multiclass=False,class_names=None,
                 plot_type='dot' ):


    #################################
#     threshold = .1
#     min_features = 30
    if not multiclass:

        global_shap_vals = np.abs(shap_values).mean(0)#[::-1]
        n_top_features = max( sum(global_shap_vals[np.argsort(global_shap_vals)]>=threshold),
                              min_features )
        n_top_features = min(n_top_features,global_shap_vals.shape[0])# can't display more features than present!
        
    #     if n_top_features <min_features:
    #         n_top_features = min_features

        ##########################

        inds = np.argsort(global_shap_vals)[-n_top_features:]

        y_pos = np.arange(n_top_features)
        
        if skip_bar_plot:
            plt.figure(figsize=(16,10))
            plt.title(target_var);
            plt.barh(y_pos, global_shap_vals[inds], color="#1E88E5")
            plt.yticks(y_pos, train.columns[inds])
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.xlabel("mean SHAP value magnitude (change in log odds)")
            plt.gcf().set_size_inches(6, 4.5)

            plt.savefig( output_subfolder + "mean_impact" + ".png", bbox_inches='tight' )

            plt.show()
    else: ## multiclass
        n_top_features = min(min_features,train.shape[1])# can't display more features than present!
        plot_type='bar'
       # class_names=["your","list","of", "class", "names"]

    ####################
    
    fig = plt.figure()
    if title is None:
        fig.suptitle(target_var);
    else:
        fig.suptitle(title);
        

       
        
    shap.summary_plot( shap_values, train, max_display=n_top_features, plot_type=plot_type, class_names=class_names );
    shap_problem = np.isnan(np.abs(shap_values).mean(0)).any()
    if shap_problem:
        summary_text = "summary_plot(approx)"
    else:
        summary_text = "summary_plot"
    
    fig.savefig( output_subfolder + summary_text + ".png", bbox_inches='tight' )
    
        ##################
    if dependence_plots:
        count = 0
        for name in train.columns[inds[::-1]]:
            fig = plt.figure(figsize = (16,10))
            fig.suptitle(target_var);
            shap.dependence_plot(name, shap_values, train)
            clean_filename(name)
            fig.savefig(output_subfolder + "featureNo "+str(count) + " " + clean_filename(name) + ".png", bbox_inches='tight')
            count = count + 1
    if save_shap_values:     
        pd.DataFrame(shap_values, columns = train.columns, index=train.index).to_pickle(output_subfolder+ "shap_values.zip", compression='zip')


objective = 'reg:squarederror'
eval_metric = 'rmse'

seed = 27
test_size = 0.33
minimum_sample = 100
early_stoppping_fraction = .1            
            
def get_non_overfit_settings( train, target, alg, seed, early_stoppping_fraction, test_size, eval_metric, verbose = True,
                              sample_weights = None,plot_validation=False):

    if target.dtype.name == 'category':
        stratification = target.cat.codes
    elif len(pd.unique(target))<=10:
        stratification = target
    else:
        stratification = pd.qcut( pd.Series( target ), q=10, duplicates = 'drop', ).cat.codes
    alg.set_params(**{"eval_metric":eval_metric[0],"early_stopping_rounds":alg.get_params()['n_estimators']*early_stoppping_fraction})
    if sample_weights is not None:

        X_train, X_test, y_train, y_test = train_test_split( pd.concat( [train,sample_weights], axis=1 ),
                                                             target, test_size=test_size,
                                                             random_state=seed, stratify= stratification)

        eval_set = [(X_train, y_train), (X_test, y_test)]
        weight_var = sample_weights.name
        sample_weight = X_train[weight_var].values
        sample_weight_eval_set = X_test[weight_var].values
        X_train.drop(weight_var, axis=1, inplace=True)
        X_test.drop(weight_var, axis=1, inplace=True)
        
        alg.fit(X_train, y_train, 
                eval_set=eval_set, verbose=verbose, sample_weight = sample_weight)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split( train,
                                                             target, test_size=test_size,
                                                             random_state=seed, stratify=stratification )
          

        eval_set = [(X_train, y_train), (X_test, y_test)]
        alg.fit(X_train, y_train, eval_set=eval_set, verbose=verbose )        
        

    

    def scale(x):
        return (np.array(x)-min(x))/(max(x)-min(x))
    if plot_validation:
        results = alg.evals_result()
        epochs = len(results['validation_0'][eval_metric[0]])
        x_axis = range(0, epochs)
        plt.figure(figsize=(14,8));
        color = iter(cm.rainbow(np.linspace(0, 1, len(eval_metric))))
        for metric in eval_metric:
            col = next(color)
            plt.plot(x_axis, np.log(scale(results['validation_0'][metric])), label=metric,marker='*',c=col)
            plt.plot(x_axis, np.log(scale(results['validation_1'][metric])), label=metric,marker='s',c=col)
            
        plt.axvline(alg.best_iteration);
        plt.legend();
        plt.title('XGBoost Validation Errors');
        plt.show();
    
    alg.set_params(n_estimators=alg.best_iteration)    
    
    # make predictions for test data
    # belt and bracers!
    predictions = alg.predict(X_test,iteration_range=(0, alg.best_iteration + 1))

    if target.dtype.name == 'category':
        ACC = accuracy_score(y_test, predictions)
        ##NLL = log_loss(y_test, predictions)
        F1 = f1_score(y_test, predictions,average='micro')
        ##ROCAUC = roc_auc_score(y_test, predictions)

        print("ACC: %.2f,  F1: %.2f"  % (ACC,  F1,) )   
           
        return (ACC, F1, alg.best_iteration)

    
    else:
    # evaluate predictions
        MSE = mean_squared_error(y_test, predictions)
        MAE = mean_absolute_error(y_test, predictions)
        EV = explained_variance_score(y_test, predictions)
        R2 = r2_score(y_test, predictions)

        print("MSE: %.2f, MAE: %.2f, EV: %.2f, R2: %.2f" % (MSE, MAE, EV, R2) )
  
        return (MSE, MAE, EV, R2, alg.best_iteration)

def shap_array(shap_values, train_columns, threshold = .1, min_features = 50):

    global_shap_vals = np.abs(shap_values).mean(0)#[::-1]
    n_top_features = max( sum(global_shap_vals[np.argsort(global_shap_vals)]>=threshold),
                          min_features )

    inds = np.argsort(global_shap_vals)[-n_top_features:]

    return pd.Series(global_shap_vals[inds][::-1],index = train_columns[inds][::-1])
    
# def shap_df(shap_values, train_columns, train_index, threshold = .1, min_features = 50):

    # global_shap_vals = np.abs(shap_values).mean(0)#[::-1]
    # n_top_features = max( sum(global_shap_vals[np.argsort(global_shap_vals)]>=threshold),
                          # min_features )

    # inds = np.argsort(global_shap_vals)[-n_top_features:]

    # return pd.Series(global_shap_vals[inds][::-1],index = train_columns[inds][::-1])
    
    

    
def get_generic_weights(BES_Panel):
    weight_vars = list(search(BES_Panel,"(wt_new_W\d+|wt_full_W\d)($|_result)").index)
    sample_weights = BES_Panel[weight_vars].mean(axis=1)
    sample_weights = sample_weights.fillna(sample_weights.median())
    sample_weights.name = "sample_weights"    
    return sample_weights
    

    

def get_xgboost_alg(
     learning_rate =0.05,
     n_estimators= 500,
     max_depth=6,
     min_child_weight=6,
     gamma=0.00065, # aka min_split_loss
     subsample=0.8,
     colsample_bytree=0.7,
     colsample_bylevel=.9,
     colsample_bynode=.85,
     objective= 'reg:squarederror',
   #  scale_pos_weight=1.09,
     reg_alpha=1.075,
     reg_lambda=1.011,
     sketch_eps=0.0,
     refresh_leaf=0,
     nthread=8,
     n_jobs =8,
     random_state=27**2, 
     base_score =.5,
     scale_pos_weight = 1,
     classification_problem = False,
     multi_strategy = None):
     
    if classification_problem==False:
        alg = XGBRegressor(
         learning_rate =learning_rate,
         n_estimators= n_estimators,
         max_depth = max_depth,
         min_child_weight = min_child_weight,
         gamma = gamma,
         subsample = subsample,
         colsample_bytree = colsample_bytree,
         colsample_bylevel = colsample_bylevel,
         colsample_bynode = colsample_bynode,
         objective = objective,
         scale_pos_weight = scale_pos_weight,
         reg_alpha=reg_alpha,
         reg_lambda=reg_lambda,
         sketch_eps=sketch_eps,
         refresh_leaf=refresh_leaf,
         nthread = nthread,
         n_jobs  = n_jobs ,
         random_state = random_state,
         base_score=base_score,
         multi_strategy = multi_strategy)
    else:
        alg = XGBClassifier(
         learning_rate =learning_rate,
         n_estimators= n_estimators,
         max_depth = max_depth,
         min_child_weight = min_child_weight,
         gamma = gamma,
         subsample = subsample,
         colsample_bytree = colsample_bytree,
         colsample_bylevel = colsample_bylevel,
         colsample_bynode = colsample_bynode,
         objective = objective,
         scale_pos_weight = scale_pos_weight,
         reg_alpha=reg_alpha,
         reg_lambda=reg_lambda,
         sketch_eps=sketch_eps,
         refresh_leaf=refresh_leaf,
         nthread = nthread,
         n_jobs  = n_jobs ,
         random_state = random_state,
         base_score=base_score,
         enable_categorical=True,
         multi_strategy = multi_strategy)        
    return alg
    
    
    
#global var_list
def xgboost_run(title, dataset, var_list,var_stub_list=[], subdir=None, min_features=30,
                dependence_plots=False , output_folder=".."+os.sep+"Output"+os.sep,Treatment="default",
                use_specific_weights = None, automatic_weights_from_wave_no = False, alg = None,verbosity=1,skip_bar_plot=False,
                eval_metric = ['rmse'],minimum_sample=50,dont_stop_on_sample_size_err=False,print_shap_chart=True,plot_validation=False,
                threshold=.1,multi_target=False,classification_problem=False,class_names=None):
    # will need to change this if you want to change alg type per variable in list of target vars!
                    
    if alg is None:
        if classification_problem:
            alg = get_xgboost_alg(classification_problem=classification_problem,
                objective= 'multi:softmax',)
            eval_metric = ['mlogloss']
        else:
            alg = get_xgboost_alg(classification_problem=classification_problem,
                objective= 'reg:squarederror',)
            
        
        alg.verbosity=verbosity
        
    if multi_target:
        alg.multi_strategy="multi_output_tree"
        
    if verbosity>=1:
        verbose=True
    else:
        verbose=False
        
    # global BES_Panel
    # for target_var,base_var in zip(var_list,base_list):
    treatment_subfolder = create_subdir(output_folder,Treatment)

    
    for target_var in var_list:
        if automatic_weights_from_wave_no:
            wave_no = get_wave_no( target_var )
            weight_var = num_to_weight[wave_no]    
            print( target_var, wave_no )

        target = create_target(dataset,target_var)
        mask   = target.notnull()
        if optional_mask & automatic_weights_from_wave_no:
            mask = mask&optional_mask_fn(wave_no)
        else:
            mask = mask&optional_mask_fn()
        target = target[mask]

        if sum(mask) < minimum_sample:
            print("Skipping - sample size beneath minimum: ",minimum_sample)
            skipping = True
            if not dont_stop_on_sample_size_err:
                raise Exception("fgflgf")
            continue
        skipping=False

        train = create_train(dataset,drop_other_waves,var_stub_list,mask,target_var)

        if subdir is None:
            output_subfolder = create_subdir(treatment_subfolder,target_var)
        else:
            output_subfolder = create_subdir(treatment_subfolder,subdir)

        if use_specific_weights is not None:
            sample_weights = use_specific_weights[mask].fillna(use_specific_weights[mask].median())




        elif automatic_weights_from_wave_no:
            sample_weights = weights[weight_var][mask]
            print("missing vals in sample weights: "+ str( sample_weights.isnull().sum() ) )
            sample_weights = sample_weights.fillna(sample_weights.median())
        
        else:
            sample_weights = None
    #         get_non_overfit_settings( train, target, alg, seed, early_stoppping_fraction, test_size, sample_weights )
    #         # fit to full dataset at non-overfitting level
    #         alg.fit(train, target, verbose = True, sample_weight = sample_weights)        
    #     else:

        if target.dtype.name == 'category':
            (ACC,  F1, alg_best_iteration) = get_non_overfit_settings( train, target, alg, seed, early_stoppping_fraction, test_size, eval_metric, verbose = verbose,
                                      sample_weights=sample_weights,plot_validation=plot_validation)
            pd.Series([ACC,  F1,  alg_best_iteration],index = ["ACC",  "F1", "alg_best_iteration"]).to_csv(output_subfolder+"scores.csv")
           
        
        else:

            (MSE, MAE, EV, R2, alg_best_iteration) = get_non_overfit_settings( train, target, alg, seed, early_stoppping_fraction, test_size, eval_metric, verbose = verbose,
                                      sample_weights=sample_weights,plot_validation=plot_validation)
            pd.Series([MSE, MAE, EV, R2, alg_best_iteration],index = ["MSE", "MAE", "EV", "R2", "alg_best_iteration"]).to_csv(output_subfolder+"scores.csv")
                                        
        alg.set_params(**{"eval_metric":eval_metric[0],"early_stopping_rounds":None})
        # fit to full dataset at non-overfitting level
        alg.fit(train, target, verbose = verbose, sample_weight = sample_weights)


    #################

        explainer = shap.TreeExplainer(alg)
        shap_values = explainer.shap_values(train)
        
#         shap_values = shap.TreeExplainer(alg).shap_values(train);

        shap_problem = np.isnan(np.abs(shap_values).mean(0)).any()
        if shap_problem:
            print("hit problem!")
            shap_values = shap.TreeExplainer(alg).shap_values(train, approximate=True);

        pickle.dump( explainer, open( output_subfolder+"explainer.pkl", "wb" ) )
        pickle.dump( explainer, open( output_subfolder+"alg.pkl", "wb" ) )

        if target.dtype.name == 'category':
            subtitle = "ACC: %.2f, F1: %.2f" % (ACC, F1)
            metrics = (ACC, F1)
        else:
            subtitle = "MSE: %.2f, MAE: %.2f, EV: %.2f, R2: %.2f" % (MSE, MAE, EV, R2)
            metrics = (MSE, MAE, EV, R2)
            
        
            
        if target.dtype.name == 'category':
            multiclass = True
            if class_names is None:
                class_names = list(target.cat.categories)
        else:
            multiclass = False
            class_names = None
            
        if print_shap_chart:
            shap_outputs(shap_values, train, target_var, output_subfolder, threshold = threshold,
                     min_features = min_features, title=title+"\n"+subtitle,
                     dependence_plots=dependence_plots,skip_bar_plot=False,multiclass=multiclass,class_names=class_names)        

    if skipping:
        return (None,None,None,None,None,None,None)
    else:
        return (explainer, shap_values, train.columns, train.index, alg,output_subfolder,metrics)






############################ BASIC SETTINGS

from sklearn import datasets
from sklearn.decomposition import PCA, IncrementalPCA, NMF, TruncatedSVD, FastICA, FactorAnalysis, SparsePCA
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    
optional_mask = False
sample_wts = False
drop_other_waves = False


# Leavers only
def optional_mask_fn(wave=[]):
    return 1



def create_train(dataset,drop_other_waves,var_stub_list,mask,target_var):
    keep_list = dataset.columns
    
    if drop_other_waves:
        # drop variables from other waves
        other_waves = get_other_wave_pattern(wave_no, max_wave, num_to_wave)
        keep_list = [x for x in keep_list if not re.search( other_waves, x )]
    
    keep_list = list(keep_list)
    # multitarget case
    if isinstance(target_var,list):
        keep_list = [x for x in keep_list if x not in target_var]
    else:
        keep_list.remove(target_var)
    # drop key variables
    keep_list = [x for x in keep_list if not any([var_stub in x for var_stub in var_stub_list])] 
    
    
    
    return dataset[keep_list][mask]


def create_target(dataset,target_var):
    
    return dataset[target_var]



########################################################################################################################

def nice_crosstab(ser_ind,ser_col,data=None,values=None,aggfunc=sum,normalize='all',round_places=0,percentage=True,
                 margins=True, add_sample_sizes = True,drop_col_list=[],drop_ind_list=[],replace_dict={}):
    if data is not None:
        ser_ind = BES_Panel[ser_ind]
        ser_col = BES_Panel[ser_col]
        if values is not None:
            values = BES_Panel[values]
    if drop_col_list:
        if ser_col.dtype == 'category':
            ser_col = ser_col.cat.remove_categories(drop_col_list)
        else:
            ser_col = ser_col.replace({k:np.nan for k in drop_col_list})
    # replace_dict
    if ser_col.dtype == 'category':
        ser_col = ser_col.cat.rename_categories(replace_dict)
    else:
        ser_col = ser_col.replace(replace_dict)
#     ind_N_dict = ser_ind.value_counts().to_dict()
#         ser_ind.columns = [x+"(N="+str(ind_N_dict[x])+")" for x in  ind_N_dict.keys()]        
    if ser_ind.dtype == 'category':
        ser_ind = ser_ind.cat.rename_categories(replace_dict)
    else:
        ser_ind = ser_ind.replace(replace_dict)            
            
    
    if drop_ind_list:
        if ser_ind.dtype == 'category':
            ser_ind = ser_ind.cat.remove_categories(drop_ind_list)
        else:
            ser_ind = ser_ind.replace({k:np.nan for k in drop_ind_list})
            
    if add_sample_sizes:
        col_N_dict = ser_col.value_counts().to_dict()
#         ser_col.columns = [x+"\n(N="+str(col_N_dict[x])+")" for x in  col_N_dict.keys()]
        if ser_col.dtype == 'category':
            ser_col = ser_col.cat.rename_categories({x:x+" (N="+str(col_N_dict[x])+")" for x in  col_N_dict.keys()})
        else:
            ser_col = ser_col.replace({x:x+" (N="+str(col_N_dict[x])+")" for x in  col_N_dict.keys()})
        ind_N_dict = ser_ind.value_counts().to_dict()
#         ser_ind.columns = [x+"(N="+str(ind_N_dict[x])+")" for x in  ind_N_dict.keys()]        
        if ser_ind.dtype == 'category':
            ser_ind = ser_ind.cat.rename_categories({x:x+" (N="+str(ind_N_dict[x])+")" for x in  ind_N_dict.keys()})
        else:
            ser_ind = ser_ind.replace({x:x+" (N="+str(ind_N_dict[x])+")" for x in  ind_N_dict.keys()})
#         ser_ind = ser_ind.replace({x:x+" (N="+str(ind_N_dict[x])+")" for x in  ind_N_dict.keys()})
    
    if percentage:
        ct = round(pd.crosstab(ser_ind,ser_col,
                values=values,aggfunc=aggfunc,normalize=normalize,margins=margins)*100,round_places)
    else:
        ct = round(pd.crosstab(ser_ind,ser_col,
                values=values,aggfunc=aggfunc,normalize=normalize,margins=margins),round_places)
    if round_places==0:
        ct = ct.astype(int)
    return ct

# drop_col_list = ["Don't know"]
# drop_ind_list = ["Don't know"]
# replace_dict = {'Conservative':"CON", 'Labour':"LAB", 'Liberal Democrat':"LD",
       # 'Scottish National Party (SNP)':"SNP", 'Plaid Cymru':"PC",
       # 'United Kingdom Independence Party (UKIP)':"UKIP", 'Green Party':"GP",
       # 'British National Party (BNP)':"BNP", 'Other':"OTH"}


# nice_crosstab("preferMPEthnicW6","p_past_vote_2015",data=BES_Panel,values="wt_new_W6",normalize='columns',
              # round_places=0,add_sample_sizes=True,drop_col_list=drop_col_list,drop_ind_list=drop_ind_list,
              # replace_dict=replace_dict)