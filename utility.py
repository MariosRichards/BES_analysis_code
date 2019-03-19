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


def intersection(lst1, lst2): 
  
    # Use of hybrid method 
    temp = set(lst2) 
    lst3 = [value for value in lst1 if value in temp] 
    return lst3 

def amalgamate_waves(df, pattern, forward_fill=True, specify_wave_order = None):
    # euref_imm = amalgamate_waves(BES_reduced_with_na,"euRefVoteW",forward_fill=False)
    # assumes simple wave structure, give a pattern that works!
    df_cols_dict = {int(re.search("W(\d+)", x).groups()[0]):x for x in df.columns if re.match(pattern, x)}
    # sort columns
    if specify_wave_order is not None:
        df_cols = [df_cols_dict[x] for x in specify_wave_order]
    else:
        df_cols = [df_cols_dict[x] for x in sorted(df_cols_dict.keys())]
    
    # forward fill and and pick last column - or backward fill and pick first column
    if len(df_cols)<=1:
        raise Exception("Can't amalgamate less than two variables!")
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


def clean_filename(filename, whitelist=valid_filename_chars, replace=' ', char_limit = 30):
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

def match(df, pattern, case_sensitive=True, mask=None):
    if mask is None:
           mask = pd.Series(np.ones( (df.shape[0]) )).astype('bool')
    if case_sensitive:
        return df[[x for x in df.columns if re.match(pattern,x)]][mask].notnull().sum()
    else:
        return df[[x for x in df.columns if re.match(pattern, x, re.IGNORECASE)]][mask].notnull().sum()

def search(df, pattern, case_sensitive=False, mask=None):
    if mask is None:
           mask = pd.Series(np.ones( (df.shape[0]) )).astype('bool')
    if case_sensitive:
        return df[[x for x in df.columns if re.search(pattern,x)]][mask].notnull().sum()
    else:
        return df[[x for x in df.columns if re.search(pattern, x, re.IGNORECASE)]][mask].notnull().sum()

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
    max_wave = int(re.match("W(\d+)_",dataset_name).groups()[0])
    num_to_wave = {x:"W"+str(x) for x in range(1,max_wave+1)}
    num_to_weight = { y:[x for x in BES_Panel.columns.sort_values(ascending=False) if re.match("wt_(new|full)_W"+str(y)+"$",x)][0] for y in range(1,max_wave+1) }
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
    df1 = input_df.copy()
    focal_var = df1[name]
    focal_mask = focal_var.notnull()


    pattern_list = [x for x in df1.columns if re.search(pattern,x)]

    variances = df1[focal_mask].var()
    low_var_list = list(variances[variances<min_variance].index)
    sample_sizes = df1[focal_mask].notnull().sum()
    low_sample_size_list = list(sample_sizes[sample_sizes<min_sample_size].index)

    drop_list = pattern_list+low_var_list+low_sample_size_list
    df1.drop(drop_list,axis=1,inplace=True)

    if corr_type == "pearson":
        df = df1.apply(lambda x: corr_simple_pearsonr(x,focal_var)).apply(pd.Series)
    elif corr_type == "spearman":
        df = df1.apply(lambda x: corr_simple_spearmanr(x,focal_var)).apply(pd.Series)

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
                    ax = ax, order = labels_restricted,
                    yerr = errors.values);        
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
        