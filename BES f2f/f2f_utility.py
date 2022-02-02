
def full_form_birth_year(x):
    if pd.isnull(x):
        return np.nan
    elif x>60:
        return 1800 + x
    else:
        return 1900 + x
    
def rep_char(x):
    return x.replace('[', '{').replace(']', '}').replace('<', '{').replace('>', '}')

list_of_scale_harm_vars = ["Age","kind_of_scool","year_past_min_sch_leave_age"]

# def replace_var_names(labels):
    # varname_replace_dict = {}
    # replace_char_dict = {"[":"{","]":"}"}
    # for col in df_simp.columns:
        # if np.any([x in col for x in generic_cols]) or (col in list_of_scale_harm_vars):
            # # 'age' is a problem as a common word component
            # varname_replace_dict[col] = rep_char(col)
            
        # elif "|" not in col:
            # varname_replace_dict[col] = rep_char(labels[col])
        # elif len(col.split("|"))==2:
            # varname_replace_dict[col] = rep_char(labels[col.split("|")[0]])\
                # +"("+ rep_char(col.split("|")[1]) +")"
        # if varname_replace_dict[col]=="":
            # varname_replace_dict[col] = rep_char(col)
    # return varname_replace_dict

def rep_char(x):
    return x.replace('[', '{').replace(']', '}').replace('<', '{').replace('>', '}')

def replace_var_names(labels,df_simp):
    varname_replace_dict = {}
    replace_char_dict = {"[":"{","]":"}"}
    for col in df_simp.columns:
        if np.any([x in col for x in generic_cols]) or (col in list_of_scale_harm_vars):
            # 'age' is a problem as a common word component
            varname_replace_dict[col] = rep_char(col)
            
        elif "|" not in col:
            varname_replace_dict[col] = rep_char(labels[col])
        elif len(col.split("|"))==2:
            varname_replace_dict[col] = rep_char(labels[col.split("|")[0]])\
                +"("+ rep_char(col.split("|")[1]) +")"
        if varname_replace_dict[col]=="":
            varname_replace_dict[col] = rep_char(col)
    return varname_replace_dict

global BES_label_list, BES_df_list

def prep_df_only(ge,target_var,target_var_replace_dict,target_var_drop_list,target_var_title_pair,
       var_stub,harm_vars,min_features,dependence_plots=False,
       drop_vars = [],demo_var_only=True,
        multi_class_target=False,dummy_na=True,specific_vars = False, drop_after_dummying = []):

    if demo_var_only:
        demo_vars = demo_var_dict[ge]
    elif specific_vars:
        demo_vars = specific_vars
    else:
        demo_vars = list(BES_label_list[ge].keys())

#     labels = BES_label_list[ge]
    df = pd.concat([f2f_harmonised[f2f_harmonised["dataset"]==ge][harm_vars],BES_df_list[ge]
                           ],axis=1)
    old_demo_vars = demo_vars.copy()
    demo_vars = demo_vars+harm_vars
    demo_vars = list(set(demo_vars).intersection(df.columns))
    if drop_vars:
        demo_vars = [x for x in demo_vars if x not in drop_vars]
        old_demo_vars = [x for x in old_demo_vars if x not in drop_vars]    

    df_simp = df[demo_vars].copy()

    df_simp[target_var] = df_simp[target_var].replace(target_var_replace_dict)
    # prep nominal and ordinal as categorical to be dummied
    old_demo_vars = [x for x in old_demo_vars if var_type_dict_nonans[ge][x] in ["nominal","ordinal"]]
    scale_vars = [x for x in old_demo_vars if var_type_dict_nonans[ge][x] in ["scale"]]
        
    if not multi_class_target:
        df_simp[target_var] = df_simp[target_var].replace(target_var_replace_dict)    
        df_simp[old_demo_vars] = df_simp[old_demo_vars].astype('category')    
        df_simp[scale_vars] = df_simp[scale_vars].astype('float')
        df_simp = pd.get_dummies(df_simp,prefix_sep='|',dummy_na=dummy_na).drop(target_var_drop_list,axis=1)
        if target_var+"|nan" in df_simp.columns:
            df_simp.loc[df_simp[target_var+"|nan"]==1,var_stub]=np.nan
            df_simp.drop(target_var+"|nan",axis=1,inplace=True,)
    else:
        old_demo_vars = [x for x in old_demo_vars if x !=target_var]
        scale_vars = [x for x in scale_vars if x !=target_var]   
        df_simp[old_demo_vars] = df_simp[old_demo_vars].astype('category')    
        df_simp[scale_vars] = df_simp[scale_vars].astype('float')        
        df_simp[target_var] = df_simp[target_var].astype('category')
        target_var_drop_list = [x for x in target_var_drop_list if x in df_simp[target_var].cat.categories]
        df_simp[target_var] = df_simp[target_var].cat.remove_categories(target_var_drop_list)        
        all_but_target = [x for x in df_simp.columns if x !=target_var]
        target_temp = df_simp[target_var].copy()
        df_simp = pd.get_dummies(df_simp[all_but_target],prefix_sep='|',dummy_na=dummy_na)
        df_simp[target_temp.name] = target_temp
        

    df_simp = df_simp.rename(columns = replace_var_names( BES_label_list[ge] , df_simp ))  
    
    df_simp = df_simp.drop(drop_after_dummying , axis=1)

    Treatment = var_stub+"_"+ge

    var_list = [var_stub]
    var_stub_list = [var_stub,]
    if not multi_class_target:
        df_simp = df_simp.select_dtypes('number')
        df_simp = df_simp.astype('float')
#     mask = df_simp[var_stub].notnull() & df_simp["wt"].notnull()
    return df_simp

def prep_df(ge,target_var,target_var_replace_dict,target_var_drop_list,target_var_title_pair,
       var_stub,harm_vars,min_features,dependence_plots=False,drop_vars = [],demo_var_only=True,alg=None,
           multi_class_target=False,dummy_na=True,specific_vars = False, drop_after_dummying = [],
           wt_col = "wt"):

    # drop drop_var variables from demo_var list
    if demo_var_only:
        demo_vars = demo_var_dict[ge]
    elif specific_vars:
        demo_vars = specific_vars
    else:
        demo_vars = list(BES_label_list[ge].keys())
        

#     labels = BES_label_list[ge]
    df = pd.concat([f2f_harmonised[f2f_harmonised["dataset"]==ge][harm_vars],BES_df_list[ge]                            
                           ],axis=1)

    
    old_demo_vars = demo_vars.copy()
    demo_vars = demo_vars+harm_vars
    demo_vars = list(set(demo_vars).intersection(df.columns))
    # drop vars after adding harm/dropping ones not present
    if drop_vars:
        demo_vars = [x for x in demo_vars if x not in drop_vars]    
        old_demo_vars = [x for x in old_demo_vars if x not in drop_vars]    

    df_simp = df[demo_vars].copy()



    # prep nominal and ordinal as categorical to be dummied
    old_demo_vars = [x for x in old_demo_vars if var_type_dict_nonans[ge][x] in ["nominal","ordinal"]]
    scale_vars = [x for x in old_demo_vars if var_type_dict_nonans[ge][x] in ["scale"]]
    
    if not multi_class_target:
        df_simp[target_var] = df_simp[target_var].replace(target_var_replace_dict)    
        df_simp[old_demo_vars] = df_simp[old_demo_vars].astype('category')    
        df_simp[scale_vars] = df_simp[scale_vars].astype('float')
        df_simp = pd.get_dummies(df_simp,prefix_sep='|',dummy_na=dummy_na).drop(target_var_drop_list,axis=1)
        if target_var+"|nan" in df_simp.columns:
            df_simp.loc[df_simp[target_var+"|nan"]==1,var_stub]=np.nan
            df_simp.drop(target_var+"|nan",axis=1,inplace=True,)
        eval_metric='rmse'
    else:
        old_demo_vars = [x for x in old_demo_vars if x !=target_var]
        scale_vars = [x for x in scale_vars if x !=target_var]
        df_simp[old_demo_vars] = df_simp[old_demo_vars].astype('category')    
        df_simp[scale_vars] = df_simp[scale_vars].astype('float')
        df_simp[target_var] = df_simp[target_var].astype('category')
        target_var_drop_list = [x for x in target_var_drop_list if x in df_simp[target_var].cat.categories]
        df_simp[target_var] = df_simp[target_var].cat.remove_categories(target_var_drop_list)
        all_but_target = [x for x in df_simp.columns if x !=target_var]
        target_temp = df_simp[target_var].copy()
        df_simp = pd.get_dummies(df_simp[all_but_target],prefix_sep='|',dummy_na=dummy_na)
        df_simp[target_temp.name] = target_temp
        eval_metric='mlogloss'



    df_simp = df_simp.rename(columns = replace_var_names( BES_label_list[ge] , df_simp ))  
    
    df_simp = df_simp.drop(drop_after_dummying , axis=1)
    
    Treatment = var_stub+"_"+ge

    var_list = [var_stub]
    var_stub_list = [var_stub,]
    
    if not multi_class_target:
        df_simp = df_simp.select_dtypes('number')
        df_simp = df_simp.astype('float')
        

    colname = var_stub
    if target_var_title_pair is not None:
        title = "\n\nMore Likely to "+target_var_title_pair[0]+" <---   ---> More Likely to"+target_var_title_pair[1]
    else:
        title = ""
        
    if wt_col =="wt":
        wt_cols = ["wt"]
        wt_ser = df_simp["wt"]
    else:
        wt_cols = ["wt",wt_col]
        wt_ser = df_simp[wt_col]        
    mask = df_simp[var_stub].notnull() & df_simp[wt_col].notnull()
    
    (explainer, shap_values, train_columns, train_index, alg,output_subfolder)=\
        xgboost_run(subdir=colname,dataset=df_simp[mask].drop(wt_cols,axis=1),
                var_list=var_list,var_stub_list=var_stub_list,
                use_specific_weights=wt_ser[mask],
                min_features = min(df_simp.shape[1]-1,min_features),verbosity=0,
                skip_bar_plot=True,dependence_plots=dependence_plots,alg=alg,eval_metric=eval_metric,                    
                title = title)
    
    return (explainer, shap_values, train_columns, train_index, alg,output_subfolder)
    
    
    
##### autoencoder

import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
import time
# from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.utils.np_utils import to_categorical
# from tensorflow.python.keras.utils import to_categorical

import shap
from tensorflow.python.ops import gradients_impl as tf_gradients_impl
tf_gradients_impl._IsBackpropagatable=True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import layers, losses
from tensorflow.python.keras.datasets import fashion_mnist
from tensorflow.python.keras.models import Model

from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder
from tensorflow.python.keras.layers import InputLayer
from sklearn.preprocessing import OneHotEncoder

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Input, Dense, Layer, InputSpec
from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.python.keras import regularizers, activations, initializers, constraints, Sequential
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.constraints import UnitNorm, Constraint

class DenseTied(Layer):
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 tied_to=None,
                 **kwargs):
        self.tied_to = tied_to
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super().__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
                
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        if self.tied_to is not None:
            self.kernel = K.transpose(self.tied_to.kernel)
            self._non_trainable_weights.append(self.kernel)
        else:
            self.kernel = self.add_weight(shape=(input_dim, self.units),
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

class WeightsOrthogonalityConstraint (Constraint):
    def __init__(self, encoding_dim, weightage = 1.0, axis = 0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
        self.axis = axis
        
    def weights_orthogonality(self, w):
        if(self.axis==1):
            w = K.transpose(w)
        if(self.encoding_dim > 1):
            m = K.dot(K.transpose(w), w) - K.eye(self.encoding_dim)
            return self.weightage * K.sqrt(K.sum(K.square(m)))
        else:
            m = K.sum(w ** 2) - 1.
            return m

    def __call__(self, w):
        return self.weights_orthogonality(w)
    
class UncorrelatedFeaturesConstraint(Constraint):
    
    def __init__(self, encoding_dim, weightage = 1.0):
        self.encoding_dim = encoding_dim
        self.weightage = weightage
    
    def get_covariance(self, x):
        x_centered_list = []

        for i in range(self.encoding_dim):
            x_centered_list.append(x[:, i] - K.mean(x[:, i]))
        
        x_centered = tf.stack(x_centered_list)
        covariance = K.dot(x_centered, K.transpose(x_centered)) / tf.cast(x_centered.get_shape()[0], tf.float32)
        
        return covariance
            
    # Constraint penalty
    def uncorrelated_feature(self, x):
        if(self.encoding_dim <= 1):
            return 0.0
        else:
#             output = K.sum(K.square(
#                 self.covariance - tf.matmul(self.covariance, K.eye(self.encoding_dim))))
            output = tf.reduce_sum(tf.square(
                self.covariance - tf.matmul(self.covariance, tf.eye(self.encoding_dim))))            
            
            # FIXED ???
         # still don't know what the problem is here!
# https://stackoverflow.com/questions/57836849/tensorflow-keras-custom-constraint-not-working        
# https://stackoverflow.com/questions/53751024/tying-autoencoder-weights-in-a-dense-keras-layer        
        
            # tf.math.multiply
            return output

    def __call__(self, x):
        self.covariance = self.get_covariance(x)
        return self.weightage * self.uncorrelated_feature(x)   
    
def extract_layers(main_model, starting_layer_ix, ending_layer_ix, input_shape):
  # create an empty model
    new_model = Sequential()
    first_layer =True
    for ix in range(starting_layer_ix, ending_layer_ix + 1):
        curr_layer = main_model.get_layer(index=ix)
        # copy this layer over to the new model
        if first_layer:
            new_model.add(input_shape)
            first_layer=False
        new_model.add(curr_layer)
    return new_model    
    
import keras.backend as K
def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)

    return 1 - numerator / denominator

def balanced_cross_entropy(y_true, y_pred):
    
    beta = beta = tf.reduce_mean(1 - y_true)
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)

    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)
    
# all layers
def get_kernel_regularizer(orthogonality_constraint,dimension,axis,weight=10e-7,weightage=1.0):
    if orthogonality_constraint:
        return WeightsOrthogonalityConstraint(dimension, weightage=weightage, axis=axis)
    else:
        return regularizers.l2(weight)

# just encoding layer (all of them or just final? Just final surely?)
def get_activity_regularizer(uncorrelated_features,encoding_dim, weightage = 1.):
    if uncorrelated_features:
        return UncorrelatedFeaturesConstraint(encoding_dim, weightage = 1.)
    else:
        return None

# all layers: kernel_constraint=get_kernel_constraint(unit_norm, axis)
def get_kernel_constraint(unit_norm, axis):
    if unit_norm:
        return UnitNorm(axis=axis)
    else:
        return None    

def run_autoencoder(encoding_dim=32,hidden_size=100,verbose=2,
                    loss='mean_squared_error',metrics=['accuracy','mse'],optimizer='adam',
                    penultimate_act = 'relu',ultimate_act = 'linear',max_epochs=100,
                    no_hidden_layers=1,
                    tied_layers = False, orthogonality_constraint = False,uncorrelated_features=False,unit_norm=False,
                    regularizer_weight=10e-7):

    # get rough intermediate sizes for hidden layers
    x = pow(X.shape[1]/encoding_dim,1/ (no_hidden_layers+1) )
    if no_hidden_layers!=1:
        hidden_layer_size = [int(encoding_dim*pow(x,y)) for y in range(1,no_hidden_layers+1)]
    else:
        hidden_layer_size = [hidden_size]
    
     
    
    # Single fully-connected neural layer as encoder and decoder
    use_regularizer = True
#     my_regularizer = None
#     my_epochs = 50
    features_path = 'simple_autoe_features.pickle'
    labels_path = 'simple_autoe_labels.pickle'

    early_stopping_monitor = EarlyStopping(patience=3)

    if use_regularizer:
        # add a sparsity constraint on the encoded representations
        # note use of 10e-5 leads to blurred results
    #     my_regularizer = regularizers.l1(10e-8)
        # and a larger number of epochs as the added regularization the model
        # is less likely to overfit and can be trained longer
#         my_epochs = 100
        features_path = 'sparse_autoe_features.pickle'
        labels_path = 'sparse_autoe_labels.pickle'

    # this is the size of our encoded representations
    # encoding_dim = encoding_dim   # 32 floats -> compression factor 24.5, assuming the input is 784 floats
    # hidden_size = hidden_size

# encoder = Dense(encoding_dim, activation="linear", input_shape=(input_dim,), use_bias=True,
#                 kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=0)) 
# decoder = Dense(input_dim, activation="linear", use_bias = True,
#                 kernel_regularizer=WeightsOrthogonalityConstraint(encoding_dim, weightage=1., axis=1))    


    

    # this is our input placeholder; 784 = 28 x 28
    # input_img = Input(shape=(784, ))

    input_img = Input(shape=(X.shape[1], ))

    encoder_list = []
    
    # "encoded" is the encoded representation of the inputs
    hidden_encoder = input_img
    for hidden_layer_number in range(no_hidden_layers-1,-1,-1):       
        
        d = Dense(hidden_layer_size[hidden_layer_number], activation = penultimate_act,
                          kernel_regularizer=get_kernel_regularizer(orthogonality_constraint,
                                                                    hidden_layer_size[hidden_layer_number],
                                                                    axis=0, weight=regularizer_weight),
                          kernel_constraint=get_kernel_constraint(unit_norm, axis=0)
                 )
        
        hidden_encoder = d(hidden_encoder)
        encoder_list.append(d)
    
    d = Dense(encoding_dim, activation=penultimate_act,
                          kernel_regularizer=get_kernel_regularizer(orthogonality_constraint,
                                                encoding_dim,
                                                axis=0, weight=regularizer_weight),
                          activity_regularizer= get_activity_regularizer(uncorrelated_features,encoding_dim, weightage = 1.),
                          kernel_constraint=get_kernel_constraint(unit_norm, axis=0)
             )
    
    encoded = d(hidden_encoder)
    encoder_list.append(d)
    # encoded = Dense(encoding_dim, activation='relu', activity_regularizer=my_regularizer)(input_img)


    # "decoded" is the lossy reconstruction of the input
    # decoded = Dense(X.shape[1], activation='sigmoid')(encoded)
    
    
#     decoder = DenseTied(input_dim, activation="linear", tied_to=encoder, use_bias = True)   
    
    hidden_decoder = encoded
    for hidden_layer_number in range(0,no_hidden_layers):
        if tied_layers:
            hidden_decoder = DenseTied(hidden_layer_size[hidden_layer_number], activation = penultimate_act,
                                      tied_to=encoder_list[no_hidden_layers-hidden_layer_number],
                                      kernel_regularizer=get_kernel_regularizer(orthogonality_constraint,
                                                                    ([encoding_dim]+hidden_layer_size)[hidden_layer_number],
                                                                    axis=1, weight=regularizer_weight),  
                                      kernel_constraint=get_kernel_constraint(unit_norm, axis=1),
                                      )(hidden_decoder)
            
        else:
            hidden_decoder = Dense(hidden_layer_size[hidden_layer_number], activation = penultimate_act,
                                      kernel_regularizer=get_kernel_regularizer(orthogonality_constraint,
                                                ([encoding_dim]+hidden_layer_size)[hidden_layer_number],
                                                 axis=1, weight=regularizer_weight),   
                                      kernel_constraint=get_kernel_constraint(unit_norm, axis=1)
                                      )(hidden_decoder)
            
    if tied_layers:
        decoded = DenseTied(X.shape[1], activation=ultimate_act,
                          tied_to=encoder_list[0],
                          kernel_regularizer=get_kernel_regularizer(orthogonality_constraint,
                                    ([encoding_dim]+hidden_layer_size)[no_hidden_layers],
                                     axis=1, weight=regularizer_weight),   
                                     kernel_constraint=get_kernel_constraint(unit_norm, axis=1)
                           )(hidden_decoder)

    else:            
        decoded = Dense(X.shape[1], activation=ultimate_act,
                          kernel_regularizer=get_kernel_regularizer(orthogonality_constraint,
                                    ([encoding_dim]+hidden_layer_size)[no_hidden_layers],
                                     axis=1, weight=regularizer_weight),  
                          kernel_constraint=get_kernel_constraint(unit_norm, axis=1)
                       )(hidden_decoder)
    # decoded = Dense(X.shape[1], activation='linear')(encoded)
    # decoded = Dense(X.shape[1], activation='softmax')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # Separate Encoder model

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # Separate Decoder model

    # create a placeholder for an encoded (32-dimensional) input
#     encoded_input = Input(shape=(encoding_dim,))
    # hidden_layer_input = Input(shape=(hidden_size,))
    # retrieve the last layer of the autoencoder model
    ## changed to -2 after adding hidden layer!
    # decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    # decoder = Model(encoded_input, decoder_layer(hidden_layer_input))
    encoded_input_layer = InputLayer(input_shape=(encoding_dim,))
    # decoder = Model(encoded_input, decoder_layer(encoded_input))

    decoder = extract_layers(autoencoder, -1-no_hidden_layers, -1, encoded_input_layer)

    # Train to reconstruct MNIST digits

    # configure model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    # autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    autoencoder.compile( optimizer = optimizer,
    #                loss = 'categorical_crossentropy',
    #                loss = 'categorical_crossentropy',
                   loss = loss,     
                   metrics=metrics)


    # # prepare input data
    # (x_train, _), (x_test, y_test) = mnist.load_data()

    # # normalize all values between 0 and 1 and flatten the 28x28 images into vectors of size 784
    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # print(x_train.shape)
    # print(x_test.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Train autoencoder for 50 epochs

    if isinstance(X_test, pd.DataFrame):
        X_test_values = X_test.values
        X_train_values = X_train.values
    else:
        X_test_values = X_test
        X_train_values = X_train
    
    autoencoder.fit(X_train_values, X_train_values, epochs=max_epochs, batch_size=512,
                    shuffle=True, validation_data=(X_test_values, X_test_values),
                    verbose=verbose, callbacks=[early_stopping_monitor])

    # after 50/100 epochs the autoencoder seems to reach a stable train/test lost value

    # Visualize the reconstructed encoded representations

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_imgs = encoder.predict(X_test_values)
    decoded_imgs = decoder.predict(encoded_imgs)

    # save latent space features 32-d vector
    pickle.dump(encoded_imgs, open(features_path, 'wb'))
    pickle.dump(y_test, open(labels_path, 'wb'))

    # n = 10  # how many digits we will display
    # plt.figure(figsize=(10, 2), dpi=100)
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.set_axis_off()

    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + n + 1)
    #     plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     plt.gray()
    #     ax.set_axis_off()kernel_constrain

    # plt.show()
    
    return decoded_imgs,X_test,encoded_imgs,autoencoder,encoder,decoder
    
def get_top_corr(x,n):
    return x[x.abs().sort_values(ascending=False).head(n).index]