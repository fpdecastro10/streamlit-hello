import optuna as opt
import pandas as pd
from functools import partial
import pickle
from prophet import Prophet
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestRegressor

from plotnine import *
from datetime import datetime, timedelta
import os

import seaborn as sns

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18

plt.style.use({
    'axes.facecolor': '#0e1118',
    'axes.edgecolor': 'white',
    'axes.labelcolor': 'white',
    'text.color': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#0e1118',
    'figure.facecolor': '#0e1118',
    'figure.edgecolor': '#0e1118',
    'savefig.facecolor': '#0e1118',
    'savefig.edgecolor': '#1a1a1a',
})

import shap
shap.initjs()

import numpy as np

class AdstockGeometric(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        
    def fit(self, X, y=None):
        X = check_array(X)
        self._check_n_features(X, reset=True)
        return self
    
    def transform(self, X: np.ndarray):
        check_is_fitted(self)
        X = check_array(X)
        self._check_n_features(X, reset=False)
        x_decayed = np.zeros_like(X)
        x_decayed[0] = X[0]
        
        for xi in range(1, len(x_decayed)):
            x_decayed[xi] = X[xi] + self.alpha* x_decayed[xi - 1]
        return x_decayed
    
def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2)) / (np.max(y_true) - np.min(y_true))

#https://github.com/facebookexperimental/Robyn
def rssd(effect_share, spend_share):
    """RSSD decomposition
    
    Decomposition distance (root-sum-square distance, a major innovation of Robyn) 
    eliminates the majority of "bad models" 
    (larger prediction error and/or unrealistic media effect like the smallest channel getting the most effect

    Args:
        effect_share ([type]): percentage of effect share
        spend_share ([type]): percentage of spend share

    Returns:
        [type]: [description]
    """
    return np.sqrt(np.sum((effect_share - spend_share) ** 2))


def plot_spend_vs_effect_share(decomp_spend: pd.DataFrame, figure_size = (15, 10)):
    """Spend vs Effect Share plot

    Args:
        decomp_spend (pd.DataFrame): Data with media decompositions. The following columns should be present: media, spend_share, effect_share per media variable
        figure_size (tuple, optional): Figure size. Defaults to (15, 10).

    Example:
        decomp_spend:
        media         spend_share effect_share
        tv_S           0.31        0.44
        ooh_S          0.23        0.34
    
    Returns:
        [plotnine]: plotnine plot
    """
    
    plot_spend_effect_share = decomp_spend.melt(id_vars = ["media"], value_vars = ["spend_share", "effect_share"])

    plt = ggplot(plot_spend_effect_share, aes("media", "value", fill = "variable")) \
    + geom_bar(stat = "identity", position = "dodge") \
    + geom_text(aes(label = "value * 100", group = "variable"), color = "darkblue", position=position_dodge(width = 0.5), format_string = "{:.2f}%") \
    + coord_flip() \
    + ggtitle("Share of Spend VS Share of Effect") + ylab("") + xlab("") \
    + theme(figure_size = figure_size, 
                    legend_direction='vertical', 
                    legend_title=element_blank(),
                    legend_key_size=20, 
                    legend_entry_spacing_y=5) 
    return plt


def calculate_spend_effect_share(df_shap_values: pd.DataFrame, media_channels, df_original: pd.DataFrame):
    """
    Args:
        df_shap_values: data frame of shap values
        media_channels: list of media channel names
        df_original: non transformed original data
    Returns: 
        [pd.DataFrame]: data frame with spend effect shares
    """
    responses = pd.DataFrame(df_shap_values[media_channels].abs().sum(axis = 0), columns = ["effect_share"])
    response_percentages = responses / responses.sum()
    response_percentages

    spends_percentages = pd.DataFrame(df_original[media_channels].sum(axis = 0) / df_original[media_channels].sum(axis = 0).sum(), columns = ["spend_share"])
    spends_percentages

    spend_effect_share = pd.merge(response_percentages, spends_percentages, left_index = True, right_index = True)
    spend_effect_share = spend_effect_share.reset_index().rename(columns = {"index": "media"})
    
    return spend_effect_share

import streamlit as st
#https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d
def shap_feature_importance(shap_values, data, figsize = (20, 10)):
    
    feature_list = data.columns
    
    if isinstance(shap_values, pd.DataFrame) == False:
        shap_v = pd.DataFrame(shap_values)
        shap_v.columns = feature_list
    else:
        shap_v = shap_values
    
        
    df_v = data.copy().reset_index().drop('index',axis=1)
    
    # Determine the correlation in order to plot with different colors
    corr_list = list()
    for i in feature_list:
        b = np.corrcoef(shap_v[i],df_v[i])[1][0]
        corr_list.append(b)
    corr_df = pd.concat([pd.Series(feature_list),pd.Series(corr_list)],axis=1).fillna(0)
    # Make a data frame. Column 1 is the feature, and Column 2 is the correlation coefficient
    corr_df.columns  = ['Variable','Corr']
    corr_df['Sign'] = np.where(corr_df['Corr']>0,'red','blue')
    
    # Plot it
    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns = ['Variable','SHAP_abs']
    k2 = k.merge(corr_df,left_on = 'Variable',right_on='Variable',how='inner')
    k2 = k2.sort_values(by='SHAP_abs',ascending = True)
    colorlist = k2['Sign']
    fig, ax = plt.subplots(figsize=figsize)
    ax = k2.plot.barh(x='Variable',y='SHAP_abs',color = colorlist, figsize=figsize,legend=False)
    ax.set_xlabel("SHAP Value (Red = Positive Impact)")
    

def model_refit(data, 
                target, 
                features, 
                media_channels,
                model_params, 
                adstock_params, 
                start_index, 
                end_index):
    data_refit = data.copy()

    best_params = model_params

    adstock_alphas = adstock_params

    #apply adstock transformation
    for feature in media_channels:
        adstock_alpha = adstock_alphas[feature]
        # print(f"applying geometric adstock transformation on {feature} with alpha {adstock_alpha}") 

        #adstock transformation
        x_feature = data_refit[feature].values.reshape(-1, 1)
        temp_adstock = AdstockGeometric(alpha = adstock_alpha).fit_transform(x_feature)
        data_refit[feature] = temp_adstock

    #build the final model on the data until the end analysis index
    x_input = data_refit.loc[0:end_index-1, features]
    y_true_all = data[target].values[0:end_index]

    #build random forest using the best parameters
    random_forest = RandomForestRegressor(random_state=0, **best_params)
    random_forest.fit(x_input, y_true_all) 


    #concentrate on the analysis interval
    y_true_interval = y_true_all[start_index:end_index]
    x_input_interval_transformed = x_input.iloc[start_index:end_index]

    #revenue prediction for the analysis interval
    # print(f"predicting {len(x_input_interval_transformed)}")
    prediction = random_forest.predict(x_input_interval_transformed)

    #transformed data set for the analysis interval 
    x_input_interval_nontransformed = data.iloc[start_index:end_index]


    #shap explainer 
    explainer = shap.TreeExplainer(random_forest)

    # get SHAP values for the data set for the analysis interval from explainer model
    shap_values_train = explainer.shap_values(x_input_interval_transformed)

    # create a dataframe of the shap values for the training set and the test set
    df_shap_values = pd.DataFrame(shap_values_train, columns=features)
    
    return {
            'df_shap_values': df_shap_values, 
            'x_input_interval_nontransformed': x_input_interval_nontransformed, 
            'x_input_interval_transformed' : x_input_interval_transformed,
            'prediction_interval': prediction, 
            'y_true_interval': y_true_interval,
            'model' : random_forest
           }
    
def plot_shap_vs_spend(df_shap_values, x_input_interval_nontransformed, x_input_interval_transformed, features, media_channels, figsize=(25, 10)):
    for channel in media_channels:
    
        #index = features.index(channel)

        mean_spend = x_input_interval_nontransformed.loc[x_input_interval_nontransformed[channel] > 0, channel].mean()

        fig, ax = plt.subplots(figsize=figsize)
        sns.regplot(x = x_input_interval_transformed[channel], y = df_shap_values[channel], label = channel,
                    scatter_kws={'alpha': 0.65}, line_kws={'color': 'C2', 'linewidth': 6},
                    lowess=True, ax=ax).set(title=f'{channel}: Spend vs Shapley')
        ax.axhline(0, linestyle = "--", color = "black", alpha = 0.5)
        ax.axvline(mean_spend, linestyle = "--", color = "red", alpha = 0.5, label=f"Average Spend: {int(mean_spend)}")
        ax.set_xlabel(f"{channel} spend")
        ax.set_ylabel(f'SHAP Value for {channel}')
        plt.legend()

def optuna_trial(trial, 
                 data:pd.DataFrame, 
                 target, 
                 features, 
                 adstock_features, 
                 adstock_features_params, 
                 media_features, 
                 tscv, 
                 is_multiobjective = False):
    
    data_temp = data.copy()
    adstock_alphas = {}
    
    for feature in adstock_features:
        adstock_param = f"{feature}_adstock"
        min_, max_ = adstock_features_params[adstock_param]
        adstock_alpha = trial.suggest_uniform(f"adstock_alpha_{feature}", min_, max_)
        adstock_alphas[feature] = adstock_alpha
        
        #adstock transformation
        x_feature = data[feature].values.reshape(-1, 1)
        temp_adstock = AdstockGeometric(alpha = adstock_alpha).fit_transform(x_feature)
        data_temp[feature] = temp_adstock
        
        
    #Random Forest parameters
    n_estimators = trial.suggest_int("n_estimators", 5, 100)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    max_depth = trial.suggest_int("max_depth", 4,7)
    ccp_alpha = trial.suggest_uniform("ccp_alpha", 0, 0.3)
    bootstrap = trial.suggest_categorical("bootstrap", [False, True])
    criterion = trial.suggest_categorical("criterion", ["squared_error"])  #"absolute_error"
    
    scores = []
    
    rssds = []
    for train_index, test_index in tscv.split(data_temp):
        x_train = data_temp.iloc[train_index][features]
        y_train =  data_temp[target].values[train_index]
        
        x_test = data_temp.iloc[test_index][features]
        y_test = data_temp[target].values[test_index]
        
        #apply Random Forest
        params = {"n_estimators": n_estimators, 
                   "min_samples_leaf":min_samples_leaf, 
                   "min_samples_split" : min_samples_split,
                   "max_depth" : max_depth, 
                   "ccp_alpha" : ccp_alpha, 
                   "bootstrap" : bootstrap, 
                   "criterion" : criterion
                 }
        
        rf = RandomForestRegressor(random_state=0, **params)
        rf.fit(x_train, y_train)
        prediction = rf.predict(x_test)
        
        rmse = mean_squared_error(y_true = y_test, y_pred = prediction, squared = False)
        scores.append(rmse)
        
        if is_multiobjective:
            
            #set_trace()
            #calculate spend effect share -> rssd
            # create explainer model by passing trained model to shap
            explainer = shap.TreeExplainer(rf)

            # get Shap values
            shap_values_train = explainer.shap_values(x_train)
            
            df_shap_values = pd.DataFrame(shap_values_train, columns=features)

            spend_effect_share = calculate_spend_effect_share(df_shap_values = df_shap_values, media_channels = media_features, df_original = data.iloc[train_index])

            decomp_rssd = rssd(effect_share = spend_effect_share.effect_share.values, spend_share = spend_effect_share.spend_share.values)
            rssds.append(decomp_rssd)
    
    trial.set_user_attr("scores", scores)
    
    trial.set_user_attr("params", params)
    trial.set_user_attr("adstock_alphas", adstock_alphas)
    
    if is_multiobjective == False:
        return np.mean(scores)
    
    
    trial.set_user_attr("rssds", rssds)
        
    #multiobjective
    return np.mean(scores), np.mean(rssds)

def optuna_optimize(trials, 
                    data: pd.DataFrame, 
                    target, 
                    features, 
                    adstock_features, 
                    adstock_features_params, 
                    media_features, 
                    tscv, 
                    is_multiobjective, 
                    seed = 42):
    # print(f"data size: {len(data)}")
    # print(f"media features: {media_features}")
    # print(f"adstock features: {adstock_features}")
    # print(f"features: {features}")
    # print(f"is_multiobjective: {is_multiobjective}")
    opt.logging.set_verbosity(opt.logging.WARNING) 
    
    if is_multiobjective == False:
        # crea un estudio de optimización en el que se minimizará una función objetivo utilizando el algoritmo de muestreo de TPE.
        study_mmm = opt.create_study(direction='minimize', sampler = opt.samplers.TPESampler(seed=seed))  
    else:
        study_mmm = opt.create_study(directions=["minimize", "minimize"], sampler=opt.samplers.NSGAIISampler(seed=seed))
        
    optimization_function = partial(optuna_trial, 
                                    data = data, 
                                    target = target, 
                                    features = features, 
                                    adstock_features = adstock_features, 
                                    adstock_features_params = adstock_features_params, 
                                    media_features = media_features, 
                                    tscv = tscv, 
                                    is_multiobjective = is_multiobjective)
    
    
    study_mmm.optimize(optimization_function, n_trials = trials, show_progress_bar = True)
    
    return study_mmm



def calculated_incerement_sales(model,
                                growing,
                                shap_values,
                                data_input_nontransformed,
                                data_store_group_wi,
                                features):
    # Si solo tiene dos variables, son trend y season. No tiene campañas asignadas. No sirve.
    if len(features) == 2:
         return "No tiene campañas asignadas."
    
    shap_values_scope = shap_values
    data_input_nontransformed = data_input_nontransformed
    feature_list = data_input_nontransformed.columns

    if isinstance(shap_values_scope, pd.DataFrame) == False:
            shap_v = pd.DataFrame(shap_values_scope)
            shap_v.columns = feature_list
    else:
        shap_v = shap_values_scope

    shap_abs = np.abs(shap_v)
    k=pd.DataFrame(shap_abs.mean()).reset_index()
    k.columns=['reason','score']

    # Conseguimos el promedio de las últimas 4 semanas
    # Cirterio asumido, tomamos el promedio de las últimos 3 meses y si no tiene datos
    # tomamos el promedio de todos los datos
    date_to_estimate = datetime.today()
    date_to_trashold = date_to_estimate - timedelta(days=1*30)
    isoweeek_serie = pd.to_datetime(data_store_group_wi['ISOweek'])
    data_store_filter_by_date = data_store_group_wi[isoweeek_serie > date_to_trashold]
    
    if not data_store_filter_by_date.empty:
        average_sales = data_store_filter_by_date['sales'].mean()
    else:
        date_to_estimate = max(isoweeek_serie) + timedelta(days=7)
        date_to_trashold = date_to_estimate - timedelta(days=1*30)
        data_store_filter_by_date = data_store_group_wi[isoweeek_serie > date_to_trashold]
        average_sales = data_store_filter_by_date['sales'].mean()

    media_channels_reason = k['reason'].tolist()
    list_channel_with_score_0 = []
    for indice, fila in k.iterrows():
        if fila['score'] == 0:
            media_channels_reason.remove(fila['reason'])
            list_channel_with_score_0.append(fila['reason'])

    media_channels_reason.remove('trend')
    media_channels_reason.remove('season')

    # Si media_channels es vacia quiere decir que facebook y google no explican ventas. No sirve.
    if media_channels_reason == []:
        return "Google o Facebook no explican las ventas."
    
    store_group_name = data_store_group_wi['concat_store_group_name'].unique()[0]

    if os.path.exists(f"models/{store_group_name}.pkl"):
        with open(f"models/{store_group_name}.pkl", 'rb') as f:
            prophet = pickle.load(f)
    else:
        prophet = Prophet(yearly_seasonality=True)
        prophet.fit(table_prophet_index)
        with open(f"models/{store_group_name}.pkl", 'wb') as f:
            pickle.dump(prophet, f)

    date_to_estimate = date_to_estimate.strftime("%Y-%m-%d")
    dataframe = pd.DataFrame(
        {"ds":[date_to_estimate]}
    )
    
    prohet_prediction = prophet.predict(dataframe)
    dataframe['trend'] = prohet_prediction['trend']
    dataframe['season'] = prohet_prediction['yearly']

    target_sales = average_sales * (1+growing/100)

    shares_to_channels = {}
    for channel in media_channels_reason:
        shares_to_channels[channel] = data_input_nontransformed[channel].sum()
    total_investment = sum(shares_to_channels.values())
    shares_channels = {}
    for key, value in shares_to_channels.items():
        shares_channels[key] = value / total_investment

    # Calculamos el incremento de ventas
    increment_sales = 0
    investment = 100
    increment = 40
    limit_max_investment = 20000
    while increment_sales < target_sales:
        prohet_prediction_cost_campaign = dataframe.copy()
        for channel in media_channels_reason:
            prohet_prediction_cost_campaign[channel] = investment * shares_channels[channel]
        for channel in list_channel_with_score_0:
            prohet_prediction_cost_campaign[channel] = 0

        # display(prohet_prediction_cost_campaign)
        increment_sales = model.predict(prohet_prediction_cost_campaign[features])[0]
        # time.sleep(3)
        if investment > limit_max_investment:
            return f"La inversión supera los {limit_max_investment}"
        investment += increment

    return f"El monto a invertir para crecer un <b>{growing}%</b> es $ <b>{investment}</b>"