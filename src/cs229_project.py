# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:38:57 2025

@author: pthar
"""

from pybaseball import statcast
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, average_precision_score, mean_squared_error, r2_score
import time

class environment_state:
    def __init__(self,
                 balls = 0,
                 strikes = 0,
                 outs_when_up = 0,
                 runner1 = 0,
                 runner2 = 0,
                 runner3 = 0,
                 stance_vs_arm_same = 1,
                 sz_top = 3.41754,
                 sz_bot = 1.60272
                 ):
        self.balls = balls
        self.strikes = strikes
        self.outs_when_up = outs_when_up
        self.runner1 = runner1
        self.runner2 = runner2
        self.runner3 = runner3
        self.stance_vs_arm_same = stance_vs_arm_same
        self.sz_top = sz_top
        self.sz_bot = sz_bot
    def update(self, **kwargs):
        for name, value in kwargs.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                raise ValueError(f'No attribute called {name}')
    def copy(self):
        return environment_state(
                balls=self.balls,
                strikes=self.strikes,
                outs_when_up=self.outs_when_up,
                runner1=self.runner1,
                runner2=self.runner2,
                runner3=self.runner3,
                stance_vs_arm_same=self.stance_vs_arm_same,
                sz_top=self.sz_top,
                sz_bot=self.sz_bot
            )
    
# Get Statcast data for a specific date range
def get_data():
    data = statcast(start_dt='2024-04-01', end_dt='2025-11-07') 
    data2 = data.copy(deep=True)

    #create booleans for runners on base
    data['runner1'] = data['on_1b'].notna()*1
    data['runner2'] = data['on_2b'].notna()*1
    data['runner3'] = data['on_3b'].notna()*1
    
    #data[['zone','plate_x','plate_z']].groupby('zone').agg(['max','min'])
    
    swing_desc = ['foul','foul_pitchout','foul_tip','hit_into_play','swinging_strike','swinging_strike_blocked', 'bunt_foul_tip','foul_bunt','missed_bunt']
    #bunt_desc = []
    
    excluded_descs = ['automatic_ball', 'automatic_strike', 'intent_ball', 'pitchout']
    data = data[~(data.description.isin(excluded_descs))]
    adj_pitches = ['Pitch Out']
    data.loc[data.pitch_name.isin(adj_pitches), 'pitch_name'] = 'Other'
    
    
    data['swing'] = 0
    data.loc[data['description'].isin(swing_desc),'swing'] = 1
    data['sz_mid'] = (data.sz_top+data.sz_bot)/2.0
    data['pitch_vs_sz_mid'] = np.sqrt((data['plate_z']-data['sz_mid'])**2+data['plate_x']**2)
    
    data = data[~(data.pitch_name.isna())]
    data = data[~(data.pitch_name=='Unknown')]
    data = data[~(data.zone.isna())]
    data = data[~(data.api_break_z_with_gravity.isna())]
    data = data[~(data.release_speed.isna())]
    
    data = data.reset_index(drop=True)
    
    data['stance_vs_arm_same'] = 1*(data['stand']==data['p_throws'])
    
    
    data['no_swing_cat'] = None
    data.loc[data.description.isin(['hit_by_pitch']),'no_swing_cat'] = 'hit_by_pitch'
    data.loc[data.description.isin(['called_strike']),'no_swing_cat'] = 'strike'
    data.loc[data.description.isin(['ball', 'blocked_ball']),'no_swing_cat'] = 'ball'
    
    data['swing_cat'] = None
    data.loc[data.description.isin(['foul','foul_pitchout']),'swing_cat'] = 'foul'
    data.loc[data.description.isin(['foul_tip','swinging_strike','swinging_strike_blocked', 'bunt_foul_tip','foul_bunt','missed_bunt']),'swing_cat'] = 'strike'
    data.loc[data.description.isin(['hit_into_play']),'swing_cat'] = 'hit_into_play'
    
    data['ball_in_play_result'] = None
    
    data.loc[data.events.isin(['home_run']),'ball_in_play_result'] = 'home_run'
    data.loc[data.events.isin(['triple']),'ball_in_play_result'] = 'triple'
    data.loc[data.events.isin(['double']),'ball_in_play_result'] = 'double'
    data.loc[data.events.isin(['single']),'ball_in_play_result'] = 'single'
    data.loc[data.events.isin(['double_play','grounded_into_double_play','sac_fly_double_play']),'ball_in_play_result'] = 'double_play'
    data.loc[data.events.isin(['triple_play']),'ball_in_play_result'] = 'triple_play'
    data.loc[data.events.isin(['field_error']),'ball_in_play_result'] = 'error'
    data.loc[data.events.isin(['catcher_interf']),'ball_in_play_result'] = 'catcher_interf'
    data.loc[data.events.isin([ 'sac_bunt', 'sac_fly']),'ball_in_play_result'] = 'sacrifice'
    data.loc[data.events.isin([ 'field_out','fielders_choice','fielders_choice_out','force_out']),'ball_in_play_result'] = 'out'
    
    data['game_date'] = pd.to_datetime(data['game_date'])
    
    data['pitch_outcome'] = None
    data.loc[data['swing'] == 0, 'pitch_outcome'] = data.loc[data['swing'] == 0, 'no_swing_cat']
    data.loc[data['swing'] == 1, 'pitch_outcome'] = data.loc[data['swing'] == 1, 'swing_cat']
    data.loc[data['swing_cat'] == 'hit_into_play', 'pitch_outcome'] = data.loc[data['swing_cat'] == 'hit_into_play', 'ball_in_play_result']
    
    data['pitch_zone'] = data['pitch_name'].astype(str) + '_' + data['zone'].astype(str)
        
    train = data.loc[data['game_date'] <= '2025-07-31'].copy(deep=True)
    validate = data.loc[(data['game_date'] >= '2025-08-01') & (data['game_date'] <= '2025-08-31')].copy(deep=True)
    test = data.loc[(data['game_date'] >= '2025-09-01') & (data['game_date'] <= '2025-09-30')].copy(deep=True)
    
    return train, validate, test, data2

def p_swing_decision(train, validate):
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'plate_x',
                           'plate_z',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'pitch_vs_sz_mid',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train[categorical_variables+numerical_variables]
    train_ys = train['swing']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', LogisticRegression(max_iter=1000))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate[categorical_variables+numerical_variables]
    validate_ys = validate['swing']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train.groupby(['pitch_name','zone'])['swing'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    
    train_predict = model.predict_proba(train_xs)[:,1]
    validate_predict = model.predict_proba(validate_xs)[:,1]
    
    evaluation_metrics = {
        'train_log_loss': log_loss(train_ys, train_predict),
        'train_auprc': average_precision_score(train_ys, train_predict),
        
        'validate_log_loss': log_loss(validate_ys, validate_predict),
        'validate_auprc': average_precision_score(validate_ys, validate_predict),
    
        'train_baseline_log_loss': log_loss(train_ys, train_baseline),
        'train_baseline_auprc': average_precision_score(train_ys, train_baseline),
    
        'validate_baseline_log_loss': log_loss(validate_ys, validate_baseline),
        'validate_baseline_auprc': average_precision_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_log_loss': log_loss(train_ys, train_smart_baseline),
        'train_smart_baseline_auprc': average_precision_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_log_loss': log_loss(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_auprc': average_precision_score(validate_ys, validate_smart_baseline)
        }
    return model, evaluation_metrics, categorical_variables+numerical_variables

def p_no_swing_outcome(train, validate):
    train_modified = train[train.swing==0].copy(deep=True)
    validate_modified = validate[validate.swing==0].copy(deep=True)
    categorical_variables = ['pitch_name','zone', 'pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'plate_x',
                           'plate_z',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'pitch_vs_sz_mid',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['no_swing_cat']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', LogisticRegression(multi_class = 'multinomial',max_iter=1000))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['no_swing_cat']
    
    baseline = train_ys.value_counts(normalize=True).reindex(model.classes_)
    train_baseline = pd.DataFrame([baseline]*len(train_ys)).reset_index(drop=True)
    validate_baseline = pd.DataFrame([train_ys.value_counts(normalize=True).reindex(model.classes_)]*len(validate_ys)).reset_index(drop=True)
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['no_swing_cat'].value_counts(normalize=True).reset_index().pivot(index=['pitch_name','zone'], columns = 'no_swing_cat', values = 'proportion').reindex(columns = model.classes_).fillna(0).reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(model.classes_)]
    train_smart_baseline[train_smart_baseline[list(model.classes_)].sum(axis=1)==0] = baseline
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(model.classes_)]
    validate_smart_baseline[validate_smart_baseline[list(model.classes_)].sum(axis=1)==0] = baseline
    
    
    train_predict = pd.DataFrame(model.predict_proba(train_xs), columns = model.classes_)
    validate_predict = pd.DataFrame(model.predict_proba(validate_xs), columns = model.classes_)
    
    train_ys_binarized = label_binarize(train_ys, classes = model.classes_)
    validate_ys_binarized = label_binarize(validate_ys, classes = model.classes_)
    
    evaluation_metrics = {
        'train_log_loss': log_loss(train_ys_binarized, train_predict),
        'train_auprc': average_precision_score(train_ys_binarized, train_predict),
        'train_auprc_by_class': average_precision_score(train_ys_binarized, train_predict, average=None),
        
        'validate_log_loss': log_loss(validate_ys_binarized, validate_predict),
        'validate_auprc': average_precision_score(validate_ys_binarized, validate_predict),
        'validate_auprc_by_class': average_precision_score(validate_ys_binarized, validate_predict, average=None),
        
        'train_baseline_log_loss': log_loss(train_ys_binarized, train_baseline),
        'train_baseline_auprc': average_precision_score(train_ys_binarized, train_baseline),
        'train_baseline_auprc_by_class': average_precision_score(train_ys_binarized, train_baseline, average=None),
    
        'validate_baseline_log_loss': log_loss(validate_ys_binarized, validate_baseline),
        'validate_baseline_auprc': average_precision_score(validate_ys_binarized, validate_baseline),
        'validate_baseline_auprc_by_class': average_precision_score(validate_ys_binarized, validate_baseline, average=None),
    
        'train_smart_baseline_log_loss': log_loss(train_ys_binarized, train_smart_baseline),
        'train_smart_baseline_auprc': average_precision_score(train_ys_binarized, train_smart_baseline),
        'train_smart_baseline_auprc_by_class': average_precision_score(train_ys_binarized, train_smart_baseline, average=None),
    
        'validate_smart_baseline_log_loss': log_loss(validate_ys_binarized, validate_smart_baseline),
        'validate_smart_baseline_auprc': average_precision_score(validate_ys_binarized, validate_smart_baseline),
        'validate_smart_baseline_auprc_by_class': average_precision_score(validate_ys_binarized, validate_smart_baseline, average=None)
        }
    return model, evaluation_metrics, categorical_variables+numerical_variables

def p_swing_outcome(train, validate):
    train_modified = train[train.swing==1].copy(deep=True)
    validate_modified = validate[validate.swing==1].copy(deep=True)
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'plate_x',
                           'plate_z',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'pitch_vs_sz_mid',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['swing_cat']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', LogisticRegression(multi_class = 'multinomial',max_iter=1000))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['swing_cat']
    
    baseline = train_ys.value_counts(normalize=True).reindex(model.classes_)
    train_baseline = pd.DataFrame([baseline]*len(train_ys)).reset_index(drop=True)
    validate_baseline = pd.DataFrame([train_ys.value_counts(normalize=True).reindex(model.classes_)]*len(validate_ys)).reset_index(drop=True)
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['swing_cat'].value_counts(normalize=True).reset_index().pivot(index=['pitch_name','zone'], columns = 'swing_cat', values = 'proportion').reindex(columns = model.classes_).fillna(0).reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(model.classes_)]
    train_smart_baseline[train_smart_baseline[list(model.classes_)].sum(axis=1)==0] = baseline
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(model.classes_)]
    validate_smart_baseline[validate_smart_baseline[list(model.classes_)].sum(axis=1)==0] = baseline
    
    
    train_predict = pd.DataFrame(model.predict_proba(train_xs), columns = model.classes_)
    validate_predict = pd.DataFrame(model.predict_proba(validate_xs), columns = model.classes_)
    
    train_ys_binarized = label_binarize(train_ys, classes = model.classes_)
    validate_ys_binarized = label_binarize(validate_ys, classes = model.classes_)
    
    evaluation_metrics = {
        'train_log_loss': log_loss(train_ys_binarized, train_predict),
        'train_auprc': average_precision_score(train_ys_binarized, train_predict),
        'train_auprc_by_class': average_precision_score(train_ys_binarized, train_predict, average=None),
        
        'validate_log_loss': log_loss(validate_ys_binarized, validate_predict),
        'validate_auprc': average_precision_score(validate_ys_binarized, validate_predict),
        'validate_auprc_by_class': average_precision_score(validate_ys_binarized, validate_predict, average=None),
        
        'train_baseline_log_loss': log_loss(train_ys_binarized, train_baseline),
        'train_baseline_auprc': average_precision_score(train_ys_binarized, train_baseline),
        'train_baseline_auprc_by_class': average_precision_score(train_ys_binarized, train_baseline, average=None),
    
        'validate_baseline_log_loss': log_loss(validate_ys_binarized, validate_baseline),
        'validate_baseline_auprc': average_precision_score(validate_ys_binarized, validate_baseline),
        'validate_baseline_auprc_by_class': average_precision_score(validate_ys_binarized, validate_baseline, average=None),
    
        'train_smart_baseline_log_loss': log_loss(train_ys_binarized, train_smart_baseline),
        'train_smart_baseline_auprc': average_precision_score(train_ys_binarized, train_smart_baseline),
        'train_smart_baseline_auprc_by_class': average_precision_score(train_ys_binarized, train_smart_baseline, average=None),
    
        'validate_smart_baseline_log_loss': log_loss(validate_ys_binarized, validate_smart_baseline),
        'validate_smart_baseline_auprc': average_precision_score(validate_ys_binarized, validate_smart_baseline),
        'validate_smart_baseline_auprc_by_class': average_precision_score(validate_ys_binarized, validate_smart_baseline, average=None)
        }
    return model, evaluation_metrics, categorical_variables+numerical_variables

def p_hit_into_play_outcome(train, validate):
    train_modified = train[train.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    validate_modified = validate[validate.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    categorical_variables = ['pitch_name','zone', 'pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'plate_x',
                           'plate_z',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'pitch_vs_sz_mid',
                           'bat_speed',
                           'swing_length',
                           'launch_angle',
                           'launch_speed'
                           ]
    
    train_modified = train_modified[~(train_modified[numerical_variables].isna().any(axis=1))]
    train_modified = train_modified[~(train_modified['events'].isna())]
    validate_modified = validate_modified[~(validate_modified[numerical_variables].isna().any(axis=1))]
    validate_modified = validate_modified[~(validate_modified['events'].isna())]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['ball_in_play_result']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', LogisticRegression(multi_class = 'multinomial', max_iter = 10000))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['ball_in_play_result']
    
    baseline = train_ys.value_counts(normalize=True).reindex(model.classes_)
    train_baseline = pd.DataFrame([baseline]*len(train_ys)).reset_index(drop=True)
    validate_baseline = pd.DataFrame([train_ys.value_counts(normalize=True).reindex(model.classes_)]*len(validate_ys)).reset_index(drop=True)
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['ball_in_play_result'].value_counts(normalize=True).reset_index().pivot(index=['pitch_name','zone'], columns = 'ball_in_play_result', values = 'proportion').reindex(columns = model.classes_).fillna(0).reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(model.classes_)]
    train_smart_baseline[train_smart_baseline[list(model.classes_)].sum(axis=1)==0] = baseline
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(model.classes_)]
    validate_smart_baseline[validate_smart_baseline[list(model.classes_)].sum(axis=1)==0] = baseline
    
    
    train_predict = pd.DataFrame(model.predict_proba(train_xs), columns = model.classes_)
    validate_predict = pd.DataFrame(model.predict_proba(validate_xs), columns = model.classes_)
    
    train_ys_binarized = label_binarize(train_ys, classes = model.classes_)
    validate_ys_binarized = label_binarize(validate_ys, classes = model.classes_)
    
    evaluation_metrics = {
        'train_log_loss': log_loss(train_ys_binarized, train_predict),
        'train_auprc': average_precision_score(train_ys_binarized, train_predict),
        'train_auprc_by_class': average_precision_score(train_ys_binarized, train_predict, average=None),
        
        'validate_log_loss': log_loss(validate_ys_binarized, validate_predict),
        'validate_auprc': average_precision_score(validate_ys_binarized, validate_predict),
        'validate_auprc_by_class': average_precision_score(validate_ys_binarized, validate_predict, average=None),
        
        'train_baseline_log_loss': log_loss(train_ys_binarized, train_baseline),
        'train_baseline_auprc': average_precision_score(train_ys_binarized, train_baseline),
        'train_baseline_auprc_by_class': average_precision_score(train_ys_binarized, train_baseline, average=None),
    
        'validate_baseline_log_loss': log_loss(validate_ys_binarized, validate_baseline),
        'validate_baseline_auprc': average_precision_score(validate_ys_binarized, validate_baseline),
        'validate_baseline_auprc_by_class': average_precision_score(validate_ys_binarized, validate_baseline, average=None),
    
        'train_smart_baseline_log_loss': log_loss(train_ys_binarized, train_smart_baseline),
        'train_smart_baseline_auprc': average_precision_score(train_ys_binarized, train_smart_baseline),
        'train_smart_baseline_auprc_by_class': average_precision_score(train_ys_binarized, train_smart_baseline, average=None),
    
        'validate_smart_baseline_log_loss': log_loss(validate_ys_binarized, validate_smart_baseline),
        'validate_smart_baseline_auprc': average_precision_score(validate_ys_binarized, validate_smart_baseline),
        'validate_smart_baseline_auprc_by_class': average_precision_score(validate_ys_binarized, validate_smart_baseline, average=None)
        }
    return model, evaluation_metrics, categorical_variables+numerical_variables

def release_speed_modeled(train, validate):
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same'
                           ]
    train_xs = train[categorical_variables+numerical_variables]
    train_ys = train['release_speed']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate[categorical_variables+numerical_variables]
    validate_ys = validate['release_speed']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train.groupby(['pitch_name','zone'])['release_speed'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    
    temp = train.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def api_break_z_with_gravity_modeled(train, validate):
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed'
                           ]
    train_xs = train[categorical_variables+numerical_variables]
    train_ys = train['api_break_z_with_gravity']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate[categorical_variables+numerical_variables]
    validate_ys = validate['api_break_z_with_gravity']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train.groupby(['pitch_name','zone'])['api_break_z_with_gravity'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def api_break_x_batter_in_modeled(train, validate):
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity'
                           ]
    train_xs = train[categorical_variables+numerical_variables]
    train_ys = train['api_break_x_batter_in']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf', RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate[categorical_variables+numerical_variables]
    validate_ys = validate['api_break_x_batter_in']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train.groupby(['pitch_name','zone'])['api_break_x_batter_in'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def plate_x_modeled(train, validate):
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in'
                           ]
    train_xs = train[categorical_variables+numerical_variables]
    train_ys = train['plate_x']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf',  RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate[categorical_variables+numerical_variables]
    validate_ys = validate['plate_x']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train.groupby(['pitch_name','zone'])['plate_x'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def plate_z_modeled(train, validate):
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'plate_x',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train[categorical_variables+numerical_variables]
    train_ys = train['plate_z']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf',  RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate[categorical_variables+numerical_variables]
    validate_ys = validate['plate_z']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train.groupby(['pitch_name','zone'])['plate_z'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def swing_length_modeled(train, validate):
    train_modified = train[train.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    validate_modified = validate[validate.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    train_modified = train_modified[~(train_modified['swing_length'].isna())]
    validate_modified = validate_modified[~(validate_modified['swing_length'].isna())]
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'plate_x',
                           'plate_z',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['swing_length']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf',  RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['swing_length']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['swing_length'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train_modified.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def bat_speed_modeled(train, validate):
    train_modified = train[train.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    validate_modified = validate[validate.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    train_modified = train_modified[~(train_modified['bat_speed'].isna())]
    validate_modified = validate_modified[~(validate_modified['bat_speed'].isna())]
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'plate_x',
                           'plate_z',
                           'swing_length'
                           ]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['bat_speed']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf',  RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['bat_speed']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['bat_speed'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train_modified.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def launch_angle_modeled(train, validate):
    train_modified = train[train.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    validate_modified = validate[validate.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    train_modified = train_modified[~(train_modified['bat_speed'].isna())]
    validate_modified = validate_modified[~(validate_modified['bat_speed'].isna())]
    train_modified = train_modified[~(train_modified['launch_angle'].isna())]
    validate_modified = validate_modified[~(validate_modified['launch_angle'].isna())]
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'plate_x',
                           'plate_z',
                           'swing_length',
                           'bat_speed',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['launch_angle']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf',  RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['launch_angle']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['launch_angle'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train_modified.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def launch_speed_modeled(train, validate):
    train_modified = train[train.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    validate_modified = validate[validate.description.isin(['hit_into_play'])].reset_index(drop=True).copy(deep=True)
    train_modified = train_modified[~(train_modified['bat_speed'].isna())]
    validate_modified = validate_modified[~(validate_modified['bat_speed'].isna())]
    train_modified = train_modified[~(train_modified['launch_angle'].isna())]
    validate_modified = validate_modified[~(validate_modified['launch_angle'].isna())]
    train_modified = train_modified[~(train_modified['launch_speed'].isna())]
    validate_modified = validate_modified[~(validate_modified['launch_speed'].isna())]
    categorical_variables = ['pitch_name','zone','pitch_zone']
    numerical_variables = ['outs_when_up',
                           'runner1',
                           'runner2',
                           'runner3',
                           'balls',
                           'strikes',
                           'stance_vs_arm_same',
                           'release_speed',
                           'api_break_z_with_gravity',
                           'api_break_x_batter_in',
                           'plate_x',
                           'plate_z',
                           'swing_length',
                           'bat_speed',
                           'launch_angle',
                           'sz_top',
                           'sz_bot'
                           ]
    train_xs = train_modified[categorical_variables+numerical_variables]
    train_ys = train_modified['launch_speed']
    
    preprocessed_data = ColumnTransformer([('categorical',OneHotEncoder(handle_unknown='ignore'), categorical_variables),('numerical',StandardScaler(), numerical_variables)])
    model = Pipeline([('preprocessed', preprocessed_data),('clf',  RidgeCV(alphas = [0.01,0.1,1,10]))])
    model.fit(train_xs,train_ys)
    
    validate_xs = validate_modified[categorical_variables+numerical_variables]
    validate_ys = validate_modified['launch_speed']
    
    baseline_rate = float(train_ys.mean())
    train_baseline = np.full(len(train_ys),float(baseline_rate))
    validate_baseline = np.full(len(validate_ys),float(baseline_rate))
    
    smart_baselines = train_modified.groupby(['pitch_name','zone'])['launch_speed'].mean().rename('smart_baseline').reset_index()
    train_smart_baseline = train_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline']
    validate_smart_baseline = validate_modified[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')['smart_baseline'].fillna(baseline_rate)
    
    train_predict = model.predict(train_xs)
    validate_predict = model.predict(validate_xs)
    
    evaluation_metrics = {
        'train_mse': mean_squared_error(train_ys, train_predict),
        'train_r2': r2_score(train_ys, train_predict),
        
        'validate_mse': mean_squared_error(validate_ys, validate_predict),
        'validate_r2': r2_score(validate_ys, validate_predict),
    
        'train_baseline_mse': mean_squared_error(train_ys, train_baseline),
        'train_baseline_r2': r2_score(train_ys, train_baseline),
    
        'validate_baseline_mse': mean_squared_error(validate_ys, validate_baseline),
        'validate_baseline_r2': r2_score(validate_ys, validate_baseline),
    
        'train_smart_baseline_mse': mean_squared_error(train_ys, train_smart_baseline),
        'train_smart_baseline_r2': r2_score(train_ys, train_smart_baseline),
    
        'validate_smart_baseline_mse': mean_squared_error(validate_ys, validate_smart_baseline),
        'validate_smart_baseline_r2': r2_score(validate_ys, validate_smart_baseline)
        }
    temp = train_modified.copy(deep=True)
    temp['residual'] = train_ys - train_predict
    
    sd_by_pitch = temp.groupby(['pitch_name'])['residual'].std().rename('stddev')
    sd_by_pitch_zone = temp.groupby(['pitch_name','zone'])['residual'].std().rename('stddev')
    counts = temp.groupby(['pitch_name','zone'])['residual'].count()
    
    std_devs = {}
    for i in sd_by_pitch_zone.index:
        if counts[i]>30:
            std_devs[i] = sd_by_pitch_zone[i]
        else:
            std_devs[i] = sd_by_pitch[i[0]]   
    
    return model, evaluation_metrics, categorical_variables+numerical_variables, std_devs

def sample_sz(data):
    sample = data.loc[~data.sz_top.isna(),['sz_top','sz_bot']].sample(n=1)
    return float(sample['sz_top'].values[0]), float(sample['sz_bot'].values[0])

def sample_stance_vs_arm_same(data):
    sample = data.loc[~data.stance_vs_arm_same.isna(),['stance_vs_arm_same']].sample(n=1)
    return float(sample['stance_vs_arm_same'].values[0])

def run_prob_models(train, validate):
    p_swing_decision_model, p_swing_decision_evaluation_metrics, p_swing_decision_variables = p_swing_decision(train, validate)
    p_no_swing_outcome_model, p_no_swing_outcome_evaluation_metrics, p_no_swing_outcome_variables = p_no_swing_outcome(train, validate)
    p_swing_outcome_model, p_swing_outcome_evaluation_metrics, p_swing_outcome_variables = p_swing_outcome(train, validate)
    p_hit_into_play_outcome_model, p_hit_into_play_outcome_evaluation_metrics, p_hit_into_play_outcome_variables = p_hit_into_play_outcome(train, validate)
    
    prob_models = { 
        'p_swing_decision_model': p_swing_decision_model,
        'p_no_swing_outcome_model': p_no_swing_outcome_model,
        'p_swing_outcome_model': p_swing_outcome_model,
        'p_hit_into_play_outcome_model': p_hit_into_play_outcome_model
        }
    
    prob_evaluation_metrics = { 
        'p_swing_decision_evaluation_metrics': p_swing_decision_evaluation_metrics,
        'p_no_swing_outcome_evaluation_metrics': p_no_swing_outcome_evaluation_metrics,
        'p_swing_outcome_evaluation_metrics': p_swing_outcome_evaluation_metrics,
        'p_hit_into_play_outcome_evaluation_metrics': p_hit_into_play_outcome_evaluation_metrics
        }
    
    prob_variables = {
        'p_swing_decision_variables': p_swing_decision_variables,
        'p_no_swing_outcome_variables': p_no_swing_outcome_variables,
        'p_swing_outcome_variables': p_swing_outcome_variables,
        'p_hit_into_play_outcome_variables': p_hit_into_play_outcome_variables
        }
    return prob_models, prob_evaluation_metrics, prob_variables

def run_numeric_models(train, validate):
    release_speed_model, release_speed_evaluation_metrics, release_speed_variables, release_speed_stddevs = release_speed_modeled(train, validate)
    api_break_z_with_gravity_model, api_break_z_with_gravity_evaluation_metrics, api_break_z_with_gravity_variables, api_break_z_with_gravity_stddevs = api_break_z_with_gravity_modeled(train, validate)
    api_break_x_batter_in_model, api_break_x_batter_in_evaluation_metrics, api_break_x_batter_in_variables, api_break_x_batter_in_stddevs = api_break_x_batter_in_modeled(train, validate)
    plate_x_model, plate_x_evaluation_metrics, plate_x_variables, plate_x_stddevs = plate_x_modeled(train, validate)
    plate_z_model, plate_z_evaluation_metrics, plate_z_variables, plate_z_stddevs = plate_z_modeled(train, validate)
    swing_length_model, swing_length_evaluation_metrics, swing_length_variables, swing_length_stddevs = swing_length_modeled(train, validate)
    bat_speed_model, bat_speed_evaluation_metrics, bat_speed_variables, bat_speed_stddevs = bat_speed_modeled(train, validate)
    launch_angle_model, launch_angle_evaluation_metrics, launch_angle_variables, launch_angle_stddevs = launch_angle_modeled(train, validate)
    launch_speed_model, launch_speed_evaluation_metrics, launch_speed_variables, launch_speed_stddevs = launch_speed_modeled(train, validate)
    
    
    numeric_models = {
        'release_speed_model':release_speed_model,
        'api_break_z_with_gravity_model': api_break_z_with_gravity_model,
        'api_break_x_batter_in_model': api_break_x_batter_in_model,
        'plate_x_model':plate_x_model,
        'plate_z_model':plate_z_model,
        'swing_length_model':swing_length_model,
        'bat_speed_model':bat_speed_model,
        'launch_angle_model':launch_angle_model,
        'launch_speed_model':launch_speed_model
        }
    
    numeric_evaluation_metrics = {
        'release_speed_evaluation_metrics':release_speed_evaluation_metrics,
        'api_break_z_with_gravity_evaluation_metrics': api_break_z_with_gravity_evaluation_metrics,
        'api_break_x_batter_in_evaluation_metrics': api_break_x_batter_in_evaluation_metrics,
        'plate_x_evaluation_metrics':plate_x_evaluation_metrics,
        'plate_z_evaluation_metrics':plate_z_evaluation_metrics,
        'swing_length_evaluation_metrics':swing_length_evaluation_metrics,
        'bat_speed_evaluation_metrics':bat_speed_evaluation_metrics,
        'launch_angle_evaluation_metrics':launch_angle_evaluation_metrics,
        'launch_speed_evaluation_metrics':launch_speed_evaluation_metrics
        }
    
    numeric_variables = {
        'release_speed_variables':release_speed_variables,
        'api_break_z_with_gravity_variables': api_break_z_with_gravity_variables,
        'api_break_x_batter_in_variables': api_break_x_batter_in_variables,
        'plate_x_variables':plate_x_variables,
        'plate_z_variables':plate_z_variables,
        'swing_length_variables':swing_length_variables,
        'bat_speed_variables':bat_speed_variables,
        'launch_angle_variables':launch_angle_variables,
        'launch_speed_variables':launch_speed_variables
        }
    
    numeric_stddevs = {
        'release_speed_stddevs':release_speed_stddevs,
        'api_break_z_with_gravity_stddevs': api_break_z_with_gravity_stddevs,
        'api_break_x_batter_in_stddevs': api_break_x_batter_in_stddevs,
        'plate_x_stddevs':plate_x_stddevs,
        'plate_z_stddevs':plate_z_stddevs,
        'swing_length_stddevs':swing_length_stddevs,
        'bat_speed_stddevs':bat_speed_stddevs,
        'launch_angle_stddevs':launch_angle_stddevs,
        'launch_speed_stddevs':launch_speed_stddevs
        }
    
    return numeric_models, numeric_evaluation_metrics, numeric_variables, numeric_stddevs

def run_test_outcomes_with_prob_models(test, train, prob_models, prob_variables, prob_evaluation_metrics):
    p_hit_into_play_outcome_variables = prob_variables['p_hit_into_play_outcome_variables']
    p_swing_decision_model, p_swing_decision_variables = prob_models['p_swing_decision_model'], prob_variables['p_swing_decision_variables']
    p_no_swing_outcome_model, p_no_swing_outcome_variables = prob_models['p_no_swing_outcome_model'], prob_variables['p_no_swing_outcome_variables']
    p_swing_outcome_model, p_swing_outcome_variables = prob_models['p_swing_outcome_model'], prob_variables['p_swing_outcome_variables']
    p_hit_into_play_outcome_model, p_hit_into_play_outcome_variables = prob_models['p_hit_into_play_outcome_model'], prob_variables['p_hit_into_play_outcome_variables']
    
    test_all_fields = test[~((test.description=='hit_into_play') & test[p_hit_into_play_outcome_variables].isna().any(axis=1))].copy(deep=True)

    test_decision = p_swing_decision_model.predict_proba(test_all_fields[p_swing_decision_variables])[:,1]
    test_no_swing = pd.DataFrame(p_no_swing_outcome_model.predict_proba(test_all_fields[p_no_swing_outcome_variables]), columns = p_no_swing_outcome_model.classes_)
    test_swing = pd.DataFrame(p_swing_outcome_model.predict_proba(test_all_fields[p_swing_outcome_variables]), columns = p_swing_outcome_model.classes_)
    test_bip = pd.DataFrame(p_hit_into_play_outcome_model.predict_proba(test_all_fields[p_hit_into_play_outcome_variables].fillna(0)), columns = p_hit_into_play_outcome_model.classes_)
    
    temp1 = (1-test_decision)[:,np.newaxis] * test_no_swing
    temp2 = test_decision[:,np.newaxis] * test_swing
    temp3 = temp2['hit_into_play'].values[:,np.newaxis] * test_bip
    
    test_predictions = temp1[['ball','hit_by_pitch']].join(temp2[['foul']]).join(temp3)
    test_predictions['strike'] = temp1['strike'] + temp2['strike']
    
    baseline = train['pitch_outcome'].value_counts(normalize=True).reindex(test_predictions.columns)
    test_baseline = pd.DataFrame([baseline]*len(test_all_fields['pitch_outcome'])).reset_index(drop=True)
    
    smart_baselines = train.groupby(['pitch_name','zone'])['pitch_outcome'].value_counts(normalize=True).reset_index().pivot(index=['pitch_name','zone'], columns = 'pitch_outcome', values = 'proportion').reindex(columns = test_predictions.columns).fillna(0).reset_index()
    test_smart_baseline = test_all_fields[['pitch_name','zone']].merge(smart_baselines, on = ['pitch_name','zone'],how='left')[list(test_predictions.columns)]
    test_smart_baseline[test_smart_baseline[list(test_predictions.columns)].sum(axis=1)==0] = baseline
    
    test_predictions_binarize = label_binarize(test_all_fields.pitch_outcome, classes = test_predictions.columns)
    test_evaluation_metrics = {
        'test_log_loss': log_loss(test_predictions_binarize, test_predictions),
        'test_auprc': average_precision_score(test_predictions_binarize, test_predictions),
        'test_auprc_by_class': average_precision_score(test_predictions_binarize, test_predictions, average=None),
        
        'test_baseline_log_loss': log_loss(test_predictions_binarize, test_baseline),
        'test_baseline_auprc': average_precision_score(test_predictions_binarize, test_baseline),
        'test_baseline_auprc_by_class': average_precision_score(test_predictions_binarize, test_baseline, average=None),
    
        'test_smart_baseline_log_loss': log_loss(test_predictions_binarize, test_smart_baseline),
        'test_smart_baseline_auprc': average_precision_score(test_predictions_binarize, test_smart_baseline),
        'test_smart_baseline_auprc_by_class': average_precision_score(test_predictions_binarize, test_smart_baseline, average=None),
        }
    
    p_swing_decision_evaluation_metrics = prob_evaluation_metrics['p_swing_decision_evaluation_metrics']
    p_no_swing_outcome_evaluation_metrics = prob_evaluation_metrics['p_no_swing_outcome_evaluation_metrics']
    p_swing_outcome_evaluation_metrics = prob_evaluation_metrics['p_swing_outcome_evaluation_metrics']
    p_hit_into_play_outcome_evaluation_metrics = prob_evaluation_metrics['p_hit_into_play_outcome_evaluation_metrics']
    
    sl_summary = pd.DataFrame.from_dict([p_swing_decision_evaluation_metrics, p_no_swing_outcome_evaluation_metrics, p_swing_outcome_evaluation_metrics, p_hit_into_play_outcome_evaluation_metrics, test_evaluation_metrics])
    test_dict = {
         'test_predictions': test_predictions,
         'test_baseline':test_baseline,
         'test_smart_baseline': test_smart_baseline,
         'test_predictions_binarize': test_predictions_binarize,
         'test_evaluation_metrics':test_evaluation_metrics,
         'prob_sl_summary': sl_summary
         }
    return test_dict

def random_numeric(model, variables, row, std_devs, pitch_name, zone):
    if type(row)==dict:
        xs = pd.DataFrame(row,index=[0])
    else:
        xs = row[variables]
    model_output =model.predict(xs)[0]
    model_std_dev = std_devs[(pitch_name, zone)]
    return float(np.random.normal(model_output, model_std_dev))

def simulate_pitch(state, pitch_name, zone, prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs):
    pitch = {
        'pitch_name': pitch_name,
        'zone': zone,
        'pitch_zone': str(pitch_name)+'_'+str(zone),
        'outs_when_up': state.outs_when_up,
        'runner1':state.runner1,
        'runner2':state.runner2,
        'runner3':state.runner3,
        'balls':state.balls,
        'strikes':state.strikes,
        'stance_vs_arm_same':state.stance_vs_arm_same,
        'sz_top':state.sz_top,
        'sz_bot':state.sz_bot
        }
    pitch['release_speed'] = random_numeric(
        numeric_models['release_speed_model'], 
        numeric_variables['release_speed_variables'], 
        pitch, 
        numeric_stddevs['release_speed_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['api_break_z_with_gravity'] = random_numeric(
        numeric_models['api_break_z_with_gravity_model'], 
        numeric_variables['api_break_z_with_gravity_variables'], 
        pitch, 
        numeric_stddevs['api_break_z_with_gravity_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['api_break_x_batter_in'] = random_numeric(
        numeric_models['api_break_x_batter_in_model'], 
        numeric_variables['api_break_x_batter_in_variables'], 
        pitch, 
        numeric_stddevs['api_break_x_batter_in_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['plate_x'] = random_numeric(
        numeric_models['plate_x_model'], 
        numeric_variables['plate_x_variables'], 
        pitch, 
        numeric_stddevs['plate_x_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['plate_z'] = random_numeric(
        numeric_models['plate_z_model'], 
        numeric_variables['plate_z_variables'], 
        pitch, 
        numeric_stddevs['plate_z_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['swing_length'] = random_numeric(
        numeric_models['swing_length_model'], 
        numeric_variables['swing_length_variables'], 
        pitch, 
        numeric_stddevs['swing_length_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['bat_speed'] = random_numeric(
        numeric_models['bat_speed_model'], 
        numeric_variables['bat_speed_variables'], 
        pitch, 
        numeric_stddevs['bat_speed_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['launch_angle'] = random_numeric(
        numeric_models['launch_angle_model'], 
        numeric_variables['launch_angle_variables'], 
        pitch, 
        numeric_stddevs['launch_angle_stddevs'], 
        pitch_name, 
        zone)
    
    pitch['launch_speed'] = random_numeric(
        numeric_models['launch_speed_model'], 
        numeric_variables['launch_speed_variables'], 
        pitch, 
        numeric_stddevs['launch_speed_stddevs'], 
        pitch_name, 
        zone)
    
    sz_mid = (pitch['sz_top']+pitch['sz_bot'])/2.0
    pitch['pitch_vs_sz_mid'] = np.sqrt((pitch['plate_z']-sz_mid)**2+pitch['plate_x']**2)
    
    
    p_swing_decision = prob_models['p_swing_decision_model'].predict_proba(pd.DataFrame(pitch,index=[0])[prob_variables['p_swing_decision_variables']])[0,1]
    if np.random.rand() > p_swing_decision:
        p_no_swing_outcome_value = prob_models['p_no_swing_outcome_model'].predict_proba(pd.DataFrame(pitch,index=[0])[prob_variables['p_no_swing_outcome_variables']])
        outcome = np.random.choice(prob_models['p_no_swing_outcome_model'].classes_, p=p_no_swing_outcome_value[0])
    else:
        p_swing_outcome_value = prob_models['p_swing_outcome_model'].predict_proba(pd.DataFrame(pitch,index=[0])[prob_variables['p_swing_outcome_variables']])
        swing_outcome = np.random.choice(prob_models['p_swing_outcome_model'].classes_, p=p_swing_outcome_value[0])
        if swing_outcome != 'hit_into_play':
            outcome = swing_outcome
        else:
            p_hit_into_play_outcome_value = prob_models['p_hit_into_play_outcome_model'].predict_proba(pd.DataFrame(pitch,index=[0])[prob_variables['p_hit_into_play_outcome_variables']])
            outcome = np.random.choice(prob_models['p_hit_into_play_outcome_model'].classes_, p=p_hit_into_play_outcome_value[0])
    if (state.runner1+state.runner2+state.runner3==0 or state.outs_when_up == 2) and outcome in ['double_play','triple_play','sacrifice']: 
        pitch['pitch_outcome'] = 'out'
    elif state.runner1+state.runner2+state.runner3==1 and outcome in ['triple_play']: 
        if state.outs_when_up <= 1:
            pitch['pitch_outcome'] = 'double_play'
        else:
            pitch['pitch_outcome'] = 'out'
    else:
        pitch['pitch_outcome'] = outcome
    return pitch
            
def update_state(pitch, old_state):
    state = old_state.copy()
    end_ab = False
    end_inning = False
    runs_scored = 0
    outcome = pitch['pitch_outcome']
    if outcome == 'ball':
        state.balls+=1
        if state.balls == 4:
            end_ab = True
            if state.runner1 == 0:
                state.runner1 = 1
            elif state.runner1 == 1 and state.runner2 == 0: 
                state.runner2 = 1
            elif state.runner1 == 1 and state.runner2 == 1 and state.runner3 == 0: 
                state.runner3 = 1
            elif state.runner1 == 1 and state.runner2 == 1 and state.runner3 == 1: 
                runs_scored += 1
    elif outcome in ['catcher_interf', 'error', 'single']:
        end_ab = True
        if state.runner3 == 1:
            runs_scored += 1
            state.runner3 = 0
        if state.runner2 == 1:
            state.runner3 = 1
            state.runner2 = 0
        if state.runner1 == 1:
            state.runner2 = 1
            state.runner1 = 0
        state.runner1 = 1
    elif outcome in ['double']:
        end_ab = True
        if state.runner3 == 1:
            runs_scored += 1
            state.runner3 = 0
        if state.runner2 == 1:
            runs_scored += 1
            state.runner2 = 0
        if state.runner1 == 1:
            state.runner3 = 1
            state.runner1 = 0
        state.runner2 = 1
    elif outcome in ['double_play']:
        end_ab = True
        if state.outs_when_up >= 1:
            state.outs_when_up = 3
        elif state.runner1 == 1 and state.runner2 == 1 and state.runner3 == 1:
            state.outs_when_up += 2
            state.runner3 = 0
        elif state.runner1 == 1 and state.runner2 == 1 and state.runner3 == 0:
            state.outs_when_up += 2
            state.runner2 = 0
        elif state.runner1 == 1 and state.runner2 == 0 and state.runner3 == 1:
            state.outs_when_up += 2
            state.runner1 = 0
        elif state.runner1 == 1 and state.runner2 == 0 and state.runner3 == 0:
            state.outs_when_up += 2
            state.runner1 = 0
        elif state.runner1 == 0 and state.runner2 == 1 and state.runner3 == 1:
            state.outs_when_up += 2
            state.runner3 = 0
        elif state.runner1 == 0 and state.runner2 == 1 and state.runner3 == 0:
            state.outs_when_up += 2
            state.runner2 = 0
        elif state.runner1 == 0 and state.runner2 == 0 and state.runner3 == 1:
            state.outs_when_up += 2
            state.runner3 = 0
    elif outcome == 'foul':
        if state.strikes < 2:
            state.strikes +=1
    elif outcome == 'hit_by_pitch':
        end_ab = True
        if state.runner1 == 0:
            state.runner1 = 1
        elif state.runner1 == 1 and state.runner2 == 0: 
            state.runner2 = 1
        elif state.runner1 == 1 and state.runner2 == 1 and state.runner3 == 0: 
            state.runner3 = 1
        elif state.runner1 == 1 and state.runner2 == 1 and state.runner3 == 1: 
            runs_scored += 1
    elif outcome == 'home_run':
        runs_scored += 1 + state.runner1 + state.runner2 + state.runner3
        end_ab = True
        state.runner1 = 0
        state.runner2 = 0
        state.runner3 = 0
    elif outcome == 'out':
        end_ab = True
        state.outs_when_up += 1
    elif outcome in ['sacrifice']:
        end_ab = True
        state.outs_when_up += 1
        if state.runner3 == 1:
            runs_scored += 1
            state.runner3 = 0
        if state.runner2 == 1:
            state.runner3 = 1
            state.runner2 = 0
        if state.runner1 == 1:
            state.runner2 = 1
            state.runner1 = 0
        state.runner1 = 0
    elif outcome == 'strike':
        state.strikes += 1
        if state.strikes == 3:
            state.outs_when_up += 1
            state.strikes = 0
            state.balls = 0
            end_ab = True
    elif outcome == 'triple':
        runs_scored += state.runner1 + state.runner2 + state.runner3
        end_ab = True
        state.runner1 = 0
        state.runner2 = 0
        state.runner3 = 1
    elif outcome == 'triple_play':
        end_ab = True
        state.outs_when_up = 3
    if end_ab:
        state.balls = 0
        state.strikes = 0
    if state.outs_when_up==3:
        end_inning = True
    if end_inning:
        state.balls = 0
        state.strikes = 0
        state.runner1 = 0
        state.runner2 = 0
        state.runner3 = 0
        state.outs_when_up = 0
    return state, end_ab, end_inning, runs_scored, old_state
    
def calc_pitch_reward(pitch, runs_scored, old_state):
    run_reward = -6
    '''
    out_reward = 0
    event_reward ={
        'ball': 0.0,
        'catcher_interf':0,
        'double': 0.0,
        'double_play':2*out_reward,
        'error':0,
        'foul':0.0,
        'hit_by_pitch': 0,
        'home_run':0,
        'out':out_reward,
        'sacrifice': out_reward,
        'single': 0.0,
        'strike': 0.0,
        'triple': 0,
        'triple_play':3*out_reward
        }
    if old_state.balls==3:
        event_reward['ball'] = 0.0
    if old_state.strikes==2:
        event_reward['strike'] = out_reward
        event_reward['foul'] = 0
    '''
    return run_reward*runs_scored #+ event_reward[pitch['pitch_outcome']]

        
def get_eligible_actions():
    pitches = ['4-Seam Fastball', 'Sinker', 'Slider', 'Changeup', 'Cutter','Sweeper', 'Curveball']
    zones = [1,2,3,4,5,6,7,8,9,11,12,13,14]
    return [(p, z) for p in pitches for z in zones]

def get_eligible_states():
    balls = [0,1,2,3]
    strikes = [0,1,2]
    outs_when_up = [0,1,2]
    runner1 = [0,1]
    runner2 = [0,1]
    runner3 = [0,1]
    stance_vs_arm_same = [0,1]
    return [(b,s,o,r1,r2,r3,same) 
            for b in balls 
            for s in strikes
            for o in outs_when_up
            for r1 in runner1
            for r2 in runner2
            for r3 in runner3
            for same in stance_vs_arm_same
            ]

def get_initial_probs(actions, states):
    n = len(actions)
    probs = {}
    for s in states:
        probs[s] = np.ones(n) / float(n)
    return probs

def pick_action_from_state(state,actions, probs):
    ps = probs[(state.balls, state.strikes, state.outs_when_up, state.runner1,state.runner2,state.runner3,state.stance_vs_arm_same)]
    return actions[np.random.choice(len(actions), p=ps)]

def get_state_tuple(state):
    return (int(state.balls), 
            int(state.strikes), 
            int(state.outs_when_up), 
            int(state.runner1),
            int(state.runner2),
            int(state.runner3),
            int(state.stance_vs_arm_same))

def simulate_at_bat(starting_state, actions, probs, prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs):
    state = starting_state.copy()
    pitch_history = []
    end_ab = False
    while not end_ab:
        state_tuple = get_state_tuple(state)
        action = pick_action_from_state(state,actions, probs)
        pitch_name = action[0]
        zone = action[1]
        pitch = simulate_pitch(state, pitch_name, zone, prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs)
        state, end_ab, end_inning, runs_scored, old_state = update_state(pitch, state)
        reward = calc_pitch_reward(pitch, runs_scored, old_state)
        pitch['reward'] = reward
        pitch['action'] = action
        pitch['state_before_tuple'] = state_tuple
        pitch['state_after_tuple'] = get_state_tuple(state)
        pitch['state_after_pitch'] = state.copy()
        pitch['end_ab'] = end_ab
        pitch['end_inning'] = end_inning
        pitch['runs_scored'] = runs_scored
        pitch['state_before_pitch'] = old_state.copy()
        pitch_history.append(pitch)
        
    return pitch_history, end_inning, runs_scored, state, starting_state

def simulate_half_inning(actions, probs, gamma, prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, data):
    state = environment_state() 
    episode_history = []
    runs_scored_total = 0
    ab_number = 1
    end_inning = False
    while not end_inning:
        state.sz_top, state.sz_bot = sample_sz(data)
        state.stance_vs_arm_same = sample_stance_vs_arm_same(data)
        pitch_history, end_inning, runs_scored, state, starting_state = simulate_at_bat(state, actions, probs, 
                                                                                        prob_models, prob_variables, numeric_models, 
                                                                                        numeric_variables, numeric_stddevs)
        for pitch in pitch_history:
            pitch['ab_number'] = ab_number
        ab_number+=1
        runs_scored_total += runs_scored
        episode_history.extend(pitch_history)
    
    '''  
    pitch_reward_plus_discounted_future_reward = 0.0
    
    for p in reversed(episode_history):
        pitch_reward_plus_discounted_future_reward = p['reward'] + gamma * pitch_reward_plus_discounted_future_reward
        p['total_reward'] = pitch_reward_plus_discounted_future_reward
    '''
    
    return episode_history, runs_scored_total

def get_episode_stats(episode_history, runs_scored_total):
    stats = {
        'plate_appearances': 0,
        'catcher_interference': 0,
        'pitches': 0,
        'balls': 0,
        'strikes': 0,
        'at_bats': 0,
        'hits': 0,
        'singles': 0,
        'doubles': 0,
        'triples': 0,
        'home_runs': 0,
        'walks': 0,
        'hit_by_pitch': 0,
        'strikeouts': 0,
        'sacrifices': 0,
        'errors': 0,
        'double_plays': 0,
        'triple_plays': 0,
        'fouls': 0,
        'runs': 0
    }
    for pitch in episode_history:
        outcome = pitch['pitch_outcome']
        state = pitch['state_before_pitch']
        stats['pitches'] += 1
        if pitch['end_ab']:
            stats['plate_appearances'] += 1
        if outcome == 'catcher_interf':
            stats['catcher_interference'] += 1
            stats['strikes'] += 1
        elif outcome == 'ball':
            stats['balls'] += 1
            if state.balls == 3:
                stats['walks'] += 1
        elif outcome == 'double':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['hits'] += 1
            stats['doubles'] += 1
        elif outcome == 'double_play':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['double_plays'] += 1
        elif outcome == 'error':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['errors'] += 1
        elif outcome == 'foul':
            stats['strikes'] += 1
            stats['fouls'] += 1
        elif outcome == 'hit_by_pitch':
            stats['balls'] += 1
            stats['hit_by_pitch'] += 1
        elif outcome == 'home_run':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['hits'] += 1
            stats['home_runs'] += 1
        elif outcome == 'out':
            stats['strikes'] += 1
            stats['at_bats'] += 1
        elif outcome == 'sacrifice':
            stats['strikes'] += 1
            stats['sacrifices'] += 1
        elif outcome == 'single':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['hits'] += 1
            stats['singles'] += 1
        elif outcome == 'strike':
            stats['strikes'] += 1
            if state.strikes == 2:
                stats['at_bats'] += 1
                stats['strikeouts'] += 1
        elif outcome == 'triple':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['hits'] += 1
            stats['triples'] += 1
        elif outcome == 'triple_play':
            stats['strikes'] += 1
            stats['at_bats'] += 1
            stats['triple_plays'] += 1
    stats['runs'] += runs_scored_total
        
    return stats
                    
def execute_policy_in_mdp(n, actions, probs, gamma, prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, data):
    episodes = []
    episode_stats = []
    it_timer = time.time()
    
    for i in range(n):
        episode_history, runs_scored_total = simulate_half_inning(actions, probs, gamma, 
                                                                  prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, data)
        episodes.extend(episode_history)
        episode_stats.append(get_episode_stats(episode_history, runs_scored_total))
        if (i+1) % 27 == 0:
            print(f'---Iteration {i+1} complete, took {round(time.time() - it_timer,2)} seconds')
            it_timer = time.time()
    return episodes, episode_stats

def update_prob_sa_reward_sa_estimates(episodes):
    transition_counter = {}
    reward_sums = {}
    reward_counter = {}
    for pitch in episodes:
        state_key = pitch['state_before_tuple']
        action_key = pitch['action']
        state_apostrophe_key = pitch['state_after_tuple']
        immediate_reward = pitch['reward']
        sa = (state_key, action_key)
        if sa not in transition_counter:
            transition_counter[sa] = {}
            reward_sums[sa] = 0.0
            reward_counter[sa] = 0
        if state_apostrophe_key not in transition_counter[sa]:
            transition_counter[sa][state_apostrophe_key] = 0
            
        transition_counter[sa][state_apostrophe_key] += 1
        
        reward_sums[sa] += immediate_reward
        reward_counter[sa] += 1
        
    prob_estimates = {}
    for state_action_pair, s_apostrophe_counts in transition_counter.items():
        sa_counter = float(sum(s_apostrophe_counts.values()))
        prob_estimates[state_action_pair] = {}
        for s_apostrophe, s_apos_counter in s_apostrophe_counts.items():
            prob_estimates[state_action_pair][s_apostrophe] = s_apos_counter / sa_counter
            
    reward_sa_estimates = {}
    for state_action_pair in reward_sums:
        reward_sa_estimates[state_action_pair] = reward_sums[state_action_pair] / float(reward_counter[state_action_pair])
        
    return prob_estimates, reward_sa_estimates

def value_iteration(states, actions, prob_estimates, reward_sa_estimates, gamma, 
                    v_initial = None, max_iterations = 1000, threshold = 1e-4):
    if v_initial is None:
        V = {}
        for s in states:
            V[s] = 0.0
    else:
        V = v_initial.copy()
    for i in range(max_iterations):
        max_diff = 0.0
        V_update = {}
        optimal_actions={}
        for s in states:
            max_a_value = -np.inf
            max_a = None
            for a in actions:
                sa = (s,a)
                if sa in prob_estimates:
                    sum_pv_next_states = 0
                    for s_apostrophe, p in prob_estimates[sa].items():
                        sum_pv_next_states += p*V[s_apostrophe]
                else:
                    continue
                max_a_test =  reward_sa_estimates.get(sa,-np.inf) + gamma*sum_pv_next_states
                if max_a_test > max_a_value:
                    max_a_value = max_a_test
                    max_a = a
                 
            if max_a_value == -np.inf:
                max_a_value = 0.0
            
            V_update[s] = max_a_value
            optimal_actions[s] = max_a
            max_diff = max(max_diff, abs(V_update[s] - V[s]))
        V = V_update
        
        if max_diff < threshold:
            return V, optimal_actions
    
    print(f'Did not converge in {max_iterations} iterations')
    return V, optimal_actions
    
def update_policy(optimal_actions, actions, probs, alpha = 0.2):
    n = len(actions)
    new_probs = {}
    for s, old_prob_s in probs.items():
        a = optimal_actions.get(s,None)
        if a is None:
            new_probs[s] = old_prob_s.copy()
        else:
            new_prob_s_array = np.zeros(n)
            new_prob_s_array[actions.index(a)] = 1.0
            new_probs[s] = alpha*new_prob_s_array + (1-alpha)*old_prob_s
            new_probs[s] = new_probs[s] / np.sum(new_probs[s])
    return new_probs
    
def run_mdp_estimation(prob_models, prob_evaluation_metrics, prob_variables, 
        numeric_models, numeric_evaluation_metrics, numeric_variables, numeric_stddevs,
        data, alpha = 0.2, gamma = 0.95, max_iterations = 20, episodes_per_iteration = 162, tolerance = 1e-4):
    actions = get_eligible_actions()
    states = get_eligible_states()
    probs = get_initial_probs(actions, states)
    n = episodes_per_iteration
    V = None
    all_episodes = []
    all_stats = []
    total_time = time.time()
    for i in range(max_iterations):
        it_timer = time.time()
        episodes, episode_stats = execute_policy_in_mdp(n, actions, probs, gamma, prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, data)
        all_episodes.extend(episodes)
        all_stats.extend(episode_stats)
        prob_estimates, reward_sa_estimates = update_prob_sa_reward_sa_estimates(all_episodes)
        V, optimal_actions = value_iteration(states, actions, prob_estimates, reward_sa_estimates, gamma, v_initial = V)
        new_probs = update_policy(optimal_actions, actions, probs)
    
        max_diff = -np.inf
        for s in states:
            diff = np.max(np.abs(new_probs[s] - probs[s]))
            if diff > max_diff:
                max_diff = diff
        probs = new_probs
        print(f'Iteration {i+1} of {max_iterations} complete, took {round(time.time() - it_timer,2)} seconds')
        print(f'---New max_diff is {round(max_diff,5)}')
    if max_diff < tolerance:
        print(f'Converged after {max_iterations} iterations, took {round(time.time() - total_time,2)} seconds')
        return probs, V, all_episodes, all_stats
    print(f'Did not converge after {max_iterations} iterations, took {round(time.time() - total_time,2)} seconds')
    return probs, V, all_episodes, all_stats

train, validate, test, data = get_data()

prob_models, prob_evaluation_metrics, prob_variables = run_prob_models(train, validate)
numeric_models, numeric_evaluation_metrics, numeric_variables, numeric_stddevs = run_numeric_models(train, validate)
prob_test_outcomes = run_test_outcomes_with_prob_models(test, train, prob_models, prob_variables, prob_evaluation_metrics)

probs, V, all_episodes, all_stats = run_mdp_estimation(prob_models, prob_evaluation_metrics, prob_variables, numeric_models, numeric_evaluation_metrics, numeric_variables, numeric_stddevs, train)
episodes_df = pd.DataFrame(all_episodes)
all_stats_df = pd.DataFrame(all_stats)
V_df = pd.DataFrame([V]).T.rename({0:'V'})
probs_df = pd.DataFrame.from_dict(probs, orient='index')
probs_df.index.name = 'state'
probs_df.columns = [f'a_{i}' for i in range(probs_df.shape[1])]

excel_file_path = 'C:/Users/pthar/OneDrive/Documents/cs229/project/output/runs_only.xlsx'

with pd.ExcelWriter(excel_file_path) as writer:
    episodes_df.to_excel(writer, sheet_name='episodes', index=True)
    all_stats_df.to_excel(writer, sheet_name='stats', index=True)
    V_df.to_excel(writer, sheet_name='v', index=False)
    probs_df.reset_index().to_excel(writer, sheet_name='probs', index=False)


def load_probs_again(file_path):
    probs_df = pd.read_excel(file_path, sheet_name = 'probs')
    if 'states' in list(probs_df):
        probs_df.set_index('state')
    probs_array = probs_df.to_numpy()
    states = get_eligible_states()
    actions = get_eligible_actions()
    if probs_array.shape[0] != len(states):
        raise ValueError("States dimension doesn't match")
    if probs_array.shape[1] != len(actions):
        raise ValueError("Actions dimension doesn't match")
    probs = {}
    for i,s in enumerate(states):
        probs[s] = probs_array[i,:]
    return probs


baseline_run_probs = load_probs_again('C:/Users/pthar/OneDrive/Documents/cs229/project/output/baseline_run.xlsx')
outs_only_probs = load_probs_again('C:/Users/pthar/OneDrive/Documents/cs229/project/output/outs_only.xlsx')
runs_only_probs = load_probs_again('C:/Users/pthar/OneDrive/Documents/cs229/project/output/runs_only.xlsx')
limit_big_innings_probs = load_probs_again('C:/Users/pthar/OneDrive/Documents/cs229/project/output/limit_big_innings.xlsx')

gamma = 0.95
actions = get_eligible_actions()
states = get_eligible_states()
start_sim = time.time()
baseline_episodes, baseline_episode_stats = execute_policy_in_mdp(1800, actions, baseline_run_probs, gamma, 
                                                                  prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, train)

outs_only_episodes, outs_only_episode_stats = execute_policy_in_mdp(1800, actions, outs_only_probs, gamma, 
                                                                  prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, train)

runs_only_episodes, runs_only_episode_stats = execute_policy_in_mdp(900, actions, runs_only_probs, gamma, 
                                                                  prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, train)

limit_big_innings_episodes, limit_big_innings_episode_stats = execute_policy_in_mdp(1800, actions, limit_big_innings_probs, gamma, 
                                                                  prob_models, prob_variables, numeric_models, numeric_variables, numeric_stddevs, train)

print(f'Sim took{round(time.time()-start_sim,0)} seconds')

excel_file_path = 'C:/Users/pthar/OneDrive/Documents/cs229/project/output/rl_eval.xlsx'

with pd.ExcelWriter(excel_file_path) as writer:
    pd.DataFrame(baseline_episode_stats).to_excel(writer, sheet_name='baseline_episode_stats', index=True)
    pd.DataFrame(outs_only_episode_stats).to_excel(writer, sheet_name='outs_only_episode_stats', index=True)
    pd.DataFrame(runs_only_episode_stats).to_excel(writer, sheet_name='runs_only_episode_stats', index=True)
    pd.DataFrame(limit_big_innings_episode_stats).to_excel(writer, sheet_name='limit_big_innings_episode_stats', index=True)
    pd.DataFrame(baseline_episodes).to_excel(writer, sheet_name='baseline_episodes', index=True)
    pd.DataFrame(outs_only_episodes).to_excel(writer, sheet_name='outs_only_episodes', index=True)
    pd.DataFrame(runs_only_episodes).to_excel(writer, sheet_name='runs_only_episodes', index=True)
    pd.DataFrame(limit_big_innings_episodes).to_excel(writer, sheet_name='limit_big_innings_episodes', index=True)
