import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math as mt
from sklearn.model_selection import RandomizedSearchCV


def nulls_percent (df):
    
    '''
    Shows percent of nulls in a data frame.
    
    Args:
        df: The dataframe we want to check out.
        
    Returns:
        A new df with 2columns:
        - 'column_name' with the name of the original df columns
        - 'nulls_percentage' with the percentage of nulls in every column
    '''
    nulls_percent = pd.DataFrame(df.isna().sum()/len(df)).reset_index()
    nulls_percent.columns = ['column_name', 'nulls_percentage']
    
    return nulls_percent


def random_grid_multimodel (list_of_models, random_grids, X_train, X_test, y_train, scoring, folds = 5, iters = 5):
    """
    By providing a list of different models and a dictionary of hyperparameter grids defined per each model,
    it searches for the best model of each kind and trains it.
    The model is selected by using a Randomized CrossValidation Search method. 
    
    """

    best_models = { elem[1]: {'model': None, 'best_params': None, 'Preds_path': None, 'Metrics': None} for elem in list_of_models }

    for i in range( len(list_of_models)):
        if list_of_models[i][1] == "random_forest" :

            random_search_rf = RandomizedSearchCV(estimator = list_of_models[i][0], param_distributions = random_grids['param_grid_rf'],
                                cv=folds, n_iter=iters, n_jobs = 3)

            random_search_rf.fit(X_train, y_train.ravel())

            # make the prediction
            y_pred_train_rf = random_search_rf.predict(X_train)
            y_pred_test_rf  = random_search_rf.predict(X_test)

            print('Random Forest best parameters are:')
            print(random_search_rf.best_params_)

            best_model_rf = random_search_rf.best_estimator_

            filename = "models/" + list_of_models[i][1] + ".pkl"
            with open(filename, "wb") as file:
                pickle.dump(best_model_rf,file)

            best_models["random_forest" ]['model'] = best_model_rf
            best_models["random_forest" ]['best_params'] = random_search_rf.best_params_


        elif list_of_models[i][1] == "knn":

            random_search_knn = RandomizedSearchCV(estimator = list_of_models[i][0], param_distributions = random_grids['param_grid_knn'],
                                cv=folds, n_iter=iters, n_jobs = 3)

            random_search_knn.fit(X_train, y_train.ravel())

            # make the prediction
            y_pred_train_knn = random_search_knn.predict(X_train)
            y_pred_test_knn = random_search_knn.predict(X_test)

            print('KNearest Neighbors best parameters are:')
            print(random_search_knn.best_params_)

            best_model_knn = random_search_knn.best_estimator_

            filename = "models/" + list_of_models[i][1] + ".pkl"
            with open(filename, "wb") as file:
                pickle.dump(best_model_knn,file)

            best_models["knn"]['model'] = best_model_knn
            best_models["knn"]['best_params'] = random_search_knn.best_params_


        elif list_of_models[i][1] == "decision_tree":

            random_search_dt = RandomizedSearchCV(estimator = list_of_models[i][0], param_distributions = random_grids['param_grid_dt'],
                                cv=folds, n_iter=iters, n_jobs = 3)

            random_search_dt.fit(X_train, y_train.ravel())

            # make the prediction
            y_pred_train_dt = random_search_dt.predict(X_train)
            y_pred_test_dt = random_search_dt.predict(X_test)

            print('Decision Tree best parameters are:')
            print(random_search_dt.best_params_)

            best_model_dt = random_search_dt.best_estimator_

            filename = "models/" + list_of_models[i][1] + ".pkl"
            with open(filename, "wb") as file:
                pickle.dump(best_model_dt,file)

            best_models["decision_tree" ]['model'] = best_model_dt
            best_models["decision_tree" ]['best_params'] = random_search_dt.best_params_

    best_models_safe = best_models
    
    filename = "models/best_models.pkl"
    pickle.dump(best_models_safe, open(filename, 'wb'))

    return best_models


def get_model_predictions(best_models, X_train, y_train, X_test, y_test):

	#Loading the Transformer from the pickle file:
	y_train = np.array(y_train).reshape(-1,1)
	y_test = np.array(y_test).reshape(-1,1)

	for key in list(best_models.keys()):

		preds = pd.DataFrame({'Set': ['Train']*len(X_train) + ['Test']*len(X_test),'Real': y_train.flatten().tolist() + y_test.flatten().tolist()})
		y_pred_train= best_models[key]['model'].predict(X_train).reshape(-1,1)
		y_pred_test = best_models[key]['model'].predict(X_test).reshape(-1,1)

		results = y_pred_train.flatten().tolist() + y_pred_test.flatten().tolist()

		preds[f"y_pred_{key}"] = results

	filename = f"results/{key}.csv"
	preds.to_csv(filename, index = False)
	best_models[key]['Preds_path'] = filename

	return best_models



def eval_models_predictions(best_models):

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    for key in list(best_models.keys()):

        metrics = pd.DataFrame({'Error_metric': ['ME','MAE','MSE','RMSE','R2'], 'Train': [0,0,0,0,0], 'Test': [0,0,0,0,0]})

        filename = best_models[key]['Preds_path']
        preds = pd.read_csv(filename)

        real_train  = preds[preds['Set'] == 'Train']['Real']
        pred_train  = preds[preds['Set'] == 'Train'][f"y_pred_{key}"]

        real_test   = preds[preds['Set'] == 'Test']['Real']
        pred_test   = preds[preds['Set'] == 'Test'][f"y_pred_{key}"]

        ME_train  = round(np.mean(real_train - pred_train),2)
        ME_test   = round(np.mean(real_test - pred_test),2)

        MAE_train = round(mean_absolute_error(real_train, pred_train),2)
        MAE_test  = round(mean_absolute_error(real_test, pred_test),2)

        MSE_train = round(mean_squared_error(real_train, pred_train),2)
        MSE_test  = round(mean_squared_error(real_test, pred_test),2)

        RMSE_train = round(np.sqrt(MSE_train),2)
        RMSE_test  = round(np.sqrt(MSE_test),2)

        #MAPE_train = round(np.mean(np.abs(real_train - pred_train)/real_train),2)
        #MAPE_test  = round(np.mean(np.abs(real_test - pred_test)/real_test),2)

        R2_train = round(r2_score(real_train, pred_train),2)
        R2_test  = round(r2_score(real_test, pred_test),2)

        metrics.iloc[0,1] = ME_train
        metrics.iloc[0,2] = ME_test

        metrics.iloc[1,1] = MAE_train
        metrics.iloc[1,2] = MAE_test

        metrics.iloc[2,1] = MSE_train
        metrics.iloc[2,2]  = MSE_test

        metrics.iloc[3,1] = RMSE_train
        metrics.iloc[3,2]  = RMSE_test

        metrics.iloc[4,1] = R2_train
        metrics.iloc[4,2]  = R2_test

        df = metrics.copy()

        del metrics

        best_models[key]['Metrics'] = df

    return best_models


def lr_perf_plots_multimodel(df_train, df_test, model = None):

    '''
    Provides a scatter plot combined with a lineplot to visually assess
    the performance of your model

    '''

    fig2, ax2 = plt.subplots(2,2, figsize=(16,8))

    sns.scatterplot(data = df_train, y = "y_pred_" + model, x= "Real", ax = ax2[0,0])
    sns.lineplot(data = df_train, x = 'Real', y = 'Real', color = 'black', ax = ax2[0,0])
    sns.histplot(df_train['Real'] - df_train["y_pred_" + model], ax = ax2[0,1])

    sns.scatterplot(data = df_test,y = "y_pred_" + model, x= "Real", ax = ax2[1,0])
    sns.lineplot(data = df_test, x = 'Real', y = 'Real', color = 'black', ax = ax2[1,0])
    sns.histplot(df_test['Real'] - df_test["y_pred_" + model], ax = ax2[1,1])

    plt.show()

def reg_performance (y_train, y_pred_train, y_test, y_pred_test):
    
    '''
    Measures the performance of a single Regression Model with y transformed.
    '''

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    ME_train = round(np.mean(y_train.values - y_pred_train),2)
    ME_test  = round(np.mean(y_test.values - y_pred_test),2)

    MAE_train = round(mean_absolute_error(y_train.values,y_pred_train),2)
    MAE_test  = round(mean_absolute_error(y_test.values,y_pred_test),2)

    MSE_train = round(mean_squared_error(y_train.values,y_pred_train),2)
    MSE_test  = round(mean_squared_error(y_test.values,y_pred_test),2)

    RMSE_train = round(np.sqrt(MSE_train),2)
    RMSE_test  = round(np.sqrt(MSE_test),2)

    MAPE_train = round(np.mean((np.abs(y_train.values-y_pred_train) / y_train.values)* 100.),2)
    MAPE_test  = round(np.mean((np.abs(y_test.values-y_pred_test) / y_test.values)* 100.),2)

    R2_train = round(r2_score(y_train.values,y_pred_train),2)
    R2_test  = round(r2_score(y_test.values,y_pred_test),2)


    performance = pd.DataFrame({'Error_metric': ['Mean error','Mean absolute error','Mean squared error',
                                             'Root mean squared error','Mean absolute percentual error',
                                             'R2'],
                            'Train': [ME_train, MAE_train, MSE_train, RMSE_train, MAPE_train, R2_train],
                            'Test' : [ME_test, MAE_test , MSE_test, RMSE_test, MAPE_test, R2_test]})

    display(performance)
    
    print('REAL vs PREDICTED PERFORMANCE')
    print('------------------------------')
    #Creating a DataFrame to show differences between predicted and Real values on Train Set:
    df_train = pd.DataFrame()
    df_train['Real_train'] = y_train
    df_train['Pred_train'] = y_pred_train

    #Creating a DataFrame differences between predicted and Real values on Test Set:
    df_test = pd.DataFrame()
    df_test['Real_test'] = y_test
    df_test['Pred_test'] = y_pred_test

    display(df_train.head())
    display(df_test.head())
    
    return performance, df_train, df_test


def lr_perf_plots(df_train, df_test):

    '''
    Provides a scatter plot combined with a lineplot to visually assess
    the performance of your model

    '''
    
    fig2, ax2 = plt.subplots(2,2, figsize=(16,8))

    sns.scatterplot(y = df_train['Pred_train'], x=df_train['Real_train'], ax = ax2[0,0])
    sns.lineplot(data = df_train, x = 'Real_train', y = 'Real_train', color = 'black', ax = ax2[0,0])
    sns.histplot(df_train['Real_train'] - df_train['Pred_train'], ax = ax2[0,1])

    sns.scatterplot(y = df_test['Pred_test'], x=df_test['Real_test'], ax = ax2[1,0])
    sns.lineplot(data = df_test, x = 'Real_test', y = 'Real_test', color = 'black', ax = ax2[1,0])
    sns.histplot(df_test['Real_test'] - df_test['Pred_test'], ax = ax2[1,1])
    
    plt.show()


def random_search_model_single (model, X_train, y_train, param_distributions, score, iters = 10, folds = 5):
    
    if type(model) == tuple:
        random_search = RandomizedSearchCV(model[0], param_distributions, n_iter= iters, scoring = score, n_jobs=1, refit=True, cv= folds, verbose=2, random_state=1)
        random_search.fit(X_train, y_train.ravel())
        print("Best MAE of %f is using %s" % (random_search.best_score_, random_search.best_params_))

        filename = "models/best_" + model[1]+".pkl"
        with open(filename, "wb") as file:
            pickle.dump(random_search, file)
        
        return random_search
        
    else: 
        random_search = RandomizedSearchCV(model, param_distributions, n_iter= iters, scoring = score, n_jobs=1, refit=True, cv= folds, verbose=2, random_state=1)
        random_search.fit(X_train, y_train.ravel())
        print("Best MAE of %f is using %s" % (random_search.best_score_, random_search.best_params_))

        filename = "models/best_" + model+".pkl"
        with open(filename, "wb") as file:
            pickle.dump(random_search, file)

        return random_search

