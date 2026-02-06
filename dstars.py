import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy
from joblib import Parallel, delayed

class DSTARS(BaseEstimator, RegressorMixin):
    """
    Deep Structure for Tracking Asynchronous Regressor Stacking (DSTARS)
    
    Parameters:
    -----------
    base_estimator : estimator object, default=None
        The base estimator to be used for each target in each layer.
        If None, defaults to RandomForestRegressor.
    epsilon : float, default=1e-4
        Minimum expected value of error decrease when adding a new regressor layer.
    phi : float, default=0.4
        Threshold for selecting regressor layers that contributed in at least phi percent of time.
        (Used in the cross-validation version DSTARST).
    n_folds_tracking : int, default=10
        Number of folds for internal cross-validation to determine the best number of layers.
    method : str, default='DSTARST'
        'DSTARS' for bootstrap/OOB version, 'DSTARST' for cross-validation version.
    random_state : int, default=None
        Random state for reproducibility.
    """
    def __init__(self, base_estimator=None, epsilon=1e-4, phi=0.4, n_folds_tracking=10, method='DSTARST', random_state=None, n_jobs=1):
        self.base_estimator = base_estimator if base_estimator is not None else RandomForestRegressor(n_estimators=100, random_state=random_state)
        self.epsilon = epsilon
        self.phi = phi
        self.n_folds_tracking = n_folds_tracking
        self.method = method
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.convergence_layers_ = None
        self.rf_importance_ = None
        self.models_ = {}
        self.targets_ = None
        self.n_targets_ = 0
        self.convergence_tracking_ = None

    def _get_rmse(self, y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
            
        if isinstance(y, pd.DataFrame):
            self.targets_ = y.columns.tolist()
            y = y.values
        else:
            self.targets_ = [f"target_{i}" for i in range(y.shape[1])]
            
        self.n_targets_ = len(self.targets_)
        n_samples = X.shape[0]

        if self.method == 'DSTARS':
            self._fit_dstars(X, y)
        else:
            self._fit_dstarst(X, y)
            
        return self

    def _fit_dstars(self, X, y):
        # Bootstrap sample
        rng = np.random.RandomState(self.random_state)
        bootstrap_idx = rng.choice(X.shape[0], X.shape[0], replace=True)
        oob_idx = np.array(list(set(range(X.shape[0])) - set(bootstrap_idx)))
        
        X_train, y_train = X[bootstrap_idx], y[bootstrap_idx]
        X_val, y_val = X[oob_idx], y[oob_idx]
        
        # Layer 0: ST models
        preds_train = np.zeros((X_train.shape[0], self.n_targets_))
        preds_val = np.zeros((X_val.shape[0], self.n_targets_))
        
        def _train_predict_single_target(i, X_train_fold, y_train_fold, X_val_fold):
            model = clone(self.base_estimator)
            model.fit(X_train_fold, y_train_fold[:, i])
            return model.predict(X_train_fold), model.predict(X_val_fold), model

        results = Parallel(n_jobs=self.n_jobs)(delayed(_train_predict_single_target)(i, X_train, y_train, X_val) for i in range(self.n_targets_))
        
        for i, (p_train, p_val, model) in enumerate(results):
            preds_train[:, i] = p_train
            preds_val[:, i] = p_val
            self.models_[i] = model # Store models for final prediction
            
        # RF Importance
        self.rf_importance_ = []
        def _calculate_rf_importance(i, preds_val_fold, y_val_fold):
            rf_aux = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf_aux.fit(preds_val_fold, y_val_fold[:, i])
            return rf_aux.feature_importances_ > 0

        self.rf_importance_ = Parallel(n_jobs=self.n_jobs)(delayed(_calculate_rf_importance)(i, preds_val, y_val) for i in range(self.n_targets_))
            
        # Tracking
        converged = np.array([all(~self.rf_importance_[i][np.arange(self.n_targets_) != i]) for i in range(self.n_targets_)])
        error_val = np.array([self._get_rmse(y_val[:, i], preds_val[:, i]) for i in range(self.n_targets_)])
        convergence_layers = np.zeros(self.n_targets_, dtype=int)
        
        all_preds_train = {0: preds_train}
        all_preds_val = {0: preds_val}
        
        rlayer = 1
        while not np.all(converged):
            new_preds_train = np.zeros((X_train.shape[0], self.n_targets_))
            new_preds_val = np.zeros((X_val.shape[0], self.n_targets_))
            
            for i in range(self.n_targets_):
                if not converged[i]:
                    chosen_targets = np.where(self.rf_importance_[i])[0]
                    
                    # Features + predictions of chosen targets from their respective current best layers
                    X_tra_ext = np.hstack([X_train, all_preds_train[0][:, chosen_targets]]) # Simplified: using layer 0 for dependencies in tracking
                    # In R code: tck.tra[,(chosen.t) := predictions.training[, paste(convergence.layers[chosen.t], chosen.t,sep="."), with = FALSE]]
                    # So it uses the latest predictions for each chosen target.
                    
                    # Let's refine this to match R logic exactly
                    X_tra_ext = X_train.copy()
                    X_val_ext = X_val.copy()
                    for ct in chosen_targets:
                        dep_layer = convergence_layers[ct]
                        X_tra_ext = np.column_stack([X_tra_ext, all_preds_train[dep_layer][:, ct]])
                        X_val_ext = np.column_stack([X_val_ext, all_preds_val[dep_layer][:, ct]])
                    
                    model = clone(self.base_estimator)
                    model.fit(X_tra_ext, y_train[:, i])
                    p_train = model.predict(X_tra_ext)
                    p_val = model.predict(X_val_ext)
                    
                    rmse_val = self._get_rmse(y_val[:, i], p_val)
                    if rmse_val + self.epsilon > error_val[i]:
                        converged[i] = True
                    else:
                        error_val[i] = rmse_val
                        new_preds_train[:, i] = p_train
                        new_preds_val[:, i] = p_val
                else:
                    # Keep old predictions if converged
                    new_preds_train[:, i] = all_preds_train[rlayer-1][:, i]
                    new_preds_val[:, i] = all_preds_val[rlayer-1][:, i]
            
            all_preds_train[rlayer] = new_preds_train
            all_preds_val[rlayer] = new_preds_val
            
            for i in range(self.n_targets_):
                if not converged[i] and np.any(new_preds_train[:, i] != 0):
                    convergence_layers[i] = rlayer
                    
            rlayer += 1
            if rlayer > 100: # Safety break
                break
                
        self.convergence_layers_ = convergence_layers
        self._final_modelling(X, y)

    def _fit_dstarst(self, X, y):
        kf = KFold(n_splits=self.n_folds_tracking, shuffle=True, random_state=self.random_state)
        
        # Initial ST predictions for RF Importance
        st_preds = np.zeros_like(y)

        def _train_predict_fold(fold_train_idx, fold_val_idx, X_full, y_full, target_idx):
            X_t, y_t = X_full[fold_train_idx], y_full[fold_train_idx]
            X_v = X_full[fold_val_idx]
            model = clone(self.base_estimator)
            model.fit(X_t, y_t[:, target_idx])
            return fold_val_idx, target_idx, model.predict(X_v)

        results = Parallel(n_jobs=self.n_jobs)(delayed(_train_predict_fold)(
            train_idx, val_idx, X, y, i
        ) for train_idx, val_idx in kf.split(X) for i in range(self.n_targets_))

        for val_idx, target_idx, predictions in results:
            st_preds[val_idx, target_idx] = predictions
                
        # RF Importance
        def _calculate_rf_importance(i, st_preds_fold, y_fold):
            rf_aux = RandomForestRegressor(n_estimators=100, random_state=self.random_state)
            rf_aux.fit(st_preds_fold, y_fold[:, i])
            return rf_aux.feature_importances_ > 0

        self.rf_importance_ = Parallel(n_jobs=self.n_jobs)(delayed(_calculate_rf_importance)(i, st_preds, y) for i in range(self.n_targets_))
            
        # Tracking with CV
        fold_convergence_layers = np.zeros((self.n_folds_tracking, self.n_targets_), dtype=int)
        
        fold_idx = 0
        for train_idx, val_idx in kf.split(X):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Layer 0
            preds_train = np.zeros((X_train.shape[0], self.n_targets_))
            preds_val = np.zeros((X_val.shape[0], self.n_targets_))

            def _train_predict_layer0(i, X_train_fold, y_train_fold, X_val_fold):
                model = clone(self.base_estimator)
                model.fit(X_train_fold, y_train_fold[:, i])
                return model.predict(X_train_fold), model.predict(X_val_fold)

            results_layer0 = Parallel(n_jobs=self.n_jobs)(delayed(_train_predict_layer0)(i, X_train, y_train, X_val) for i in range(self.n_targets_))
            for i, (p_train, p_val) in enumerate(results_layer0):
                preds_train[:, i] = p_train
                preds_val[:, i] = p_val
            
            error_val = np.array([self._get_rmse(y_val[:, i], preds_val[:, i]) for i in range(self.n_targets_)])
            conv_layers = np.zeros(self.n_targets_, dtype=int)
            converged = np.array([all(~self.rf_importance_[i][np.arange(self.n_targets_) != i]) for i in range(self.n_targets_)])
            
            all_preds_train = {0: preds_train}
            all_preds_val = {0: preds_val}
            
            rlayer = 1
            def _train_predict_tracking(i, converged_i, rf_imp_i, X_train_fold, y_train_fold, X_val_fold, y_val_fold, conv_layers_fold, all_preds_train_fold, all_preds_val_fold, error_val_i):
                if not converged_i:
                    chosen_targets = np.where(rf_imp_i)[0]
                    X_tra_ext = X_train_fold.copy()
                    X_val_ext = X_val_fold.copy()
                    for ct in chosen_targets:
                        dep_layer = conv_layers_fold[ct]
                        X_tra_ext = np.column_stack([X_tra_ext, all_preds_train_fold[dep_layer][:, ct]])
                        X_val_ext = np.column_stack([X_val_ext, all_preds_val_fold[dep_layer][:, ct]])
                    
                    model = clone(self.base_estimator)
                    model.fit(X_tra_ext, y_train_fold[:, i])
                    p_train = model.predict(X_tra_ext)
                    p_val = model.predict(X_val_ext)
                    
                    rmse_val = self._get_rmse(y_val_fold[:, i], p_val)
                    if rmse_val + self.epsilon > error_val_i:
                        return True, error_val_i, p_train, p_val, False
                    else:
                        return False, rmse_val, p_train, p_val, True
                else:
                    return True, error_val_i, all_preds_train_fold[rlayer-1][:, i], all_preds_val_fold[rlayer-1][:, i], False

            while not np.all(converged):
                results_tracking = Parallel(n_jobs=self.n_jobs)(delayed(_train_predict_tracking)(
                    i, converged[i], self.rf_importance_[i], X_train, y_train, X_val, y_val, conv_layers, all_preds_train, all_preds_val, error_val[i]
                ) for i in range(self.n_targets_))
                
                new_preds_train = np.zeros_like(preds_train)
                new_preds_val = np.zeros_like(preds_val)
                
                for i, (conv_i, err_i, p_train, p_val, improved) in enumerate(results_tracking):
                    converged[i] = conv_i
                    error_val[i] = err_i
                    new_preds_train[:, i] = p_train
                    new_preds_val[:, i] = p_val
                    if improved:
                        conv_layers[i] = rlayer
                
                all_preds_train[rlayer] = new_preds_train
                all_preds_val[rlayer] = new_preds_val
                rlayer += 1
                if rlayer > 50: break
            
            fold_convergence_layers[fold_idx] = conv_layers
            fold_idx += 1
            
        # Determine final layers based on phi
        # convergence_tracking in R is a count of how many folds reached at least that layer
        max_layers = fold_convergence_layers.max()
        self.convergence_tracking_ = np.zeros((max_layers + 1, self.n_targets_))
        for layer in range(max_layers + 1):
            self.convergence_tracking_[layer] = np.sum(fold_convergence_layers >= layer, axis=0) / self.n_folds_tracking
            
        # Final layers: last layer where tracking >= phi
        self.convergence_layers_ = np.zeros(self.n_targets_, dtype=int)
        for i in range(self.n_targets_):
            valid_layers = np.where(self.convergence_tracking_[:, i] >= self.phi)[0]
            if len(valid_layers) > 0:
                self.convergence_layers_[i] = valid_layers[-1]
                
        self._final_modelling(X, y)

    def _final_modelling(self, X, y):
        # Final training on full dataset
        self.models_ = {}
        max_layer = self.convergence_layers_.max()
        
        preds = np.zeros((X.shape[0], self.n_targets_))
        all_preds = {}
        
        # Layer 0
        def _train_final_layer0(i, X_full, y_full):
            model = clone(self.base_estimator)
            model.fit(X_full, y_full[:, i])
            return i, model, model.predict(X_full)

        results_layer0 = Parallel(n_jobs=self.n_jobs)(delayed(_train_final_layer0)(i, X, y) for i in range(self.n_targets_))
        for i, model, p_train in results_layer0:
            self.models_[(0, i)] = model
            preds[:, i] = p_train
        all_preds[0] = preds
        
        # Subsequent layers
        def _train_final_subsequent(i, rlayer, conv_layers, rf_imp, X_full, y_full, all_preds_dict):
            if conv_layers[i] >= rlayer:
                chosen_targets = np.where(rf_imp)[0]
                X_ext = X_full.copy()
                for ct in chosen_targets:
                    dep_layer = min(rlayer - 1, conv_layers[ct])
                    X_ext = np.column_stack([X_ext, all_preds_dict[dep_layer][:, ct]])
                
                model = clone(self.base_estimator)
                model.fit(X_ext, y_full[:, i])
                return i, model, model.predict(X_ext)
            else:
                return i, None, all_preds_dict[rlayer-1][:, i]

        for rlayer in range(1, max_layer + 1):
            new_preds = np.zeros_like(preds)
            results_subsequent = Parallel(n_jobs=self.n_jobs)(delayed(_train_final_subsequent)(
                i, rlayer, self.convergence_layers_, self.rf_importance_[i], X, y, all_preds
            ) for i in range(self.n_targets_))
            
            for i, model, p_train in results_subsequent:
                if model is not None:
                    self.models_[(rlayer, i)] = model
                new_preds[:, i] = p_train
            all_preds[rlayer] = new_preds

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        max_layer = self.convergence_layers_.max()
        preds = np.zeros((X.shape[0], self.n_targets_))
        all_preds = {}
               # Layer 0
        def _predict_layer0(i, X_input, models_dict):
            return i, models_dict[(0, i)].predict(X_input)

        results_layer0 = Parallel(n_jobs=self.n_jobs)(delayed(_predict_layer0)(i, X, self.models_) for i in range(self.n_targets_))
        for i, p in results_layer0:
            preds[:, i] = p
        all_preds[0] = preds
        
        # Subsequent layers
        def _predict_subsequent(i, rlayer, conv_layers, rf_imp, X_input, all_preds_dict, models_dict):
            if conv_layers[i] >= rlayer:
                chosen_targets = np.where(rf_imp)[0]
                X_ext = X_input.copy()
                for ct in chosen_targets:
                    dep_layer = min(rlayer - 1, conv_layers[ct])
                    X_ext = np.column_stack([X_ext, all_preds_dict[dep_layer][:, ct]])
                return i, models_dict[(rlayer, i)].predict(X_ext)
            else:
                return i, all_preds_dict[rlayer-1][:, i]

        for rlayer in range(1, max_layer + 1):
            new_preds = np.zeros_like(preds)
            results_subsequent = Parallel(n_jobs=self.n_jobs)(delayed(_predict_subsequent)(
                i, rlayer, self.convergence_layers_, self.rf_importance_[i], X, all_preds, self.models_
            ) for i in range(self.n_targets_))
            
            for i, p in results_subsequent:
                new_preds[:, i] = p
            all_preds[rlayer] = new_preds
            
        # Return prediction from the specific convergence layer for each target
        final_preds = np.zeros((X.shape[0], self.n_targets_))
        for i in range(self.n_targets_):
            final_preds[:, i] = all_preds[self.convergence_layers_[i]][:, i]
            
        return final_preds

def aCC(y_true, y_pred):
    """Average Correlation Coefficient"""
    corrs = []
    for i in range(y_true.shape[1]):
        c = np.corrcoef(y_true[:, i], y_pred[:, i])[0, 1]
        if np.isnan(c): c = 0
        corrs.append(c)
    return np.mean(corrs)

def aRMSE(y_true, y_pred):
    """Average Root Mean Squared Error"""
    rmses = []
    for i in range(y_true.shape[1]):
        rmses.append(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
    return np.mean(rmses)

def aRRMSE(y_true, y_pred):
    """Average Relative Root Mean Squared Error"""
    rrrmses = []
    for i in range(y_true.shape[1]):
        num = np.sum((y_true[:, i] - y_pred[:, i])**2)
        den = np.sum((y_true[:, i] - np.mean(y_true[:, i]))**2)
        if den == 0: den = 1
        rrrmses.append(np.sqrt(num/den))
    return np.mean(rrrmses)
