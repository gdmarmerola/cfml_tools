# importing required modules
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from pynndescent import NNDescent

# class for computing effects using forest embeddings
class ForestEmbeddingsCounterfactual:
    
    """
    Counterfactual estimation using forest embeddings.

    Given explanatory variables X, target variable y and treatment variable W, 
    this class implements an individual counterfactual estimation model. 
    We can break down the process in four steps:

    1 - model step) Fit and validate an ensemble of trees (ET, RF, etc) from X to y
    2 - embedding step) Build a supervised embedding using forest's trees leaves
    3 - kNN step) For each sample, find K nearest neighbors in this new space 
    4 - comparison step) Compare W and y for each of the neighborhoods to determine the counterfactuals for each sample

    Parameters
    ----------

    model : object, optinal (default=None)

    Forest-based model which implements sklearn's API, particularly the .apply() method. 
    Must be already configured. Classification and regression models accepted.

    If None, model will be ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=5, bootstrap=True, n_jobs=-1).

    n_neighbors : int, optional (default=200)

    Number of neighbors to be considered at the kNN step. There's a bias-variance tradeoff here: 
    set n_neighbors too low, estimates will be volatile and unreliable. 
    Set n_neighbors too high, and the estimate will be biased (neighbors won't be comparable). 

    min_sample_effect : int, optional (default=10)

    The minimum number of samples in a neighborhood for the counterfactual estimate to be valid, for a given W. 
    If there's less treated/untreated elements than min_sample_effect in a neighborhood, the counterfactual will be NaN.

    save_explanatory : bool, optional (default=False)

    Save explanatory variables for explaining predictions. May cause large memory overhead.

    random_state : int, optional (default=None)

    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.
    
    """   

    # initializing
    def __init__(self, model=None, n_neighbors=200, min_sample_effect=10, save_explanatory=False, random_state=None):

        # storing model
        if model == None:
            self.model = ExtraTreesRegressor(n_estimators=1000, min_samples_leaf=5, bootstrap=True, n_jobs=-1)
        else:
            self.model = model

        # storing variables
        self.n_neighbors = int(n_neighbors)
        self.min_sample_effect = int(min_sample_effect)
        self.save_explanatory = save_explanatory
        self.random_state = random_state

    # method for computing embedding
    def _get_forest_embed(self, X):

        """
        Wrapper for extracting embeddings from forests given selected mode.
        Model must be fitted.
        """

        # applying the model to get leaves
        this_embed = self.model.apply(X)

        # returning forest embedding
        return this_embed


    # fit model and neighbors
    def fit(self, X, W, y, verbose=0):

        """
        Fit a counterfactual estimation model given explanatory variables X, treatment variable W and target y
        This method fits a forest-based model, extracts a supervised embedding from its leaves, 
        and builds an nearest neighbor index on the embedding

        Parameters
        ----------
        
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Data with explanatory variables, with possible confounders of treatment assignment and effect.

        W : array-like, shape = [n_samples] 

        Treatment variable. The model will try to estimate a counterfactual outcome for each unique value in this variable.
        Should not exceed 10 values.

        y: array-like, shape = [n_samples]
    
        Target variable. 

        verbose : int, optional (default=0)

        Verbosity level.

        Returns
        -------

        self: object

        """
        
        # checking if W has too many unique values
        if len(np.unique(W)) > 10:
            raise ValueError('More than 10 unique values for W. Too many unique values will make the process very expensive.')

        # fitting the model
        self.model.fit(X, y)

        # getting forest embedding from model
        self.train_embed_ = self._get_forest_embed(X)

        # create neighbor index
        self.nn_index = NNDescent(self.train_embed_, metric='hamming')

        # creating a df with treatment assignments and outcomes
        self.train_outcome_df = pd.DataFrame({'neighbor': range(X.shape[0]), 'y': y, 'W': W})

        # saving explanatory variables
        if self.save_explanatory:
            self.X_train = X.assign(W=W, y=y)

        # return self
        return self

    # method for predicting counterfactuals
    def predict(self, X, verbose=0):

        """
        Predict counterfactual outcomes for X. 
        This method will search the nearest neighbor index built using .fit(), and estimate
        counterfactual outcomes using kNN

        Parameters
        ----------
        
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Data with explanatory variables, with possible confounders of treatment assignment and effect.

        verbose : int, optional (default=0)

        Verbosity level.

        Returns
        -------
        
        counterfactual_df : pd.DataFrame

        Counterfactual outcomes per sample.

        """

        # getting forest embedding from model
        X_embed_ = self._get_forest_embed(X)

        # getting nearest neighbors and distances from index
        neighs, dists = self.nn_index.query(X_embed_, k=self.n_neighbors + 1)
                        
        # creating a df for neighbor ids
        neighs_df = (
            pd.DataFrame(neighs)
            .reset_index()
            .melt(id_vars='index')
            .rename(columns={'index':'id','value':'neighbor'})
            .reset_index(drop=True)
        )

        # creating a df for the similarities
        similarities_df = (
            pd.DataFrame(1 - dists)
            .reset_index()
            .melt(id_vars='index')
            .rename(columns={'index':'id','value':'weight'})
            .reset_index(drop=True)
        )

        # joining the datasets and adding weighted y variable
        nearest_neighs_df = (
            neighs_df
            .merge(similarities_df)
            .drop('variable', axis=1)
            .merge(self.train_outcome_df, on='neighbor', how='left')
            .assign(y_weighted = lambda x: x.y*(x.weight))
            .sort_values('id')
        )
        
        # processing to get the effects
        counterfactual_df = nearest_neighs_df.assign(count=1).groupby(['id','W']).sum()
        #counterfactual_df['y_hat'] = counterfactual_df['y']/counterfactual_df['count']
        counterfactual_df['y_hat'] = counterfactual_df['y_weighted']/counterfactual_df['weight']
        counterfactual_df.loc[counterfactual_df['count'] < self.min_sample_effect,'y_hat'] = np.nan
        counterfactual_df = counterfactual_df.pivot_table(values=['y_hat'], columns='W', index='id')
        
        # returning counterfactual df
        return counterfactual_df


    # running CV for model parameters
    def get_cross_val_scores(self, X, y, scoring=None, verbose=0):

        """
        Estimate model generalization power with 5-fold CV.

        Parameters
        ----------
        
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Data with explanatory variables, with possible confounders of treatment assignment and effect.

        y: array-like, shape = [n_samples]

        Target variable. 
        
        scoring : string, callable or None, optional, default: None
        
        Scoring method for sklearn's cross_val_score function:

        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)`` which should return only
        a single value.

        Similar to :func:`cross_validate`
        but only a single metric is permitted.

        If None, the estimator's default scorer (if available) is used.
        
        verbose : int, optional (default=0)

        Verbosity level for sklearn's function cross_val_score.

        Returns
        -------
        
        scores : array of float, shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.
        
        """
        
        # CV method
        kf = KFold(
            n_splits=5, 
            shuffle=True, 
            random_state=self.random_state
        )

        # generating validation predictions
        scores = cross_val_score(
            self.model, 
            X, 
            y, 
            cv=kf, 
            scoring=scoring, 
            verbose=verbose
        )

        # calculating result
        return scores

    # generating manifold with UMAP
    def get_umap_embedding(self, X, verbose=0):

        """
        Compute a 2D manifold from the forest embedding for validation and criticism.

        Parameters
        ----------
        
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Data with explanatory variables, with possible confounders of treatment assignment and effect.

        verbose : int, optional (default=0)

        Verbosity level for UMAP.

        Returns
        -------
        
        reduced_embed : array of shape = [n_samples, 2]

        2D representation of forest embedding using UMAP. 

        """
        
        # getting forest embedding from model
        X_embed_ = self._get_forest_embed(X)

        # reducing embedding to 2 dimensions
        reduced_embed = (
            UMAP(metric='hamming', verbose=verbose)
            .fit_transform(X_embed_)
        )
        
        # returning 
        return reduced_embed

    # method for explaning predictions
    def explain(self, sample):

        """
        Explain predcitions of counterfactual outcomes for one sample. 
        This method shows diagnostics and comparables so you can trust
        and explain counterfactual predictions to others

        Parameters
        ----------
        
        sample : array-like or sparse matrix of shape = [1, n_features]
        
        Sample that you want to get explanations for

        Returns
        -------
        
        comparables_table : pd.DataFrame

        Table of comparable elements.

        """

        # getting forest embedding from model
        sample_embed = self._get_forest_embed(sample)

        # getting nearest neighbors and distances from index
        neighs, dists = self.nn_index.query(sample_embed, k=self.n_neighbors + 1)

        # querying comparables
        if self.save_explanatory:
            comparables_table = self.X_train.iloc[neighs[0]]
        else:
            raise ValueError('Model did not store training samples to get explanations from. Setting save_explanatory=True will solve the issue')

        # returning comparables table
        return comparables_table