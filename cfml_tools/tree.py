# importing required modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score

# class for using a decision tree to compute effects
class DecisionTreeCounterfactual:

    """
    Counterfactual estimation using a decision tree.

    Given explanatory variables X, target variable y and treatment variable W, 
    this class implements an individual counterfactual estimation model. 
    We can break down the process in three steps:

    1 - model step) Fit a decision tree to X and y
    2 - comparison step) at each of the tree's leaves, compare W and y to determine the counterfactuals for the leaf 
    3 - prediction step) assign new samples to a leaf, and predict counterfactuals

    Parameters
    ----------

    model : object, optinal (default=None)

    Tree-based model which implements sklearn's API, particularly the .apply() method.
    Must be already configured.

    If None, model will be DecisionTreeRegressor(min_samples_leaf=100).

    min_sample_effect : int, optional (default=10)

    The minimum number of samples in a neighborhood to deem a counterfactual estimate valid, for a given W. 
    If there's less treated/untreated elements than min_sample_effect, the counterfactual will be NaN.

    save_explanatory : bool, optional (default=False)

    Save explanatory variables for explaining predictions. May cause large memory overhead.

    random_state : int, optional (default=None)

    If int, random_state is the seed used by the random number generator;
    If RandomState instance, random_state is the random number generator;
    If None, the random number generator is the RandomState instance used
    by `np.random`.
    
    """   

    # initializing
    def __init__(self, model=None, min_sample_effect=10, save_explanatory=False, random_state=None):

        # storing model
        if model == None:
            self.model = DecisionTreeRegressor(min_samples_leaf=100)
        else:
            self.model = model

        # storing variables
        self.min_sample_effect = int(min_sample_effect)
        self.random_state = random_state
        self.save_explanatory = save_explanatory

    def _test_treatment_linear_discriminative_power(self, leaf_df):
        """
        Using data from elements on leaf, test if treatments are randomly assigned by using a linear model to predict it.

        Parameters
        ----------
        
        leaf_df : pd.DataFrame
        
        Training datafarme with features (X), treatment assignments (W) and target (y)

        Returns
        -------

        return : float

        Average AUC (if multiclass) of treatment assignment predictive model for leaf

        """

        # organizing and standardizing data for model 
        W_leaf = leaf_df['W']
        X_leaf = leaf_df.drop(['W', 'y'], axis=1)
        X_leaf = StandardScaler().fit_transform(X_leaf)

        # fitting model
        lr = LogisticRegression(solver='lbfgs')
        lr.fit(X_leaf, W_leaf)
        
        # predicting
        W_predicted = lr.predict_proba(X_leaf)
        
        # if we have a single treatment treat as binary 
        # classification problem, if not do nothing and
        # roc_auc_score function will take care of it
        if W_predicted.shape[1] == 2:
            W_predicted = W_predicted[:,1]
        
        # computing score (avg. AUC)
        score = roc_auc_score(
            W_leaf, 
            W_predicted, 
            multi_class='ovr', 
            average='weighted'
        )
        
        return score
    
    def _compute_treatment_confounding(self, filtered_train_df):
        """
        Apply tests to determine if treatments are randomly assigned for all leaves

        Parameters
        ----------
        
        filtered_train_df : pd.DataFrame
        
        Subset of training dataframe for elements on leaves that effects are valid (given min_sample_effect parameter)

        Returns
        -------

        confounding_df: pd.DataFrame

        Dataframe with confouding scores for each leaf

        """

        # just apply _test_treatment_linear_discriminative_power
        # for all leaves
        confounding_df = (
            filtered_train_df
            .groupby('leaf')
            .apply(self._test_treatment_linear_discriminative_power)
            .to_frame(name='confounding_score')
        )

        # using multi index to work in final dataframe
        confounding_df.columns = pd.MultiIndex.from_tuples([('confounding_score', '')])

        return confounding_df

    def _compute_leaf_counterfactuals(self, filtered_train_df):
        """
        Compute counterfactuals for each valid leaf

        Parameters
        ----------
        
        filtered_train_df : pd.DataFrame
        
        Subset of training dataframe for elements on leaves that effects are valid (given min_sample_effect parameter)

        Returns
        -------

        leaf_counterfactual_df : pd.DataFrame

        Dataframe with expected outcomes for each treatment

        """
        
        # computing avg outcomes for each treatment
        leaf_counterfactual_df = (
            filtered_train_df
            .pivot_table(values='y', columns='W', index='leaf')
            .reset_index()
            .set_index('leaf')
        )
        
        # fomatting column names 
        leaf_counterfactual_df.columns = (
            pd.MultiIndex
            .from_product([ ['avg_outcome'], leaf_counterfactual_df.columns ],
                          names=[None,'W'])
        )
        
        return leaf_counterfactual_df
        
    def _compute_feature_dispersion(self, train_df):
        """
        Computes feature dispersion between treatments in leaves, to help diagnosing if effects are valid

        Parameters
        ----------
        
        train_df : pd.DataFrame
        
        Training dataframe, as stored using the "save_explanatory=True" parameter

        Returns
        -------

        feat_dispersion : pd.DataFrame

        Difference in percentiles between elements with different treatment in each leaf.

        """

        # computing rank (percentiles) for each feature
        # and pivot by treatment to show user
        feat_percentiles_pivot = (
            train_df
            .set_index(['leaf','W'])
            .drop(['y'], axis=1)
            .rank(pct=True)
            .pivot_table(index='leaf', columns='W')
            .dropna()
        )

        # putting levels to same column to match final output #
        # add prefix to first level
        level_0 = (
            'percentile_' + 
            feat_percentiles_pivot.columns.get_level_values(0)
        )
        
        # second level stays the same
        level_1 = (
            feat_percentiles_pivot.columns.get_level_values(1)
        )

        # applying to df
        feat_percentiles_pivot.columns = (
            pd.MultiIndex.from_arrays([level_0, level_1])
        )
        
        return feat_percentiles_pivot

    # fit model
    def fit(self, X, W, y, verbose=0):

        """
        Get counterfactual estimates given explanatory variables X, treatment variable W and target y
        This method will fit a decision tree from X to y and store outcomes given distinct W values at each 
        of its leaves

        Parameters
        ----------
        
        X : array-like or sparse matrix of shape = [n_samples, n_features]
        
        Data with explanatory variables, with possible confounders of treatment assignment and effect.

        W : array-like, shape = [n_samples] 

        Treatment variable. The model will try to estimate a counterfactual outcome for each unique value in this variable.
        Should not exceed 10 unique values.

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

        # storing column names
        self.col_names = X.columns

        # saving explanatory variables, if applicable
        if self.save_explanatory:
            self.train_df = X.assign(leaf=self.model.apply(X), W=W, y=y)

        # initializing a df with counterfactuals for each leaf
        self.leaf_counterfactual_df = (
            pd.DataFrame({'leaf': self.model.apply(X), 'y': y, 'W': W})
            .assign(count=1)
            .groupby(['leaf','W']).sum()
        )

        # making estimates based on small samples invalid
        invalid_estimate_mask = (
            self.leaf_counterfactual_df['count'] < 
            self.min_sample_effect
        )
        self.leaf_counterfactual_df.loc[invalid_estimate_mask, 'y'] = np.nan

        # correcting y by taking average
        self.leaf_counterfactual_df['y'] = (
            self.leaf_counterfactual_df['y'] / 
            self.leaf_counterfactual_df['count']
        )
        
        # return self
        return self

    # method for predicting counterfactuals
    def predict(self, X, verbose=0):

        """
        Predict counterfactual outcomes for X. 
        This method runs new samples through the tree, and predicts counterfactuals
        given which leaf new samples ended up into

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

        # getting decision tree cluster assignments
        leaves_score = pd.DataFrame({'leaf': self.model.apply(X), 'id': X.index})

        # to get counterfactual df we just need to join leaves_test with leaf_counterfactual_df
        counterfactual_df = (
            leaves_score
            .merge(self.leaf_counterfactual_df.reset_index(), how='left')
            .pivot(values='y', columns='W', index='id')
        )
        
        # correcting columns
        counterfactual_df.columns = (
            pd.MultiIndex
            .from_product([['y_hat'], counterfactual_df.columns,],names=[None,'W'])
        )
        
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


    def run_leaf_diagnostics(self):
        """
        Run leaf diagnostics, showing counfounding score, feature distribuitions and counterfactuals for each leaf.

        Returns
        -------

        leaf_diagnostics_df : pd.DataFrame

        Dataframe with leaf diagnostics

        """

        # first, we calculate only where effects are valid #

        # effect is invalid on leaves marked with nan
        # or leaves that only have one kind of assignment
        mask_nan = self.leaf_counterfactual_df['y'].isnull()
        mask_single_assignment = self.leaf_counterfactual_df.groupby('leaf').size() == 1

        # joining masks and getting invalid leaves
        mask_invalid_effect = mask_nan | mask_single_assignment
        invalid_leaves = self.leaf_counterfactual_df.loc[mask_invalid_effect].index.get_level_values('leaf').values

        # filtering train df out of invalid leaves
        mask_invalid_leaves = self.train_df['leaf'].isin(invalid_leaves)
        filtered_train_df = self.train_df.loc[~mask_invalid_leaves]

        # then, we calculate quantities like #
        # counfounding, dispersion and counterfactuals #
        # for each leaf, so we can perform criticism #

        # computing discriminative power
        confounding_df = self._compute_treatment_confounding(filtered_train_df)

        # computing leaf effects
        leaf_counterfactual_df = self._compute_leaf_counterfactuals(filtered_train_df)

        # computing feature dispersion
        feat_percentiles_df = self._compute_feature_dispersion(self.train_df)

        # leaf diagnostics df
        dfs = [leaf_counterfactual_df, feat_percentiles_df, confounding_df]
        leaf_diagnostics_df = pd.concat(
            dfs, 
            axis=1, 
            join='inner', 
            levels=[0,1]
        )

        return leaf_diagnostics_df

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

        # checking which leaf sample is assigned to
        sample_leaf = self.model.apply(sample)

        # querying comparables
        if self.save_explanatory:
            comparables_table = (
                self.train_df
                .query('leaf == {}'.format(sample_leaf))
                .drop('leaf', axis=1)
            )
        else:
            raise ValueError('Model did not store training samples to get explanations from. Setting save_explanatory=True will solve the issue')

        # returning comparables table
        return comparables_table