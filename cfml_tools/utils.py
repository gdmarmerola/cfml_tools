import numpy as np
from numpy import nan
import pandas as pd

# REPRODUCING make_confounded_data FUNCTION FROM
# FKLEARN FOR SYMPLICITY, AS FKLEARN HAS STRICT REQUIREMENTS
# FOR SKLEARN, PANDAS AND NUMPY AND WE WANTED TO DECOUPLE FROM IT
# WE'RE NOT CREDIBLE FOR THIS CODE
# LINK: https://github.com/nubank/fklearn/blob/master/src/fklearn/data/datasets.py

def make_confounded_data(n):
    """
    Generates fake data for counterfactual experimentation. The covariants are
    sex, age and severity, the treatment is a binary variable, medication and the response
    days until recovery.
    Parameters
    ----------
    n : int
        The number of samples to generate
    Returns
    ----------
    df_rnd : pd.DataFrame
        A dataframe where the treatment is randomly assigned.
    df_obs : pd.DataFrame
        A dataframe with confounding.
    df_df : pd.DataFrame
        A counter factual dataframe with confounding. Same as df_obs, but
        with the treatment flipped.
    """

    def get_severity(df):
        return ((np.random.beta(1, 3, size=df.shape[0]) * (df["age"] < 30))
                + (np.random.beta(3, 1.5, size=df.shape[0]) * (df["age"] >= 30)))

    def get_treatment(df):
        return (.33 * df["sex"]
                + 1.5 * df["severity"]
                + 0.15 * np.random.normal(size=df.shape[0]) > 0.8).astype(float)

    def get_recovery(df):
        return np.random.poisson(np.exp(2
                                        + 0.5 * df["sex"]
                                        + 0.03 * df["age"]
                                        + df["severity"]
                                        - df["medication"]))

    np.random.seed(1111)
    sexes = np.random.randint(0, 2, size=n)
    ages = np.random.gamma(8, scale=4, size=n)
    meds = np.random.randint(0, 2, size=n)

    # random data
    df_rnd = pd.DataFrame(dict(sex=sexes, age=ages, medication=meds))
    df_rnd['severity'] = get_severity(df_rnd)
    df_rnd['recovery'] = get_recovery(df_rnd)

    features = ['sex', 'age', 'severity', 'medication', 'recovery']
    df_rnd = df_rnd[features]  # to enforce column order

    # obs data
    df_obs = df_rnd.copy()
    df_obs['medication'] = get_treatment(df_obs)
    df_obs['recovery'] = get_recovery(df_obs)

    # counter_factual data
    df_ctf = df_obs.copy()
    df_ctf['medication'] = ((df_ctf['medication'] == 1) ^ 1).astype(float)
    df_ctf['recovery'] = get_recovery(df_ctf)

    return df_rnd, df_obs, df_ctf
