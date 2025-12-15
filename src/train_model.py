import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    QuantileTransformer,
    RobustScaler,
)
from skopt import BayesSearchCV
from skopt.space import Categorical, Integer, Real


class ColumnRemover(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.columns_to_drop:
            return X

        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.columns_to_drop, errors="ignore")

        return X

    def get_params(self, deep=True):
        return {"columns_to_drop": self.columns_to_drop}

    def set_params(self, **params):
        self.columns_to_drop = params["columns_to_drop"]
        return self


class OutlierHandler(BaseEstimator, TransformerMixin):
    def __init__(self, contamination=0.05):
        self.contamination = contamination

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for col in X.select_dtypes(include=["float64", "int64"]).columns:
            q1 = X[col].quantile(0.01)
            q3 = X[col].quantile(0.99)
            X[col] = X[col].clip(q1, q3)

        return X


class ImprovedImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_imputer = KNNImputer(n_neighbors=10)
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.numeric_columns = None
        self.categorical_columns = None

    def fit(self, X, y=None):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        self.numeric_columns = X_df.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()
        self.categorical_columns = X_df.select_dtypes(
            exclude=["int64", "float64"]
        ).columns.tolist()

        if self.numeric_columns:
            self.num_imputer.fit(X_df[self.numeric_columns])
        if self.categorical_columns:
            self.cat_imputer.fit(X_df[self.categorical_columns])

        return self

    def transform(self, X):
        X_df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X.copy())

        for col in X_df.columns:
            missing_rate = X_df[col].isnull().mean()
            if missing_rate > 0.05:
                X_df[f"{col}_was_missing"] = X_df[col].isnull().astype(int)

        if self.numeric_columns:
            cols_to_impute = [
                col for col in self.numeric_columns if col in X_df.columns
            ]
            if cols_to_impute:
                X_df[cols_to_impute] = self.num_imputer.transform(X_df[cols_to_impute])

        if self.categorical_columns:
            cols_to_impute = [
                col for col in self.categorical_columns if col in X_df.columns
            ]
            if cols_to_impute:
                X_df[cols_to_impute] = self.cat_imputer.transform(X_df[cols_to_impute])

        if "hasPrivacyLink" in X_df.columns and X_df["hasPrivacyLink"].isnull().any():
            X_df["hasPrivacyLink"] = X_df["hasPrivacyLink"].fillna(0)

        if (
            "isCorporateEmailScore" in X_df.columns
            and X_df["isCorporateEmailScore"].isnull().any()
        ):
            X_df["isCorporateEmailScore"] = X_df["isCorporateEmailScore"].fillna(25)

        return X_df


class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
        else:
            try:
                X_transformed = pd.DataFrame(X)
            except Exception:
                return X

        if "downloads" in X_transformed.columns:
            X_transformed["min_downloads"] = X_transformed["downloads"].apply(
                lambda x: int(str(x).split("-")[0].strip().replace(",", ""))
                if isinstance(x, str) and "-" in x
                else np.nan
            )
            X_transformed["max_downloads"] = X_transformed["downloads"].apply(
                lambda x: int(str(x).split("-")[-1].strip().replace(",", ""))
                if isinstance(x, str) and "-" in x
                else np.nan
            )
            X_transformed["log_min_downloads"] = np.log1p(
                X_transformed["min_downloads"].fillna(0)
            )
            X_transformed["log_max_downloads"] = np.log1p(
                X_transformed["max_downloads"].fillna(0)
            )

        if "developerCountry" in X_transformed.columns:
            X_transformed["dev_in_US"] = X_transformed["developerCountry"].apply(
                lambda x: 1 if str(x) == "US" else 0
            )
            X_transformed["dev_unknown"] = X_transformed["developerCountry"].apply(
                lambda x: 1
                if str(x)
                in ["ADDRESS NOT LISTED IN PLAYSTORE", "CANNOT IDENTIFY COUNTRY"]
                else 0
            )

        if "countryCode" in X_transformed.columns:
            X_transformed["global_market"] = X_transformed["countryCode"].apply(
                lambda x: 1 if str(x) == "GLOBAL" else 0
            )
            X_transformed["north_america_market"] = X_transformed["countryCode"].apply(
                lambda x: 1 if str(x) == "NA" else 0
            )
            X_transformed["coppa_jurisdiction"] = X_transformed["countryCode"].apply(
                lambda x: 1 if str(x) in ["NA", "GLOBAL"] else 0
            )

        compliance_cols = ["hasPrivacyLink", "hasTermsOfServiceLink"]
        if all(col in X_transformed.columns for col in compliance_cols):
            for col in compliance_cols:
                if X_transformed[col].dtype == "object":
                    X_transformed[col] = X_transformed[col].apply(
                        lambda x: 1
                        if str(x).lower() in ["true", "1", "yes", "t"]
                        else 0
                    )
            X_transformed["compliance_score"] = X_transformed[compliance_cols].sum(
                axis=1
            )
            X_transformed["missing_privacy_policy"] = (
                1 - X_transformed["hasPrivacyLink"]
            ).astype(int)

        if "hasTermsOfServiceLinkRating" in X_transformed.columns:
            X_transformed["tos_quality_numeric"] = X_transformed[
                "hasTermsOfServiceLinkRating"
            ].map({"low": 1, "medium": 2, "high": 3})

        if "appAge" in X_transformed.columns:
            if X_transformed["appAge"].dtype == "object":
                X_transformed["appAge"] = pd.to_numeric(
                    X_transformed["appAge"], errors="coerce"
                )

            X_transformed["app_age_years"] = X_transformed["appAge"] / 365
            X_transformed["is_new_app"] = (X_transformed["appAge"] < 90).astype(
                int
            )  # < 3 months
            X_transformed["is_established_app"] = (
                X_transformed["appAge"] > 365
            ).astype(int)  # > 1 year

        if "deviceType" in X_transformed.columns:
            X_transformed["is_tablet"] = X_transformed["deviceType"].apply(
                lambda x: 1 if isinstance(x, str) and "tablet" in x.lower() else 0
            )
            X_transformed["is_smartphone"] = X_transformed["deviceType"].apply(
                lambda x: 1 if isinstance(x, str) and "smartphone" in x.lower() else 0
            )
            X_transformed["is_multi_device"] = X_transformed["deviceType"].apply(
                lambda x: 1 if isinstance(x, str) and "," in x else 0
            )

        if "primaryGenreName" in X_transformed.columns:
            child_related_genres = [
                "Games",
                "Education",
                "Family",
                "Entertainment",
                "Kids",
            ]
            X_transformed["is_child_related_genre"] = X_transformed[
                "primaryGenreName"
            ].apply(lambda x: 1 if str(x) in child_related_genres else 0)

            X_transformed["is_game"] = X_transformed["primaryGenreName"].apply(
                lambda x: 1 if str(x) == "Games" else 0
            )
            X_transformed["is_education"] = X_transformed["primaryGenreName"].apply(
                lambda x: 1 if str(x) == "Education" else 0
            )
            X_transformed["is_family"] = X_transformed["primaryGenreName"].apply(
                lambda x: 1 if str(x) == "Family" else 0
            )

        if "isCorporateEmailScore" in X_transformed.columns:
            if X_transformed["isCorporateEmailScore"].dtype == "object":
                X_transformed["isCorporateEmailScore"] = pd.to_numeric(
                    X_transformed["isCorporateEmailScore"], errors="coerce"
                )

            X_transformed["is_professional_developer"] = (
                X_transformed["isCorporateEmailScore"] > 75
            ).astype(int)
            X_transformed["is_amateur_developer"] = (
                X_transformed["isCorporateEmailScore"] < 25
            ).astype(int)

        if (
            "averageUserRating" in X_transformed.columns
            and "userRatingCount" in X_transformed.columns
        ):
            for col in ["averageUserRating", "userRatingCount"]:
                if X_transformed[col].dtype == "object":
                    X_transformed[col] = pd.to_numeric(
                        X_transformed[col], errors="coerce"
                    )

            X_transformed["weighted_rating"] = X_transformed[
                "averageUserRating"
            ] * np.log1p(X_transformed["userRatingCount"].fillna(0))
            X_transformed["high_engagement"] = (
                X_transformed["userRatingCount"] > 1000
            ).astype(int)

        safety_cols = [
            "appContentBrandSafetyRating",
            "appDescriptionBrandSafetyRating",
            "mfaRating",
        ]
        if all(col in X_transformed.columns for col in safety_cols):
            for col in safety_cols:
                X_transformed[f"{col}_numeric"] = X_transformed[col].map(
                    {"low": 1, "medium": 2, "high": 3}
                )

            numeric_cols = [f"{col}_numeric" for col in safety_cols]
            X_transformed["brand_safety_score"] = X_transformed[numeric_cols].mean(
                axis=1
            )

        if "adSpent" in X_transformed.columns:
            if X_transformed["adSpent"].dtype == "object":
                X_transformed["adSpent"] = pd.to_numeric(
                    X_transformed["adSpent"], errors="coerce"
                )

            X_transformed["log_ad_spent"] = np.log1p(X_transformed["adSpent"].fillna(0))
            X_transformed["high_ad_spender"] = (
                X_transformed["adSpent"] > X_transformed["adSpent"].median()
            ).astype(int)

        if (
            "dev_in_US" in X_transformed.columns
            and "is_child_related_genre" in X_transformed.columns
        ):
            X_transformed["us_child_app"] = (
                X_transformed["dev_in_US"] & X_transformed["is_child_related_genre"]
            ).astype(int)

        if (
            "is_new_app" in X_transformed.columns
            and "is_child_related_genre" in X_transformed.columns
        ):
            X_transformed["new_child_app"] = (
                X_transformed["is_new_app"] & X_transformed["is_child_related_genre"]
            ).astype(int)

        if (
            "log_min_downloads" in X_transformed.columns
            and "missing_privacy_policy" in X_transformed.columns
        ):
            X_transformed["popular_no_privacy"] = (
                (
                    X_transformed["log_min_downloads"]
                    > X_transformed["log_min_downloads"].median()
                )
                & X_transformed["missing_privacy_policy"]
            ).astype(int)

        if (
            "is_child_related_genre" in X_transformed.columns
            and "is_amateur_developer" in X_transformed.columns
        ):
            X_transformed["child_app_amateur_dev"] = (
                X_transformed["is_child_related_genre"]
                & X_transformed["is_amateur_developer"]
            ).astype(int)

        if (
            "is_child_related_genre" in X_transformed.columns
            and "high_engagement" in X_transformed.columns
        ):
            X_transformed["popular_child_app"] = (
                X_transformed["is_child_related_genre"]
                & X_transformed["high_engagement"]
            ).astype(int)

        return X_transformed


class EnhancedFeatureExtractor(FeatureExtractor):
    def transform(self, X):
        X = super().transform(X)

        if isinstance(X, pd.DataFrame):
            reference_date = pd.to_datetime("2023-11-01")

            if "appAge" in X.columns:
                X["release_day_of_year"] = (
                    reference_date - pd.to_timedelta(X["appAge"], unit="D")
                ).dt.dayofyear
                X["release_sin"] = np.sin(2 * np.pi * X["release_day_of_year"] / 365)
                X["release_cos"] = np.cos(2 * np.pi * X["release_day_of_year"] / 365)

                X["post_coppa_2013"] = (
                    X["appAge"] < (reference_date - pd.to_datetime("2013-07-01")).days
                ).astype(int)
                X["post_ftc_crackdown"] = (
                    X["appAge"] < (reference_date - pd.to_datetime("2021-01-01")).days
                ).astype(int)

                X["app_age_quantile"] = pd.qcut(X["appAge"], 4, labels=False)
                X["log_app_age"] = np.log1p(X["appAge"])

            if "lastUpdated" in X.columns:
                update_date = pd.to_datetime(X["lastUpdated"], errors="coerce")
                days_since_update = (reference_date - update_date).dt.days

                X["days_since_update"] = days_since_update
                X["update_frequency"] = X["appAge"] / (X["versionCount"] + 1e-6)
                X["stale_app"] = (days_since_update > 365).astype(int)
                X.update_frequency = np.where(
                    X.update_frequency == np.inf, 365, X.update_frequency
                )

                X["update_regularity"] = (
                    X.groupby("developerId")["days_since_update"]
                    .transform("std")
                    .fillna(0)
                )

            if "firstMadeAvailable" in X.columns:
                available_date = pd.to_datetime(
                    X["firstMadeAvailable"], errors="coerce"
                )
                X["years_on_market"] = (reference_date - available_date).dt.days / 365
                X["compliance_era"] = (
                    (reference_date.year - available_date.dt.year) > 3
                ).astype(int)

            if "userRatingCount" in X.columns and "averageUserRating" in X.columns:
                X["recent_rating_velocity"] = X["userRatingCount"] / (
                    X["appAge"] + 1e-6
                )
                X["rating_trend"] = (
                    X["averageUserRating"].rolling(window=30, min_periods=1).mean()
                )

                if "lastUpdated" in locals():
                    rating_decay = 0.95**days_since_update
                    X["weighted_rating"] = X["averageUserRating"] * rating_decay

            if "developerId" in X.columns:
                dev_stats = (
                    X.groupby("developerId")["appAge"]
                    .agg(["mean", "count", "std"])
                    .reset_index()
                )
                dev_stats.columns = [
                    "developerId",
                    "dev_exp_mean",
                    "apps_count",
                    "dev_exp_std",
                ]
                X = X.merge(dev_stats, on="developerId", how="left")

                dev_start = (
                    X.groupby("developerId")["appAge"]
                    .max()
                    .reset_index(name="dev_start_age")
                )
                X = X.merge(dev_start, on="developerId", how="left")
                X["dev_seniority"] = X["appAge"] - X["dev_start_age"]

            if all(c in X.columns for c in ["dev_in_US", "app_age_quantile"]):
                X["us_modern_app"] = X["dev_in_US"] * (
                    X["app_age_quantile"] > 2
                ).astype(int)

            if (
                "post_ftc_crackdown" in X.columns
                and "is_child_related_genre" in X.columns
            ):
                X["recent_child_app"] = (
                    X["post_ftc_crackdown"] * X["is_child_related_genre"]
                )

            if all(c in X.columns for c in ["years_on_market", "compliance_score"]):
                X["compliance_trend"] = X["compliance_score"] * np.log1p(
                    X["years_on_market"]
                )

            if "release_day_of_year" in X.columns:
                X["holiday_release"] = X["release_day_of_year"].apply(
                    lambda d: 1 if d in range(300, 320) or d in range(150, 180) else 0
                )

        for col in X.select_dtypes(include=[np.number]).columns:
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            X[col] = (
                X[col].fillna(X[col].median())
                if X[col].dtype.kind in "iufc"
                else X[col]
            )

        return X


numerical_pipeline = Pipeline(
    [
        (
            "quantile_transform",
            QuantileTransformer(output_distribution="normal", random_state=42),
        ),
        ("scaler", RobustScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
)

high_card_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        ("tgt_enc", TargetEncoder(smoothing=0.3)),
    ]
)


if __name__ == "__main__":
    df_train = pd.read_csv("./data/train.csv")
    df_test = pd.read_csv("./data/test.csv")

    train_set = df_train.copy()

    X_train = train_set.drop(columns=["coppaRisk"])
    y_train = train_set["coppaRisk"]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    numerical_columns = df_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df_train.select_dtypes(include=["object"]).columns.tolist()

    columns_to_drop = []

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                [col for col in numerical_columns if col not in columns_to_drop],
            ),
            (
                "cat",
                categorical_pipeline,
                [col for col in categorical_columns if col not in columns_to_drop],
            ),
            (
                "cat_high",
                high_card_pipeline,
                [
                    col
                    for col in categorical_columns
                    if X_train[col].nunique() >= 5 and col not in columns_to_drop
                ],
            ),
        ]
    )

    def create_train_pipeline(classifier):
        return ImbPipeline(
            [
                ("feature_engineer", EnhancedFeatureExtractor()),
                ("column_remover", ColumnRemover(columns_to_drop)),
                ("imputer", ImprovedImputer()),
                ("preprocessor", preprocessor),
                ("variance_selector", VarianceThreshold(threshold=0.01)),
                ("classifier", classifier),
            ]
        )

    search_spaces = {
        "classifier__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "classifier__n_estimators": Integer(200, 2000),
        "classifier__max_depth": Integer(3, 10),
        "classifier__num_leaves": Integer(20, 120),
        "classifier__min_child_weight": Integer(1, 10),
        "classifier__subsample": Real(0.5, 1.0),
        "classifier__colsample_bytree": Real(0.5, 1.0),
        "classifier__colsample_bylevel": Real(0.5, 1.0),
        "classifier__reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
        "classifier__reg_lambda": Real(1e-3, 10.0, prior="log-uniform"),
        "classifier__boosting_type": Categorical(["gbdt", "dart"]),
    }

    base_classifier = lgb.LGBMClassifier(
        random_state=42, verbose=-1, metric="auc", n_jobs=1
    )

    imb_pipeline = create_train_pipeline(base_classifier)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    bayes_search = BayesSearchCV(
        estimator=imb_pipeline,
        search_spaces=search_spaces,
        n_iter=10,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )

    bayes_search.fit(X_train, y_train)

    print(f"Best parameters: {bayes_search.best_params_}")
    print(f"Best CV score: {bayes_search.best_score_:.4f}")

    best_index = bayes_search.best_index_
    cv_results = bayes_search.cv_results_

    print("CV Results for Best Model:")
    print(
        f"Mean Test Score for Best Model: {cv_results['mean_test_score'][best_index]:.4f}"
    )
    print(
        f"Std Test Score for Best Model: {cv_results['std_test_score'][best_index]:.4f}"
    )
    print(
        f"Max Test Score for Best Model: {cv_results['mean_test_score'][best_index] + cv_results['std_test_score'][best_index]:.4f}"
    )

    best_pipeline = bayes_search.best_estimator_
    best_pipeline.fit(X_train, y_train)

    filename = "./model/best_lgbm_pipeline.pkl"
    joblib.dump(best_pipeline, filename)
    print(f"Model saved to {filename}")
