from dataclasses import dataclass
from typing import Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

@dataclass
class ModelZoo:
    params: Dict[str, Any]

    def logreg(self):
        return LogisticRegression(C=self.params.get("C",1.0), max_iter=self.params.get("max_iter",1000))

    def tree(self):
        return DecisionTreeClassifier(
            max_depth=self.params.get("max_depth",5),
            min_samples_leaf=self.params.get("min_samples_leaf",50)
        )

    def xgb(self):
        p = self.params
        return XGBClassifier(
            n_estimators=p.get("n_estimators",300),
            max_depth=p.get("max_depth",4),
            learning_rate=p.get("learning_rate",0.05),
            subsample=p.get("subsample",0.9),
            colsample_bytree=p.get("colsample_bytree",0.9),
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1
        )
