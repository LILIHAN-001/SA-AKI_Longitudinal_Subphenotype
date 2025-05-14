import pandas as pd
import shap
import os
from collections import Counter
#os.environ['RAY_memory_monitor_refresh_ms'] = "0"
os.environ['RAY_memory_usage_threshold'] = "0.99"
from sklearn.metrics import confusion_matrix
from autogluon.tabular import TabularDataset, TabularPredictor
save_path = './Result-b21a1234_raw_dataset_MimiceICU_AUMC_CorrMICfilt/'  # raw_dataset

label = 'groupHPD'#
metric = "roc_auc_ovo_macro" #'accuracy',"f1_macro"
train_set = pd.read_csv(save_path + "/input/train_set.csv")
train_set = train_set.drop(columns=["stay_id"])

X_train = train_set.drop(columns=[label]) 
y_train = train_set[label]


test_set1 = pd.read_csv(save_path + "./input/test_set1.csv")
test_set1 = test_set1.drop(columns=["stay_id"])
y_test = test_set1[label]
X_test = test_set1.drop(columns=[label]) 

predictor_multi = TabularPredictor.load(save_path)  # unnecessary, just demonstrates how to load previously-trained predictor from file
#model_fi = predictor_multi.feature_importance(train_set) # ,model=

class AutogluonWrapper:
    def __init__(self, predictor, feature_names, target_class=None):
        self.ag_model = predictor
        self.feature_names = feature_names
        self.target_class = target_class
        if target_class is None and predictor.problem_type != 'regression':
            print("Since target_class not specified, SHAP will explain predictions for each class")
    
    def predict_proba(self, X):
        if isinstance(X, pd.Series):
            X = X.values.reshape(1,-1)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        preds = self.ag_model.predict_proba(X)
        if predictor.problem_type == "regression" or self.target_class is None:
            return preds
        else:
            return preds[self.target_class] 


baseline = X_train.sample(1000, random_state=30)

ag_wrapper = AutogluonWrapper(predictor_multi, X_train.columns)
explainer = shap.KernelExplainer(ag_wrapper.predict_proba, baseline)
print("Baseline prediction: \n", ag_wrapper.predict_proba(baseline).mean())  # this is the same as explainer.expected_value

NSHAP_SAMPLES = 500  # how many samples to use to approximate each Shapely value, larger values will be slower

shap_values = explainer.shap_values(X_test, nsamples=NSHAP_SAMPLES,model="CatBoost_BAG_L2")
shap_values.to_csv(save_path + "/result/result_test1_shap_all.csv")

