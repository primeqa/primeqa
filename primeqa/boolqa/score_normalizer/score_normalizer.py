import pickle
import numpy
import os
import json
import importlib
from primeqa.boolqa.processors.dataset.mrc2dataset import create_dataset_from_run_mrc_output
import datasets
import argparse
import sys
from sklearn.linear_model import LogisticRegression
from sklearn import svm

class ScoreNormalizer(object):
    """
    Class for normalizeing the score for boolean and extractive questions.
    """

    def __init__(self, model_file_path=None, google_format=False):
        """
        Args:
            score_normalizer_model_path: Path of score normalizer model, a picke file.
        """
        self._model_file_path=model_file_path
        self._google_format=google_format
        

    def load_model(self):
        if not self._model_file_path:
            raise ValueError(f"No score normalizer model path was provided.")
        try:
            self._model = pickle.load(open(self._model_file_path, 'rb'))
        except Exception as ex:
            raise ValueError(f"Unable to load confidence model from {self._model_file_path}")
    
    def normalize_scores(self,input_file : str, output_dir : str,
                        qtc_is_boolean_label : str = 'boolean',
                        evc_no_answer_class : str = 'no_answer'):
        
        qa_pred_data = create_dataset_from_run_mrc_output(input_file, unpack=True)
        
        normalized_predictions=[]
        for i, qa_pred in enumerate(qa_pred_data):
           
            n = self.create_prediction(qtc_is_boolean_label, qa_pred)
            
            features = self.create_features(qtc_is_boolean_label, qa_pred)
            new_score = self._model.predict_proba(features)[0][1]
            
            n['confidence_score'] = float(new_score)
            
            normalized_predictions.append(n)

        # if the output directory does not exist, create a new directory 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save the normalized predictions
        with open(os.path.join(output_dir, 'eval_predictions_processed.json'), 'w') as f:
            json.dump(normalized_predictions, f, indent=4)

    def create_prediction(self, qtc_is_boolean_label, qa_pred):
        
        n={'example_id': qa_pred['example_id'],
            'start_position': qa_pred['span_answer_start_position'],
            'end_position': qa_pred['span_answer_end_position'],
            'passage_index': qa_pred['passage_index'],
            'yes_no_answer': qa_pred['yes_no_answer']
            }
            
        # Update the prediction to be YES/NO
        question_label = 1 if qa_pred['question_type_pred'] == qtc_is_boolean_label else 0
        if question_label == 1:
            yes_answer = qa_pred['boolean_answer_pred']
            if yes_answer == "yes": 
                n['yes_no_answer'] = 3
            else: 
                n['yes_no_answer'] =  4
            n['start_position'] = -1
            n['end_position'] = -1
            
        return n

    def create_features(self, qtc_is_boolean_label, qa_pred):
        # Apply the score normalizer
        # qa_conf_score = qa_pred['span_answer_score']
        # evc_conf_score = float(qa_pred['boolean_answer_scores'][evc_no_answer_class])
        b_score = qa_pred['start_logit']
        e_score = qa_pred['end_logit']
        na_score = qa_pred['target_type_logits'][0] if 'target_type_logits' in  qa_pred else 0.0
        question_label = 1 if qa_pred['question_type_pred'] == qtc_is_boolean_label else 0
        feature_list = [question_label,b_score,e_score,na_score]
        features = numpy.array(feature_list).reshape(1, -1)
        return features

    def train(self,input_file : str, 
            gold_file : str,
            output_dir : str,
            qtc_is_boolean_label : str = 'boolean',
            evc_no_answer_class : str = 'no_answer'):
        
        numpy.random.seed(42)
        
        qa_pred_data = create_dataset_from_run_mrc_output(input_file, unpack=True)
        dataset = datasets.load_dataset('json', data_files={'validation': gold_file})['validation']
        
        #YN_Gold = [] # is it is a YN question
        HSA_Gold = [] # if it has a SA 
        QSA_Scores = [] # the SA score for the QA SA prediction/ not currently used/ might want to experiment
        YN_Pred = [] #if it was predicted to be YN
        #EVC_Scores = [] # not used in this version
        B_Scores = []
        E_Scores = []
        NA_Scores = []

        gold_annotations = dataset['annotations']
        for i, qa_pred in enumerate(qa_pred_data):
            gold = gold_annotations[i]

            if self._google_format:
                yn_q_gold = gold[0]['yes_no_answer'] in ['YES','NO']
                ha_q_gold = gold[0]['minimal_answer']['plaintext_start_byte'] != -1 or yn_q_gold
            else:
                yn_q_gold = gold['yes_no_answer'][0] in ['YES','NO']
                ha_q_gold = gold['minimal_answers_start_byte'][0] != -1 or yn_q_gold
        
            HSA_Gold.append(ha_q_gold)
            
            b_score = qa_pred['start_logit']
            B_Scores.append(b_score)
            
            e_score = qa_pred['end_logit']
            E_Scores.append(e_score)
            
            na_score = qa_pred['target_type_logits'][0]
            NA_Scores.append(na_score)
            
            yn_pred = qa_pred['question_type_pred'] == qtc_is_boolean_label
            YN_Pred.append(int(yn_pred))
            
        features = numpy.array([YN_Pred, B_Scores, E_Scores, NA_Scores])
        features = numpy.transpose(features)
        labels = numpy.array([int(hsa) for hsa in HSA_Gold])

        #clf = LogisticRegression(random_state=0).fit(features, labels)
        #clf  = tree.DecisionTreeClassifier().fit(features, labels)
        clf = svm.SVC(probability=True).fit(features,labels)
        
        pickle.dump(clf, open(output_dir+"/score_normalizer_svm.pickle", 'wb'))
            