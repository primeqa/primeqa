import pickle
import numpy
import os
import json
import importlib
from examples.boolqa.mrc2dataset import create_dataset_from_run_mrc_output
import argparse
import sys

class ScoreNormalizer(object):
    """
    Class for normalizeing the score for boolean and extractive questions.
    """

    def __init__(self, model_file_path=None):
        """
        Args:
            score_normalizer_model_path: Path of score normalizer model, a picke file.
        """
        if not model_file_path:
            raise ValueError(f"No score normalizer model path was provided: {model_file_path}")
        try:
            self._model = pickle.load(open(model_file_path, 'rb'))
        except Exception as ex:
            raise ValueError(f"Unable to load confidence model from {model_file_path}")
    
    def normalize_scores(self,input_file, output_dir):
        qa_pred_data = create_dataset_from_run_mrc_output(input_file, unpack=True)
        
        normalized_predictions=[]
        for i, qa_pred in enumerate(qa_pred_data):
            n={'example_id': qa_pred['example_id'],
            'start_position': qa_pred['span_answer_start_position'],
            'end_position': qa_pred['span_answer_end_position'],
            'passage_index': qa_pred['passage_index'],
            'yes_no_answer': qa_pred['yes_no_answer']
            }
        
            # Apply the score normalizer
            qa_conf_score = qa_pred['span_answer_score']
            evc_conf_score = float(qa_pred['boolean_answer_scores']['NONE'])
            b_score = qa_pred['start_logit']
            e_score = qa_pred['end_logit']
            na_score = qa_pred['target_type_logits'][0] if 'target_type_logits' in  qa_pred else 0.0
            question_label = 1 if qa_pred['question_type_pred'] == "YN" else 0
            feature_list = [question_label,b_score,e_score,na_score]
            features = numpy.array(feature_list).reshape(1, -1)
            new_score = self._model.predict_proba(features)[0][1]
            n['confidence_score'] = float(new_score)
            
            # Update the prediction to be YES/NO
            if question_label == 1:
                yes_answer = qa_pred['boolean_answer_pred']
                if yes_answer == 'True': 
                    n['yes_no_answer'] = 3
                else: 
                    n['yes_no_answer'] =  4
                n['start_position'] = -1
                n['end_position'] = -1
         
            normalized_predictions.append(n)

        # if the output directory does not exist, create a new directory 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save the normalized predictions
        with open(os.path.join(output_dir, 'eval_predictions_processed.json'), 'w') as f:
            json.dump(normalized_predictions, f, indent=4)

def main(args):
    if len(args) == 1:
        args = argparse.Namespace(**(args[0]))
    else:
        args = parse_arguments()
        #args = args[0]
    sn = ScoreNormalizer(args.model_name_or_path)
    sn.normalize_scores(args.test_file, args.output_dir)
  
def parse_arguments():
    parser = argparse.ArgumentParser(description='Assigns YES/NO answers to the QA prediction file based on the question boolean classifier')
    parser.add_argument('--test_file',  
                    help='the prediction file produced by the boolean answer classifier',
                    type=str)
    parser.add_argument('--model_name_or_path',  
                    help='the model for the score normalizer',
                    type=str)        
    parser.add_argument('--output_dir',  
                    help='the output prediction files with YES/NO normalized answers',
                    type=str)  
    args = parser.parse_args()
    return args
  
if __name__ == '__main__':
    main(sys.argv)