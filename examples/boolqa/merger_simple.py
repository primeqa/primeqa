import pandas as pd
import json
import numpy
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import argparse
from mrc2dataset import create_dataset_from_run_mrc_output

def parse_arguments():
    parser = argparse.ArgumentParser(description='Assigns YES/NO answers to the QA prediction file based on the question boolean classifier')
    parser.add_argument('--answer_predictions_file',  
                    help='the prediction file produced by the boolean answer classifier',
                    type=str)
    parser.add_argument('--sn_model_file',  
                    help='the model for the score normalizer',
                    type=str)        
    parser.add_argument('--output_predictions_file',  
                    help='the output prediction files with YES/NO normalized answers',
                    type=str)

    args = parser.parse_args()
    return args


def main():
    
    args = parse_arguments()

    if args.sn_model_file and args.sn_model_file!='NONE':
        clf = pickle.load(open(args.sn_model_file, 'rb')) 
        print(clf.coef_)
    else:
        clf = None



    qa_pred_data = create_dataset_from_run_mrc_output(args.answer_predictions_file, unpack=True)
        
    cnt = 0
    bq = 0
    yq = 0
    nq = 0
    merged=[]
    for i, qa_pred in enumerate(qa_pred_data):
        m={'example_id': qa_pred['example_id'],
           'start_position': qa_pred['span_answer_start_position'],
           'end_position': qa_pred['span_answer_end_position'],
           'passage_index': qa_pred['passage_index'],
           'yes_no_answer': qa_pred['yes_no_answer']
        }
        cnt += 1
        question_label = 1 if qa_pred['question_type_pred'] == "YN" else 0
        # Update the prediction to be YES/NO
        if question_label == 1:
            bq += 1
            yes_answer = qa_pred['boolean_answer_pred']
            if yes_answer == 'True': 
                yq += 1
                m['yes_no_answer'] = 3
            else: 
                nq += 1
                m['yes_no_answer'] =  4
            m['start_position'] = -1
            m['end_position'] = -1
        # Apply the score normalizer
        feature_list = [question_label,qa_pred['span_answer_score']]
        features = numpy.array(feature_list).reshape(1, -1)
        new_score = clf.predict_proba(features)[0][1] if clf is not None else qa_pred['span_answer_score']
        m['confidence_score'] = float(new_score)
        #m['qt_prediction'] = question_label
        merged.append(m)


    with open(args.output_predictions_file, 'w') as json_file:
        json.dump(merged, json_file, indent=4)
    print("CNT",cnt,bq,yq,nq)


if __name__ == "__main__":
    main()