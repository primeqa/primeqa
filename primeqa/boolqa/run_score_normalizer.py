from ast import Raise
from xmlrpc.client import boolean
from primeqa.boolqa.score_normalizer.score_normalizer import ScoreNormalizer
import argparse
import sys

def main(args):
    if len(args) == 1:
        args = argparse.Namespace(**(args[0]))
    else:
        args = parse_arguments()
        #args = args[0]
    if args.do_apply:
        sn = ScoreNormalizer(args.model_name_or_path)
        sn.load_model()
        sn.normalize_scores(args.test_file, 
                        args.output_dir,
                        args.qtc_is_boolean_label, 
                        args.evc_no_answer_class)
    elif args.do_train:
        sn = ScoreNormalizer(google_format=args.google_format)
        sn.train(args.train_file,
                 args.gold_file,
                 args.output_dir,
                 args.qtc_is_boolean_label, 
                 args.evc_no_answer_class)
    else:
        pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Assigns YES/NO answers to the QA prediction file based on the question boolean classifier')
    parser.add_argument('--do_apply',  
                    help='Apply the score normalizer',
                    action='store_true') 
    parser.add_argument('--test_file',  
                    help='the prediction file produced by the boolean answer classifier',
                    type=str)
    parser.add_argument('--model_name_or_path',  
                    help='the model for the score normalizer',
                    type=str) 
    parser.add_argument('--do_train',  
                    help='train the score normalizer',
                    action='store_true')
    parser.add_argument('--train_file',  
                    help='the train file produced by the boolean answer classifier',
                    type=str) 
    parser.add_argument('--gold_file',  
                    help='the original train file with the QA annotations',
                    type=str)      
    parser.add_argument('--output_dir',  
                    help='the output prediction files with YES/NO normalized answers',
                    type=str)  
    parser.add_argument('--qtc_is_boolean_label', type=str, default='boolean',
                    help='the value assigned to the question_type_pred field for boolean questions')
    parser.add_argument('--evc_no_answer_class', type=str, default='no_answer',
                    help='the class label in the boolean_answer_scores field for no_answer questions')
    parser.add_argument('--google_format', action='store_true',
                    help='Use Google TyDi format instead of HF TyDi format')
    
    args = parser.parse_args()
    return args
  
if __name__ == '__main__':
    main(sys.argv)