from primeqa.boolqa.score_normalizer.score_normalizer import ScoreNormalizer
import argparse
import sys

def main(args):
    if len(args) == 1:
        args = argparse.Namespace(**(args[0]))
    else:
        args = parse_arguments()
        #args = args[0]
    sn = ScoreNormalizer(args.model_name_or_path)
    sn.load_model()
    sn.normalize_scores(args.test_file, 
                        args.output_dir,
                        args.qtc_is_boolean_label, 
                        args.evc_no_answer_class)

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
    parser.add_argument('--qtc_is_boolean_label', type=str, default='boolean',
                    help='the value assigned to the question_type_pred field for boolean questions')
    parser.add_argument('--evc_no_answer_class', type=str, default='no_answer',
                    help='the class label in the boolean_answer_scores field for no_answer questions')
    
    args = parser.parse_args()
    return args
  
if __name__ == '__main__':
    main(sys.argv)