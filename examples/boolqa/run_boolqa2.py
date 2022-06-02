#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import logging
import subprocess
from pathlib import Path



logger=logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def systemx(cmd):
    logger.info(cmd)
    try:
        completed_process=subprocess.run(cmd, shell=True, capture_output=True, encoding='utf-8')
        completed_process.check_returncode()
    except subprocess.CalledProcessError as x:
        logger.info(f'caught CalledProcessError: {x}')
        logger.info('exiting')
        sys.exit(1)
    return completed_process # members: args returncode stdout stderr

#----------------------------------------------------------------

def handle_args():
    usage='usage'
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--mrcmodel', type=str, 
                        help='path to directory contain pytorch_model.bin for mrc')
    #parser.add_argument('-mrc_file', '-m', required=True, type=str,
    #                    help='output json file from the mrc system')
    parser.add_argument('--tempdir', '-t', required=True,
                        help='temporary workspace')
    parser.add_argument('--outdir', '-o', required=True)
    parser.add_argument('--qtcmodel', type=str, required=True,
                        help='path to directory containing pytorch_model.bin for query type classification')
    parser.add_argument('--evcmodel', type=str, required=True,
                        help='path to directory containing pytorch_model.bin for boolean answer classification')
    parser.add_argument('--snmodel', type=str, required=True,
                        help='path to score normalizer pickle model file')
    args = parser.parse_args()

    return args

#----------------------------------------------------------------

def main():
    args=handle_args()


    ws=args.tempdir
    outdir=args.outdir
    Path(ws).mkdir(parents=True, exist_ok=True)
    #mrcfile=args.mrc_file
    mrcmodel=args.mrcmodel
    mrcfile=f'{ws}/mrc/eval_predictions.json'
    mrcfile_processed=mrcfile[:-5]+'_processed.json'
    qtc_prediction_file=f'{ws}/qtc/predict_results_qtc.json'
    evc_prediction_file=f'{ws}/evc/predict_results_evc.json'
    qtcmodel=args.qtcmodel
    evcmodel=args.evcmodel
    #merge_prediction_file=f'{ws}/merge/eval_predictions_merge.json'
    merge_prediction_file=f'{outdir}/eval_predictions_merge.json'
    eval_file=f'{outdir}/eval.out'
    sn_model_file=args.snmodel
    force_mrc=False
    force_qtc=False
    force_evc=False
    force_merge=False
    force_eval=False


    if force_mrc or not Path(mrcfile).is_file():
        cmd=f'''
            python examples/mrc/run_mrc.py --model_name_or_path {mrcmodel} \
        --output_dir {ws}/mrc/ --fp16 --learning_rate 4e-5 \
        --do_eval --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
        --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
        --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no \
        --postprocessor oneqa.boolqa.processors.postprocessors.extractive.ExtractivePipelinePostProcessor
        '''
#        --max_eval_samples 100 \
        systemx(cmd)
        #print(cmd)





    if force_qtc or not Path(qtc_prediction_file).is_file():
        cmd = f'''
    python examples/boolqa/run_nway_classifier_1qa.py \
    --task_name qtc \
    --overwrite_cache \
    --model_name_or_path {qtcmodel} \
    --test_file {mrcfile} \
    --output_dir {ws}/qtc
    '''
        systemx(cmd)

    if force_evc or not Path(evc_prediction_file).is_file():
        cmd=f'''
    python examples/boolqa/run_nway_classifier_1qa.py \
    --task_name evc \
    --overwrite_cache \
    --drop_label no_answer \
    --model_name_or_path {evcmodel} \
    --test_file {ws}/qtc/eval_predictions.json \
    --output_dir {ws}/evc
    '''
        systemx(cmd)


#
# run the merger
# TODO add evc, clf to command
#
    #Path(f'{ws}/merge').mkdir(parents=True, exist_ok=True)
    Path(f'{outdir}').mkdir(parents=True, exist_ok=True)
    if force_merge or not Path(merge_prediction_file).is_file():
        cmd=f'''
    python examples/boolqa/merger_simple.py \
    --answer_predictions_file {ws}/evc/eval_predictions.json \
    --sn_model_file {sn_model_file} \
    --output_predictions_file {merge_prediction_file}
        '''
        systemx(cmd)

    
    if force_eval or not Path(eval_file).is_file():
        cmd=f'''
        python  ../OneQA.bool_framewok/oneqa/mrc/tydi_eval.py \
            --qa_predictions_file {merge_prediction_file}
        '''
        eval=systemx(cmd).stdout
        with open(eval_file,'wt') as eval_out:
            eval_out.write(eval)

        logger.info(eval.split('\n')[-2])

#-------------------------------------------------------------------        
# do main
if __name__=='__main__':
   main()