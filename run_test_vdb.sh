checkError() {
  # Exitcode has to be saved first or else shift clobbers it
  EXIT_CODE=$?
  ERR_TITLE=$1
  ERR_DESCP=$2
  ERR_CMD=$3
  FILE_TO_CHECK=$4
  CMD_TO_RUN=$5

  if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: $ERR_TITLE $ERR_DESCP"

    if [ "$JSON_LOG_FILE" != "" ]; then
      current_time=$(date --rfc-3339=seconds)
      current_time=$(printf %q "$current_time")

      if [ "$ERR_TITLE" != "" ]; then
        ERR_TITLE=$(printf %q "$ERR_TITLE")
      fi
      if [ "$ERR_DESCP" != "" ]; then
        ERR_DESCP=$(printf %q "$ERR_DESCP")
      fi

      echo "{\"timestamp\":\"$current_time\", " >>$JSON_LOG_FILE
      echo " \"error\": \"$ERR_TITLE\", " >>$JSON_LOG_FILE
      echo " \"description\":\"$ERR_DESCP\", " >>$JSON_LOG_FILE
      echo " \"command\":\"$ERR_CMD\"}" >>$JSON_LOG_FILE
      echo " \"code\":\"$EXIT_CODE\"}" >>$JSON_LOG_FILE
    fi

    if [ "$FILE_TO_CHECK" != "" ]; then
      if [ ! -e $FILE_TO_CHECK ]; then
        echo "The file $FILE_TO_CHECK does not exist - want to run the command '$CMD_TO_RUN' ([Y]es/[n]o)?"
        while true; do
          read -t 2 response
          if [ "$response" == "y" -o "$response" = "Y" -o "$response" = "yes" -o "$response" = "Yes" -o "$response" = "" ]; then
            eval $CMD_TO_RUN
            break
          elif [ "$response" = "n" -o "$response" = "no" ]; then
            break
          else
            echo "Please answer with 'y' or 'yes' or 'n' or 'no'"
          fi
        done
      fi
    fi
    exit $EXIT_CODE
  fi
}

# run a command and check its return code
runCmd() {
  echo "================================================================"
  printf "Running \e[38;5;87m %s \e[0m" "$1"
  echo
  # echo Running "\e[38;5;87m" "$1" "\e[0m"
  time eval $1
  EXIT_CODE=$?

  if [ $EXIT_CODE -ne 0 ]; then
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
    echo "failed running $1"
    echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
  fi
  echo "================================================================"

  return $EXIT_CODE
}

function join_by { local IFS="$1"; shift; echo "$*"; }

echo "$0 $*" >>run_cmmds

export PATH=/home/raduf/anaconda3/bin:$PATH

#env CUDA_VISIBLE_DEVICES=1 python primeqa/ir/scripts/test_vectordb.py --db_engine chromadb
#  --data benchmark/rh --input_queries benchmark/rh/questions.tsv
#  --input_passages benchmark/rh/rhel_split_issues_unescaped.tsv --top_k 10000 --actions cir --evaluate
#  --ranks 1,5,10,40,10000 --model all-MiniLM-L6-v2 -o new-results/rh-all-MiniLM-L6-v2-nonorm-10k-unescaped
#  --create_own_embeddings --ingestion_batch_size 256

fp16=0
fp16_opt_level=O2
model=all-MiniLM-L6-v2
data=""
queries=""
passages=""


SHORT_FLAGS="-o s:e:o:M:C:dh"
lng_flags=(
  "fp16"
  "fp16_opt_level:"
  "fp32"
  "model:"
  "data:"
  "queries:"
  "passages:"
  "index:"
  "save_index:"
  "output:"
  "debug"
  "devices:"
  "top_k"
  "create_own_embeddings"
  "ingestion_batch_size"
  "actions:"
  "evaluate"
  "ranks:"
)
#LONG_FLAGS="--long fp16_opt_level:,seed:,train_batch_size:,eval_batch_size:,gradient_accumulation_steps:,loss_scale:,num_training_epochs:,output:,max_seq_length:,bert_model:,train_file:,dev_file:,pretrained_model:,learning_rate:,dataset:,lr:,debug,sent_stride:,full_seq_overlap:,test_only,distributed:,devices:"
LONG_FLAGS="--long "$(join_by "," ${lng_flags[*]})
echo "LONG_FLAGS value is ${LONG_FLAGS}"

TEMP=$(getopt $SHORT_FLAGS $LONG_FLAGS -n 'run_random_seed.sh' -- "$@")

eval set -- "$TEMP"

while true; do
  echo "Processing flag $1, with opt arg $2"
  case "$1" in
  -s | --seed)  seed="$2"; shift 2;;
  -d | --debug) debug=1; set -x; shift;;
  --per_gpu_train_batch_size) per_gpu_train_batch_size=$2; shift 2;;
  --per_gpu_eval_batch_size) per_gpu_eval_batch_size=$2; shift 2;;
  --train_batch_size) train_batch_size="$2"; shift 2;;
  --eval_batch_size)  eval_batch_size="$2"; shift 2;;
  --train_file)       train_file=$2; shift 2;;
  --dev_file)         dev_file=$2; shift 2;;
  --gradient_accumulation_steps)  gradient_accumulation_steps="$2"; shift 2;;
  --loss_scale)    loss_scale="$2"; shift 2;;
  -o | --output)  output="$2"; shift 2;;
  -e | --num_training_epochs) epochs="$2"; shift 2;;
  -M | --max_seq_length)      max_seq_length="$2"; shift 2;;
  --model_type)               bert_model="$2"; shift 2;;
  --model_name)               model_name=$2; shift 2;;
  --pretrained_model)         pretrained_model="--pretrained_model $2"; shift 2;;
  --learning_rate | --lr)     lr=$2; shift 2;;
  --dataset) dataset=$2; shift 2;;
  --data_dir)            data_dir="$2"; shift 2;;
  --use_full_seq)        use_full_seq=1; shift;;
  --sent_stride | --full_seq_overlap)    sent_stride=$2; use_full_seq=1; shift 2;;
  --test_only)    test_only=1; shift;;
  --fp32)             fp16=0; shirt;;
  --fp16)             fp16=1; shift;;
  --fp16_opt_level)   fp16_opt_level=$2; fp16=1; shift 2;;
  --distributed)      export CUDA_VISIBLE_DEVICES=$2; numgpu=$(echo $2 | tr ',' ' ' | wc -w); dgpus="-m torch.distributed.launch --nproc_per_node ${numgpu}"; shift 2;;
  --devices)          export CUDA_VISIBLE_DEVICES=$2; shift 2;;
  --task_name)        task_name=$2; shift 2;;
  --overwrite_output_dir) overwrite_output_dir=1; shift;;
  --evaluate_during_training) evaluate_during_training=$2; shift 2;; # --evaluate_during_training 0 turns off evaluation during training
  -h)                 echo "$SHORT_FLAGS $LONG_FLAGS"; quit=1; break;;
  --)    shift; break;;
  *)     break ;;
  esac
done
