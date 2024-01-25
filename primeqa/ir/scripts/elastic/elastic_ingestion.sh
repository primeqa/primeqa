#!/bin/bash
set -x
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

      echo "{\"timestamp\":\"$current_time\" " >>$JSON_LOG_FILE
      echo " \"error\": \"$ERR_TITLE\" " >>$JSON_LOG_FILE
      echo " \"description\":\"$ERR_DESCP\" " >>$JSON_LOG_FILE
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

checkFile() {
  if [ ! -e $1 ]; then
    echo "The file $1 does not exist!"
    exit 11
  fi
}

function join_by { local IFS="$1"; shift; echo "$*"; }

echo "$0 $*" >>run_cmmds

export AILANG_PASSWORD="EnGRdKDNGBC1jPE+ZFxa"
export AILANG_API_KEY="dmliRlNvd0J0dnEwNW1YTFF0U1Q6eHNBLUtkaHNSQUNwVUJLSWxiOVk2dw=="
export AILANG_SSL_FINGERPRINT="ffa7f5456ce6388697d7bbdf17df27910813101da7cfa25cb256490f9ee5abfb"

data_dir=/dccstor/awasthyp3/awasthyp/sap

lang=en
db_engine=es-dense
passages=
questions=
model_name=intfloat/multilingual-e5-base
index_name=
actions=ir  # default: ingest (i), retrieve(r)
ingestion_batch_size=40
max_doc_length=512
stride=256
output="benchmark/nlp-sap-e5-${lang}-512t-256t-2024-01-23.json"
k=40
ranks=1,3,5,10,40
server=AILANG
evaluate=1
data_type=sap
docid_map=${data_dir}/${lang}_docid.tsv
hana_file2url=${data_dir}/${lang}_hana_file2url.tsv
date=$(date +"%m-%d-%Y")
replace_links=0
doc_based=1

SHORT_FLAGS="-o p:q:o:I:A:k:l:r:h"
lng_flags=(
  "passages:"
  "questions:"
  "db_engine:"
  "model_name:"
  "index_name:"
  "actions:"
  "ingestion_batch_size:"
  "max_doc_length:"
  "stride:"
  "output:"
  "ranks:"
  "server:"
  "docid_map:"
  "doc_based"
  "paragraph_based:"
  "language_code:"
  "hana_file2url:"
  "top_k:"
  "evaluate"
  "replace_links"
)

LONG_FLAGS="--long "$(join_by "," ${lng_flags[*]})
echo "LONG_FLAGS value is ${LONG_FLAGS}"

quit=0

TEMP=$(getopt $SHORT_FLAGS $LONG_FLAGS -n 'run_exp.sh' -- "$@")

eval set -- "$TEMP"

while true; do
  echo "Processing flag $1, with opt arg $2"
  case "$1" in
  -d | --db_engine) db_engine=$2; shift 2;;
  -m | --model_name) model_name=$2; shift 2;;
  -o | --output) output=$2; shift 2;;
  --actions) actions=$2; shift 2;;
  --ingestion_batch_size) ingestion_batch_size=$2; shift 2;;
  -r | --ranks)  ranks=$2; shift 2;;
  -p | --passages) passages=$2; shift 2;;
  -q | --questions) questions=$2; shift 2;;
  -k | --top_k) k=$2; shift 2;;
  --max_doc_length) max_doc_length=$2; shift 2;;
  -S | --server) server=$2; shift 2;;
  --docid_map) docid_map=$2; shift 2;;
  -l | --language_code) lang=$2; shift 2;;
  --evaluate) evaluate=1; shift;;
  --doc_based) doc_based=1; shift;;
  --paragraph_based) doc_based=0; shift;;
  --hana_file2url) hana_file2url=$2; shift;;
  -I | --index_name) index_name=$2; shift;;
  -h)                 echo "$SHORT_FLAGS $LONG_FLAGS"; quit=1; break;;
  --)    shift; break;;
  *)     break ;;
  esac
done

if [ "$quit" == 1 ]; then
  exit
fi

if [ "$passages" == "" ]; then
  passages=/u/raduf/sandbox2/5lang_docs/sap-${lang}.jsonl
fi

if [ "$questions" == "" ]; then
  questions=/dccstor/jlquinn-mt/for-parul/benchmark_v2.txt.sap-finetuned.en${lang}.csv
fi

if [ "$index_name" == "" ]; then
  index_name="nlp-sap-${model_name}-${lang}-${max_doc_length}t-${stride}t-${date}"
  index_name=${index_name/\//_}
fi

cmd=(
  "python primeqa/ir/scripts/elastic_ingestion.py"
  "-p ${passages}"
  "-q ${questions}"
  "--db_engine ${db_engine}"
  "--model_name ${model_name}"
  "--index ${index_name}"
  "--actions ${actions}"
  "--ingestion_batch_size ${ingestion_batch_size}"
  "--max_doc_length ${max_doc_length}"
  "--stride ${stride}"
  "-o ${output}"
  "-k ${k}"
  "--server ${server}"
  "--docid_map ${docid_map}"
  "--language_code ${lang}"
  "--hana_file2url ${hana_file2url}"
)

if [ "$ranks" != "" ]; then
  cmd+=("--ranks ${ranks}")
fi

if [ "$evaluate" == 1 ]; then
  cmd+=("--evaluate")
fi

if [ "$doc_based" == 1 ]; then
  cmd+=("--doc_based")
fi

if [ "$replace_links" == 1 ]; then
  cmd+=("--replace_links")
fi

if [ "$data_type" != "" ]; then
  cmd+=("--data_type ${data_type}")
fi

runCmd "${cmd[*]}"

cat "${output%.*}.metrics"
