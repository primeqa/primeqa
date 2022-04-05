# experiments with different number of where clauses
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 1 -ia False"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 2 -ia False"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 3 -ia False"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 4 -ia False"

jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 1 -ia True"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 2 -ia True"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 3 -ia True"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 5g "source activate tabert; python t5_generation.py -nw 4 -ia True"

# group-wise training
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_0"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_1"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_2"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_4"

jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g all"

# Training with column headers
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_0 -col True"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_1 -col True"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_2 -col True"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_3 -col True"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_4 -col True"

jbsub -require k80 -cores 2+2 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_0"
jbsub -require k80 -cores 2+2 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_1"
jbsub -require k80 -cores 2+2 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_2"
jbsub -require k80 -cores 2+2 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3"
jbsub -require k80 -cores 2+2 -q x86_24h -mem 20g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_4"

# Inference
jbsub -require k80 -cores 1+1 -q x86_12h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_0"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_1"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_2"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_3"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_4"

jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 0"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 1"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 2"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 3"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 4"

jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 5"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 6"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 7"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 8"
jbsub -require k80 -cores 1+1 -q x86_6h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_3 -p 9"

jbsub -require k80 -cores 1+1 -q x86_12h -mem 15g "source activate tabert; python t5_generation.py -nw 4 -ia True -g g_1 -ns 30"

# QG for WTQ
jbsub -require k80 -cores 1+0 -q x86_6h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python sql_sampler.py -g g_0 -ns 30 -d wtq"
jbsub -require k80 -cores 1+0 -q x86_6h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python sql_sampler.py -g g_1 -ns 30 -d wtq"
jbsub -require k80 -cores 1+0 -q x86_6h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python sql_sampler.py -g g_2 -ns 30 -d wtq"
jbsub -require k80 -cores 1+0 -q x86_6h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python sql_sampler.py -g g_3 -ns 30 -d wtq"
jbsub -require k80 -cores 1+0 -q x86_6h -mem 20g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python sql_sampler.py -g g_4 -ns 30 -d wtq"

jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_0 -ns 30 -d wtq -suf _beam-10"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_1 -ns 30 -d wtq -suf _beam-10"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_2 -ns 30 -d wtq -suf _beam-10"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_3 -ns 30 -d wtq -suf _beam-10"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -nw 4 -ia True -g g_4 -ns 30 -d wtq -suf _beam-10"

# QG WTQ with column headers
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -g g_0 -ns 30 -d wtq"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -g g_1 -ns 30 -d wtq"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -g g_2 -ns 30 -d wtq"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -g g_3 -ns 30 -d wtq"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -g g_4 -ns 30 -d wtq"

# QG WTQ with column headers and beam 10
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -ns 30 -d wtq -suf _beam-10 -g g_0"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -ns 30 -d wtq -suf _beam-10 -g g_1"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -ns 30 -d wtq -suf _beam-10 -g g_2"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -ns 30 -d wtq -suf _beam-10 -g g_3"
jbsub -require k80 -cores 1+1 -q x86_12h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -col True -nw 4 -ia True -ns 30 -d wtq -suf _beam-10 -g g_4"

# QG for WikiSQL column headers
jbsub -require k80 -cores 1+1 -q x86_24h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -col True -g g_0"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -ns 30 -d wikisql -col True -g g_1"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -col True -g g_2"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -col True -g g_3"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -col True -g g_4"
    
# QG for WikiSQL column headers with beam 10
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -suf _beam-10 -col True -g g_0"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -ns 30 -d wikisql -suf _beam-10 -col True -g g_1"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -suf _beam-10 -col True -g g_2"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -suf _beam-10 -col True -g g_3"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -ia True -d wikisql -suf _beam-10 -col True -g g_4"

# WikiSQL dev
jbsub -require k80 -cores 2+2 -q x86_6h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -s dev -g g_0"
jbsub -require k80 -cores 2+2 -q x86_6h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -s dev -g g_1"
jbsub -require k80 -cores 2+2 -q x86_6h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -s dev -g g_2"
jbsub -require k80 -cores 2+2 -q x86_6h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -s dev -g g_3"
jbsub -require k80 -cores 2+2 -q x86_6h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -s dev -g g_4"



# QG training on finetuned T5 models
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_0 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_0/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_1 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_1/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_2 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_2/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_3 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_3/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_4 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_4/pytorch_model_epoch_3.bin"

jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_0 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_0/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_1 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_1/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_2 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_2/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_3 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_3/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_4 -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_4/pytorch_model_epoch_2.bin"

# QG training on finetuned T5 models with columns
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_0 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_0/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_1 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_1/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_2 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_2/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_3 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_3/pytorch_model_epoch_3.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_4 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_4/pytorch_model_epoch_3.bin"

jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_0 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_0/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_1 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_1/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_2 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_2/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_3 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_3/pytorch_model_epoch_2.bin"
jbsub -require k80 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -d wikisql -ia True -g g_4 -col True -t5 /dccstor/cssblr/samarth/nlqTable-t5-domainspecific/wkisql/g_4/pytorch_model_epoch_2.bin"

# Inference QG on finetuned T5 models
# 3 Epoch finetuning
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_0 -m t5_FT-wkisql-g_0-epoch_3_nw-4_if-agg-True_group-g_0"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_1 -ns 30 -m t5_FT-wkisql-g_1-epoch_3_nw-4_if-agg-True_group-g_1"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_2 -m t5_FT-wkisql-g_2-epoch_3_nw-4_if-agg-True_group-g_2"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_3 -m t5_FT-wkisql-g_3-epoch_3_nw-4_if-agg-True_group-g_3"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_4 -m t5_FT-wkisql-g_4-epoch_3_nw-4_if-agg-True_group-g_4"

# 2 epoch FT
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_0 -m t5_FT-wkisql-g_0-epoch_2_nw-4_if-agg-True_group-g_0"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_1 -ns 30 -m t5_FT-wkisql-g_1-epoch_2_nw-4_if-agg-True_group-g_1"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_2 -m t5_FT-wkisql-g_2-epoch_2_nw-4_if-agg-True_group-g_2"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_3 -m t5_FT-wkisql-g_3-epoch_2_nw-4_if-agg-True_group-g_3"
jbsub -require v100 -cores 1+1 -q x86_24h -mem 15g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env; python t5_generation.py -t genq -d wikisql -ia True -g g_4 -m t5_FT-wkisql-g_4-epoch_2_nw-4_if-agg-True_group-g_4"
