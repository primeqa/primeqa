# WTQ runs
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_0_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_1_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_2_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_3_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_4_.json"

# WikiSQL runs
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/gen_quest_g_0.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/gen_quest_g_1.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/gen_quest_g_2.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/gen_quest_g_3.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/gen_quest_g_4.json"

# WikiSQL runs
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wikisql_gen_quest_g_0_col-header_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wikisql_gen_quest_g_1_col-header_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wikisql_gen_quest_g_2_col-header_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wikisql_gen_quest_g_3_col-header_.json"
jbsub -require k80 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wikisql_gen_quest_g_4_col-header_.json"

# WTQ col-header runs
jbsub -require v100 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_0_col-header__beam-10.json"
jbsub -require v100 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_1_col-header__beam-10.json"
jbsub -require v100 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_2_col-header__beam-10.json"
jbsub -require v100 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_3_col-header__beam-10.json"
jbsub -require v100 -cores 1+1 -q x86_1h -mem 10g "source activate /dccstor/cmv/saneem/nlqTable/irl_git/t5-env/; python filter_generated_text.py -f data/generated_question/wtq_gen_quest_g_4_col-header__beam-10.json"
