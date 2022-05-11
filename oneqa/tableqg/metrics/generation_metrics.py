import nltk
import numpy as np

def bleuscore(pred_list):
	hyp = [p['predictions'][0] for p in pred_list]
	ref = [p['quesion'] for p in pred_list]

	hyp = [h.lower().split(' ') for h in hyp]
	ref = [r.lower().split(' ') for r in ref]

	bleuscore = []
	for weights in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
		score_list = []
		for i in range(len(hyp)):
			score = nltk.translate.bleu_score.sentence_bleu([ref[i]], hyp[i], weights)
			score_list.append(score)
		bleuscore.append(np.mean(score_list))
	return bleuscore
