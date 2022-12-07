import numpy as np
import torch
from torch.utils.data import DataLoader
from primeqa.hybridqa.processors.dataset import TableQADatasetQRSconcat
import pdb

def partial_label_data_loader(data, tokenized_data, batch_size, pos_fraction=0.001, group_fraction=1.0):

	# data is the original data with all question ids and extra info. 
	# tokenized_data is the one after bert tokenization
	batch_list, group_id_list = _create_batches_ids(data, batch_size, pos_fraction, group_fraction)
	batch_matrix_list = _create_batch_matrix(group_id_list)
	# chaning the order in train_dataset so that when shuffle=False and DataLoader is used it gives the batches as we want.
	pl_tokenized_data = []
	for batch in batch_list:
		for idx in batch:
			pl_tokenized_data.append(tokenized_data[idx])
	pl_data_loader = DataLoader(pl_tokenized_data, batch_size=batch_size,shuffle=False, 
										batch_sampler=None, pin_memory=True)
	return pl_data_loader, batch_matrix_list

def _create_batches_ids(data, batch_size, positive_fraction, group_fraction):
	# positive_fraction: what fraction of dataset should have positives. 
	#                    All positives are kept in training, negatives are sampled.
	# group_fraction: what fraction of ambiguous positives to be added to dataset. 
	#                 1.0 means all group positive instances to be used.

	# We will use question id to identify instances belonging to same questions and group all the positive labels.

	positive_question_dict = {} # a dict with all positive labelled rows per question
	negatives_list = [] # ids of all negative labelled rows which need not be be in groups
	for i,d in enumerate(data):
		qid = d['question_id']
		if d['label_new'] == 1:
			if qid not in positive_question_dict:
				positive_question_dict[qid] = [i]
			else:
				positive_question_dict[qid].append(i)
		else:
			negatives_list.append(i)

	# Finding positive instances with only one correct row
	singleton_positive_list = []
	qid_list = list(positive_question_dict.keys())
	for q in qid_list:
		if len(positive_question_dict[q]) == 1:
			singleton_positive_list.append(positive_question_dict[q][0])
			positive_question_dict.pop(q)

	# Identifying number of instances to take from each of sources. (singleton positives are always taken full)
	num_psingle = len(singleton_positive_list)
	total_pgroup = sum([len(positive_question_dict[x]) for x in positive_question_dict])
	num_pgroup = round(group_fraction * total_pgroup)

	num_negatives = round((1/positive_fraction - 1)*(num_psingle + num_pgroup))
	num_negatives = min(num_negatives, len(negatives_list))

	num_batches = int((num_psingle+num_pgroup+num_negatives)/batch_size)

	# creating batches
	pointer = 0
	batch_list = [[] for i in range(num_batches)]
	group_id_list = [[] for i in range(num_batches)]

	# adding positive groups to batches:  one group to a batch at a time
	pgroup_keys = np.random.permutation(list(positive_question_dict.keys()))
	count_pg = 0
	for gid,q in enumerate(pgroup_keys):
		inst_ids = positive_question_dict[q]
		if len(inst_ids) <= batch_size:
			while batch_size - len(batch_list[pointer]) < len(inst_ids):
				pointer += 1
				if pointer >= num_batches:
					pointer = pointer % num_batches
			batch_list[pointer].extend(inst_ids)
			group_id_list[pointer].extend([gid for x in inst_ids])
			pointer += 1
			if pointer >= num_batches:
				pointer = pointer % num_batches
			count_pg += len(inst_ids)
		else:
			print('size of positives: ' + str(len(inst_ids)))
		if count_pg >= num_pgroup:
			break

	# Adding singleton postive questions and negative questions
	negatives_list = list(np.random.permutation(negatives_list)[:num_negatives]) # random selecting negatives
	singletons_list = np.random.permutation(negatives_list + singleton_positive_list)
	batch_id = 0
	for sid in singletons_list:
		while batch_id < len(batch_list) and len(batch_list[batch_id]) >= batch_size:
			batch_id += 1
		if batch_id >= len(batch_list):
			break
		batch_list[batch_id].append(sid)
		group_id_list[batch_id].append('no group')

	# random permuting the batches
	permuted_batch_list = []
	permuted_group_id_list = []
	permuted_batch_indices = np.random.permutation(len(batch_list))
	for idx in permuted_batch_indices:
		permuted_batch_list.append(batch_list[idx])
		permuted_group_id_list.append(group_id_list[idx])
	return permuted_batch_list, permuted_group_id_list

def _create_batch_matrix(group_id_list):
	bmat_list = []
	for batch in group_id_list:
		batch_size = len(batch)

		group_ids = list(np.unique(batch))
		if 'no group' in group_ids:
			group_ids.remove('no group')
		num_groups = len(group_ids) + batch.count('no group')
		# batch_matrix = torch.zeros(batch_size,num_groups)
		# wiedest way to initialize with high values so that min of losses could be found
		batch_matrix = torch.ones(batch_size,num_groups) * 10000000 # a matrix of big number
		group_ids_loc_dict = {}
		pointer = 0
		for i,gid in enumerate(batch):
			gid = str(gid)
			if gid == 'no group':
				batch_matrix[i][pointer] = 1.
				pointer += 1
			elif gid in group_ids_loc_dict:
				batch_matrix[i][group_ids_loc_dict[gid]] = 1.
			elif gid not in group_ids_loc_dict:
				group_ids_loc_dict[gid] = pointer
				batch_matrix[i][group_ids_loc_dict[gid]] = 1.
				pointer += 1
		bmat_list.append(batch_matrix)
	return bmat_list

def pl_min_group_loss(logits, targets, label_matrix, criterion):
	# label_matrix has 1s for group member and inf(or high value) for non member
	loss_vec = criterion(logits,targets)
	loss_mat = vec_mat_multiplication(loss_vec, label_matrix)
	return loss_mat.min(axis=0).values.mean()

# def pl_softmax_loss(logits, targets)

def vec_mat_multiplication(vec, mat, axis=0):
	mat = mat.transpose(axis, -1)
	return (mat * vec.expand_as(mat)).transpose(axis, -1)

def retrieval_accuracy(predictions, qa_data):
	# predictions : dict with question_id as key and value is list of scores per row
	# qa_data : list of dicts. dicts contain, question_id and label

	qa_dict = {}
	for  d in qa_data:
		qid = d['question_id']
		if qid not in qa_dict:
			qa_dict[qid] = [d['label']]
		else:
			qa_dict[qid].append(d['label'])
	
	label_match_list = []
	for qid in predictions:
		max_score_row = np.argmax(predictions[qid])
		label_match_list.append(qa_dict[qid][max_score_row])
	accuracy = np.mean(label_match_list)

	return accuracy