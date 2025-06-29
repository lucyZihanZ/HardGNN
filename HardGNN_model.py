# TensorFlow 2.x Compatible HardGNN Model - PRODUCTION VERSION
# This is the single main model for Google Colab Pro+ deployment
# Preserves all original SelfGNN functionality with hard negative sampling enhancement

import os
import numpy as np
import tensorflow as tf
import pickle
import scipy.sparse as sp
from random import randint

# Enable TensorFlow 1.x behavior in TensorFlow 2.x environment
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

from Params import args
import Utils.NNLayers_tf2 as NNs
from Utils.NNLayers_tf2 import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from Utils.attention_tf2 import AdditiveAttention, MultiHeadSelfAttention
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from DataHandler import negSamp, negSamp_fre, transpose, DataHandler, transToLsts

# Ensure tensorflow is imported as tf for the AMP call
import tensorflow as tf 

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'HR', 'NDCG']
		if args.use_hard_neg:
			mets.append('contrastiveLoss')
		for met in mets:
			self.metrics[met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			if save and metric in self.metrics:
				self.metrics[metric].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.compat.v1.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		maxndcg=0.0
		maxres=dict()
		maxepoch=0
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0 and reses['NDCG']>maxndcg:
				self.saveHistory()
				maxndcg=reses['NDCG']
				maxres=reses
				maxepoch=ep
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('max', maxepoch, maxres, True))

	def makeTimeEmbed(self):
		divTerm = 1 / (10000 ** (tf.range(0, args.latdim * 2, 2, dtype=tf.float32) / args.latdim))
		pos = tf.expand_dims(tf.range(0, self.maxTime, dtype=tf.float32), axis=-1)
		sine = tf.expand_dims(tf.math.sin(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		cosine = tf.expand_dims(tf.math.cos(pos * divTerm) / np.sqrt(args.latdim), axis=-1)
		timeEmbed = tf.reshape(tf.concat([sine, cosine], axis=-1), [self.maxTime, args.latdim*2]) / 4.0
		return timeEmbed

	def messagePropagate(self, srclats, mat, type='user'):
		timeEmbed = FC(self.timeEmbed, args.latdim, reg=True)
		srcNodes = tf.squeeze(tf.slice(mat.indices, [0, 1], [-1, 1]))
		tgtNodes = tf.squeeze(tf.slice(mat.indices, [0, 0], [-1, 1]))
		edgeVals = mat.values
		srcEmbeds = tf.nn.embedding_lookup(srclats, srcNodes)
		lat=tf.pad(tf.math.segment_sum(srcEmbeds, tgtNodes),[[0,100],[0,0]])
		if(type=='user'):
			lat=tf.nn.embedding_lookup(lat,self.users)
		else:
			lat=tf.nn.embedding_lookup(lat,self.items)
		return Activate(lat, self.actFunc)

	def edgeDropout(self, mat):
		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			newVals = tf.nn.dropout(tf.cast(values,dtype=tf.float32), self.keepRate)
			return tf.sparse.SparseTensor(indices, tf.cast(newVals,dtype=tf.int32), shape)
		return dropOneMat(mat)

	def ours(self):
		user_vector,item_vector=list(),list()
		# Embedding layers
		uEmbed=NNs.defineParam('uEmbed', [args.graphNum, args.user, args.latdim], reg=True)
		iEmbed=NNs.defineParam('iEmbed', [args.graphNum, args.item, args.latdim], reg=True)
		posEmbed=NNs.defineParam('posEmbed', [args.pos_length, args.latdim], reg=True)
		pos= tf.tile(tf.expand_dims(tf.range(args.pos_length),axis=0),[args.batch,1])
		self.items=tf.range(args.item)
		self.users=tf.range(args.user)
		self.timeEmbed=NNs.defineParam('timeEmbed', [self.maxTime+1, args.latdim], reg=True)

		# Graph neural network layers
		for k in range(args.graphNum):
			embs0=[uEmbed[k]]
			embs1=[iEmbed[k]]
			for i in range(args.gnn_layer):
				a_emb0= self.messagePropagate(embs1[-1],self.edgeDropout(self.subAdj[k]),'user')
				a_emb1= self.messagePropagate(embs0[-1],self.edgeDropout(self.subTpAdj[k]),'item')
				embs0.append(a_emb0+embs0[-1]) 
				embs1.append(a_emb1+embs1[-1]) 
			user=tf.add_n(embs0)
			item=tf.add_n(embs1)
			user_vector.append(user)
			item_vector.append(item)

		user_vector=tf.stack(user_vector,axis=0)
		item_vector=tf.stack(item_vector,axis=0)
		user_vector_tensor=tf.transpose(user_vector, perm=[1, 0, 2])
		item_vector_tensor=tf.transpose(item_vector, perm=[1, 0, 2])

		# Keras 3 / TF 2.16+ compatible RNN using Keras native layers
		# Define LSTM cell instances. Create separate instances if they need to have different weights.
		user_lstm_cell = tf.keras.layers.LSTMCell(args.latdim)
		item_lstm_cell = tf.keras.layers.LSTMCell(args.latdim) # Separate instance for item RNN

		# Use tf.keras.layers.RNN to create the dynamic RNN layer(s)
		# return_sequences=True to get all outputs over the time/sequence dimension
		user_rnn_layer = tf.keras.layers.RNN(user_lstm_cell, return_sequences=True, name='user_rnn_layer')
		item_rnn_layer = tf.keras.layers.RNN(item_lstm_cell, return_sequences=True, name='item_rnn_layer')

		user_vector_rnn_outputs = user_rnn_layer(user_vector_tensor)
		item_vector_rnn_outputs = item_rnn_layer(item_vector_tensor)

		# Apply dropout to the outputs of the RNN layer, similar to what DropoutWrapper aimed to do
		# self.keepRate is a placeholder (e.g., 0.5 for training, 1.0 for testing)
		# The rate for tf.nn.dropout is the probability to drop, so 1.0 - keepRate
		user_vector_tensor = tf.nn.dropout(user_vector_rnn_outputs, rate=(1.0 - self.keepRate))
		item_vector_tensor = tf.nn.dropout(item_vector_rnn_outputs, rate=(1.0 - self.keepRate))

		# Attention mechanisms
		self.additive_attention0 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.additive_attention1 = AdditiveAttention(args.query_vector_dim,args.latdim)
		self.multihead_self_attention0 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)
		self.multihead_self_attention1 = MultiHeadSelfAttention(args.latdim,args.num_attention_heads)

		# Keras native LayerNormalization
		ln_user = tf.keras.layers.LayerNormalization(name='ln_user_rnn_output')
		ln_item = tf.keras.layers.LayerNormalization(name='ln_item_rnn_output')

		multihead_user_vector = self.multihead_self_attention0.attention(ln_user(user_vector_tensor))
		multihead_item_vector = self.multihead_self_attention1.attention(ln_item(item_vector_tensor))
		final_user_vector = tf.reduce_mean(multihead_user_vector,axis=1)
		final_item_vector = tf.reduce_mean(multihead_item_vector,axis=1)

		# Save for hard negative sampling
		self.final_item_vector = final_item_vector
		iEmbed_att=final_item_vector

		# Sequence attention
		self.multihead_self_attention_sequence = list()
		for i in range(args.att_layer):
			self.multihead_self_attention_sequence.append(MultiHeadSelfAttention(args.latdim,args.num_attention_heads))

		# Keras native LayerNormalization for sequence_batch inputs
		ln_seq_embed = tf.keras.layers.LayerNormalization(name='ln_seq_embed')
		ln_pos_embed = tf.keras.layers.LayerNormalization(name='ln_pos_embed')
		ln_att_layer = tf.keras.layers.LayerNormalization(name='ln_att_layer_input') # For the input to the attention loop

		sequence_batch = ln_seq_embed(tf.matmul(tf.expand_dims(self.mask,axis=1),tf.nn.embedding_lookup(iEmbed_att,self.sequence)))
		sequence_batch += ln_pos_embed(tf.matmul(tf.expand_dims(self.mask,axis=1),tf.nn.embedding_lookup(posEmbed,pos)))
		att_layer=sequence_batch # Initial input to the loop
		for i in range(args.att_layer):
			# Instantiate LayerNormalization for each attention layer's input if weights should be distinct, or reuse
			# For simplicity here, let's create new instances or ensure the scope handles variable naming
			# However, often a single LN instance is reused if the transformation is meant to be the same.
			# Given the original code tf.compat.v1.layers.layer_norm(att_layer) implies new computation (possibly shared weights via TF1 default reuse rules)
			# we will instantiate it here to be safe for distinctness or rely on Keras layer naming for reuse within scope if intended.
			ln_current_att_input = tf.keras.layers.LayerNormalization(name=f'ln_att_layer_input_loop_{i}')
			att_layer1=self.multihead_self_attention_sequence[i].attention(ln_current_att_input(att_layer))
			att_layer=Activate(att_layer1,"leakyRelu")+att_layer
		att_user=tf.reduce_sum(att_layer,axis=1)

		# Save for hard negative sampling
		self.att_user = att_user
		
		# Prediction computation
		pckIlat_att = tf.nn.embedding_lookup(iEmbed_att, self.iids)		
		pckUlat = tf.nn.embedding_lookup(final_user_vector, self.uids)
		pckIlat = tf.nn.embedding_lookup(final_item_vector, self.iids)
		preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)
		preds += tf.reduce_sum(Activate(tf.nn.embedding_lookup(att_user,self.uLocs_seq),"leakyRelu")* pckIlat_att,axis=-1)

		# SSL loss computation
		self.preds_one=list()
		self.final_one=list()
		sslloss = 0	
		user_weight=list()
		for i in range(args.graphNum):
			meta1=tf.concat([final_user_vector*user_vector[i],final_user_vector,user_vector[i]],axis=-1)
			meta2=FC(meta1,args.ssldim,useBias=True,activation='leakyRelu',reg=True,reuse=True,name="meta2")
			user_weight.append(tf.squeeze(FC(meta2,1,useBias=True,activation='sigmoid',reg=True,reuse=True,name="meta3")))
		user_weight=tf.stack(user_weight,axis=0)	

		for i in range(args.graphNum):
			sampNum = tf.shape(self.suids[i])[0] // 2
			pckUlat = tf.nn.embedding_lookup(final_user_vector, self.suids[i])
			pckIlat = tf.nn.embedding_lookup(final_item_vector, self.siids[i])
			pckUweight =  tf.nn.embedding_lookup(user_weight[i], self.suids[i])
			pckIlat_att = tf.nn.embedding_lookup(iEmbed_att, self.siids[i])
			S_final = tf.reduce_sum(Activate(pckUlat* pckIlat, self.actFunc),axis=-1)
			posPred_final = tf.stop_gradient(tf.slice(S_final, [0], [sampNum]))
			negPred_final = tf.stop_gradient(tf.slice(S_final, [sampNum], [-1]))
			posweight_final = tf.slice(pckUweight, [0], [sampNum])
			negweight_final = tf.slice(pckUweight, [sampNum], [-1])
			S_final = posweight_final*posPred_final-negweight_final*negPred_final
			pckUlat = tf.nn.embedding_lookup(user_vector[i], self.suids[i])
			pckIlat = tf.nn.embedding_lookup(item_vector[i], self.siids[i])
			preds_one = tf.reduce_sum(Activate(pckUlat* pckIlat , self.actFunc), axis=-1)
			posPred = tf.slice(preds_one, [0], [sampNum])
			negPred = tf.slice(preds_one, [sampNum], [-1])
			sslloss += tf.reduce_sum(tf.maximum(0.0, 1.0 -S_final * (posPred-negPred)))
			self.preds_one.append(preds_one)
		
		return preds, sslloss

	def prepareModel(self):
		self.keepRate = tf.compat.v1.placeholder(dtype=tf.float32, shape=[])
		self.is_train = tf.compat.v1.placeholder_with_default(True, (), 'is_train')
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)
		self.uids = tf.compat.v1.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.compat.v1.placeholder(name='iids', dtype=tf.int32, shape=[None])
		self.sequence = tf.compat.v1.placeholder(name='sequence', dtype=tf.int32, shape=[args.batch,args.pos_length])
		self.mask = tf.compat.v1.placeholder(name='mask', dtype=tf.float32, shape=[args.batch,args.pos_length])
		self.uLocs_seq = tf.compat.v1.placeholder(name='uLocs_seq', dtype=tf.int32, shape=[None])
		self.suids=list()
		self.siids=list()
		self.suLocs_seq=list()
		for k in range(args.graphNum):
			self.suids.append(tf.compat.v1.placeholder(name='suids%d'%k, dtype=tf.int32, shape=[None]))
			self.siids.append(tf.compat.v1.placeholder(name='siids%d'%k, dtype=tf.int32, shape=[None]))
			self.suLocs_seq.append(tf.compat.v1.placeholder(name='suLocs%d'%k, dtype=tf.int32, shape=[None]))
		self.subAdj=list()
		self.subTpAdj=list()
		for i in range(args.graphNum):
			seqadj = self.handler.subMat[i]
			idx, data, shape = transToLsts(seqadj, norm=True)
			self.subAdj.append(tf.sparse.SparseTensor(idx, data, shape))
			idx, data, shape = transToLsts(transpose(seqadj), norm=True)
			self.subTpAdj.append(tf.sparse.SparseTensor(idx, data, shape))
		self.maxTime=self.handler.maxTime

		# Main model computation
		self.preds, self.sslloss = self.ours()
		sampNum = tf.shape(self.uids)[0] // 2
		self.posPred = tf.slice(self.preds, [0], [sampNum])
		self.negPred = tf.slice(self.preds, [sampNum], [-1])
		self.preLoss = tf.reduce_mean(tf.maximum(0.0, 1.0 - (self.posPred - self.negPred)))
		
		# Hard negative sampling with InfoNCE loss
		if args.use_hard_neg:
			self.contrastive_loss = self.compute_infonce_loss(sampNum)
		else:
			self.contrastive_loss = tf.constant(0.0, dtype=tf.float32) # Ensure it's a TF tensor
			
		self.regLoss = args.reg * Regularize() + args.ssl_reg * self.sslloss
		
		# Combined loss with contrastive component
		if args.use_hard_neg:
			self.loss = self.preLoss + self.regLoss + args.contrastive_weight * self.contrastive_loss
		else:
			self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.compat.v1.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		
		# Create the optimizer instance first
		optimizer_instance = tf.compat.v1.train.AdamOptimizer(learningRate)

		# Conditionally apply AMP if enabled via args
		# Add 'enable_amp' to Params.py or manage it in the Colab script that calls this model
		if hasattr(args, 'enable_amp') and args.enable_amp:
			try:
				optimizer_instance = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer_instance)
			except Exception as e:
				pass  # Continue without AMP if unavailable
		
		self.optimizer = optimizer_instance.minimize(self.loss, global_step=globalStep)

	def compute_infonce_loss(self, sampNum):
		"""
		Compute InfoNCE contrastive loss for hard negative sampling
		Memory-optimized implementation for Google Colab Pro+
		"""
		# Get user and item representations
		user_reps = tf.nn.embedding_lookup(self.att_user, self.uLocs_seq)
		item_reps = tf.nn.embedding_lookup(self.final_item_vector, self.iids)
		
		# Split into anchor-positive and anchor-negative pairs
		anchor_reps = user_reps[:sampNum]
		pos_item_reps = item_reps[:sampNum]
		neg_item_reps = item_reps[sampNum:]
		
		# Normalize embeddings for cosine similarity
		anchor_norm = tf.nn.l2_normalize(anchor_reps, axis=1)
		pos_norm = tf.nn.l2_normalize(pos_item_reps, axis=1)
		neg_norm = tf.nn.l2_normalize(neg_item_reps, axis=1)
		
		# Compute positive similarities
		pos_sim = tf.reduce_sum(anchor_norm * pos_norm, axis=1)
		pos_sim = tf.expand_dims(pos_sim, axis=1)
		
		# Memory optimization: limit negative samples
		max_neg_samples = 50
		neg_count = tf.minimum(tf.shape(neg_norm)[0], max_neg_samples)
		neg_indices = tf.random.shuffle(tf.range(tf.shape(neg_norm)[0]))[:neg_count]
		neg_norm_subset = tf.gather(neg_norm, neg_indices)
		
		# Compute negative similarities
		neg_sim = tf.matmul(anchor_norm, tf.transpose(neg_norm_subset))
		
		# Concatenate and apply temperature
		logits = tf.concat([pos_sim, neg_sim], axis=1)
		logits = logits / args.temp
		
		# InfoNCE loss
		labels = tf.zeros(sampNum, dtype=tf.int32)
		infonce_loss = tf.reduce_mean(
			tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		)
		
		return infonce_loss

	def sampleTrainBatch(self, batIds, labelMat, timeMat, train_sample_num):
		temTst = self.handler.tstInt[batIds]
		temLabel=labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * train_sample_num
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		uLocs_seq = [None]* temlen
		sequence = [None] * args.batch
		mask = [None]*args.batch
		cur = 0

		# Prepare data for batch hard negative sampling
		batch_user_interactions = []
		batch_sampNums = []
		batch_need_hard_neg = []
		
		# First pass: collect all necessary data for batch processing
		for i in range(batch):
			posset=self.handler.sequence[batIds[i]][:-1]
			sampNum = min(train_sample_num, len(posset))
			batch_sampNums.append(sampNum)
			batch_user_interactions.append(temLabel[i])
			batch_need_hard_neg.append(sampNum > 0 and args.use_hard_neg)
		
		# Batch hard negative sampling for all users at once
		if args.use_hard_neg and any(batch_need_hard_neg):
			all_hard_negatives = self.batch_sample_hard_negatives(batIds, batch_user_interactions, batch_sampNums)
		else:
			all_hard_negatives = [[] for _ in range(batch)]

		# Second pass: build the training batch using pre-computed hard negatives
		for i in range(batch):
			posset=self.handler.sequence[batIds[i]][:-1]
			sampNum = min(train_sample_num, len(posset))
			choose=1
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs = []
				choose = randint(1,max(min(args.pred_num+1,len(posset)-3),1))
				poslocs.extend([posset[-choose]]*sampNum)
				
				# Use pre-computed hard negatives or fallback to random
				if args.use_hard_neg and i < len(all_hard_negatives) and len(all_hard_negatives[i]) > 0:
					neglocs = all_hard_negatives[i]
				else:
					neglocs = negSamp(temLabel[i], sampNum, args.item, [self.handler.sequence[batIds[i]][-1],temTst[i]], self.handler.item_with_pop)
			
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				uLocs_seq[cur] = uLocs_seq[cur+temlen//2] = i
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
			sequence[i]=np.zeros(args.pos_length,dtype=int)
			mask[i]=np.zeros(args.pos_length)
			posset=posset[:-choose]
			if(len(posset)<=args.pos_length):
				sequence[i][-len(posset):]=posset
				mask[i][-len(posset):]=1
			else:
				sequence[i]=posset[-args.pos_length:]
				mask[i]+=1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		uLocs_seq = uLocs_seq[:cur] + uLocs_seq[temlen//2: temlen//2 + cur]
		if(batch<args.batch):
			for i in range(batch,args.batch):
				sequence[i]=np.zeros(args.pos_length,dtype=int)
				mask[i]=np.zeros(args.pos_length)
		return uLocs, iLocs, sequence,mask, uLocs_seq

	def sample_hard_negatives(self, user_id, user_interactions, sampNum):
		"""
		Optimized hard negative sampling with pre-computed embeddings
		"""
		# Handle K=0 case - no hard negatives
		if args.hard_neg_top_k == 0:
			return negSamp(user_interactions, sampNum, args.item, 
				[self.handler.sequence[user_id][-1] if len(self.handler.sequence[user_id]) > 0 else None, 
				self.handler.tstInt[user_id]], self.handler.item_with_pop)
		
		# Fast fallback for empty sequences
		user_seq = self.handler.sequence[user_id][:-1]
		if len(user_seq) == 0:
			return negSamp(user_interactions, sampNum, args.item, 
				[self.handler.sequence[user_id][-1] if len(self.handler.sequence[user_id]) > 0 else None, 
				self.handler.tstInt[user_id]], self.handler.item_with_pop)
		
		# Use cached embeddings if available
		if hasattr(self, '_cached_item_embeddings') and self._cached_item_embeddings is not None:
			items_embed = self._cached_item_embeddings
		else:
			# Compute once and cache for the entire batch
			try:
				feed_dict = {self.is_train: False, self.keepRate: 1.0}
				# Minimal dummy values
				dummy_vals = np.zeros(2, dtype=np.int32)
				feed_dict[self.uids] = dummy_vals
				feed_dict[self.iids] = dummy_vals
				feed_dict[self.uLocs_seq] = dummy_vals
				feed_dict[self.sequence] = np.zeros((args.batch, args.pos_length), dtype=int)
				feed_dict[self.mask] = np.zeros((args.batch, args.pos_length))
				
				for k in range(args.graphNum):
					feed_dict[self.suids[k]] = dummy_vals
					feed_dict[self.siids[k]] = dummy_vals
					feed_dict[self.suLocs_seq[k]] = dummy_vals
				
				items_embed = self.sess.run(self.final_item_vector, feed_dict=feed_dict)
				self._cached_item_embeddings = items_embed
			except:
				return negSamp(user_interactions, sampNum, args.item, 
					[self.handler.sequence[user_id][-1], self.handler.tstInt[user_id]], 
					self.handler.item_with_pop)
		
		# Fast user embedding computation using cached historical patterns
		if hasattr(self, '_user_profile_cache') and user_id in self._user_profile_cache:
			user_embed = self._user_profile_cache[user_id]
		else:
			# Simplified user representation: average of recent item embeddings
			if len(user_seq) > 0:
				recent_items = user_seq[-min(10, len(user_seq)):]  # Last 10 items max
				user_embed = np.mean(items_embed[recent_items], axis=0)
			else:
				user_embed = np.mean(items_embed, axis=0)  # Global average fallback
		
		# Fast approximate similarity using optimized numpy operations
		user_norm = np.linalg.norm(user_embed)
		items_norm = np.linalg.norm(items_embed, axis=1)
		user_norm = max(user_norm, 1e-10)
		items_norm = np.maximum(items_norm, 1e-10)
		
		# Vectorized cosine similarity
		similarities = np.dot(items_embed, user_embed) / (items_norm * user_norm)
		
		# Fast exclusion of interacted items
		interacted_items = np.where(user_interactions > 0)[0]
		similarities[interacted_items] = -np.inf
		if self.handler.tstInt[user_id] is not None:
			similarities[self.handler.tstInt[user_id]] = -np.inf
		
		# Fast top-k selection using argpartition (O(n) vs O(nlogn) for full sort)
		if args.hard_neg_top_k < len(similarities):
			# Use argpartition for faster top-k selection
			partition_idx = len(similarities) - args.hard_neg_top_k
			hard_neg_indices = np.argpartition(similarities, partition_idx)[partition_idx:]
		else:
			hard_neg_indices = np.argsort(similarities)[-args.hard_neg_top_k:]
		
		# Handle sample number requirements
		if sampNum > len(hard_neg_indices):
			additional_negs = negSamp(user_interactions, sampNum - len(hard_neg_indices), 
									args.item, [self.handler.sequence[user_id][-1], self.handler.tstInt[user_id]], 
									self.handler.item_with_pop)
			return list(hard_neg_indices) + additional_negs
		
		if sampNum < len(hard_neg_indices):
			return np.random.choice(hard_neg_indices, sampNum, replace=False).tolist()
		
		return hard_neg_indices.tolist()

	def clear_hard_neg_cache(self):
		"""Clear cached embeddings - call between epochs"""
		if hasattr(self, '_cached_item_embeddings'):
			self._cached_item_embeddings = None
		if hasattr(self, '_user_profile_cache'):
			self._user_profile_cache = {}
		if hasattr(self, '_cache_step_counter'):
			self._cache_step_counter = 0

	def should_refresh_cache(self, force_refresh=False):
		"""Determine if embeddings cache should be refreshed"""
		if force_refresh:
			return True
		
		if not hasattr(self, '_cache_step_counter'):
			self._cache_step_counter = 0
		
		# Refresh cache every N training steps to balance accuracy and speed
		cache_refresh_interval = getattr(args, 'cache_refresh_steps', 50)  # Refresh every 50 steps by default
		self._cache_step_counter += 1
		
		if self._cache_step_counter >= cache_refresh_interval:
			self._cache_step_counter = 0
			return True
		
		return False

	def batch_sample_hard_negatives(self, batIds, user_interactions_batch, sampNums):
		"""
		Optimized batch hard negative sampling with smart caching
		"""
		if args.hard_neg_top_k == 0:
			return [negSamp(user_interactions_batch[i], sampNums[i], args.item, 
				[self.handler.sequence[batIds[i]][-1] if len(self.handler.sequence[batIds[i]]) > 0 else None, 
				self.handler.tstInt[batIds[i]]], self.handler.item_with_pop) 
				for i in range(len(batIds))]
		
		# Use cached embeddings if available and fresh enough
		need_refresh = self.should_refresh_cache() or not hasattr(self, '_cached_item_embeddings') or self._cached_item_embeddings is None
		
		# Compute embeddings only when needed
		if need_refresh:
			try:
				batch_size = len(batIds)
				seq_batch = np.zeros((args.batch, args.pos_length), dtype=int)
				mask_batch = np.zeros((args.batch, args.pos_length))
				
				for i, user_id in enumerate(batIds):
					if i >= args.batch:
						break
					user_seq = self.handler.sequence[user_id][:-1]
					if len(user_seq) > 0:
						if len(user_seq) <= args.pos_length:
							seq_batch[i, -len(user_seq):] = user_seq
							mask_batch[i, -len(user_seq):] = 1
						else:
							seq_batch[i, :] = user_seq[-args.pos_length:]
							mask_batch[i, :] = 1
				
				feed_dict = {
					self.sequence: seq_batch,
					self.mask: mask_batch,
					self.is_train: False,
					self.keepRate: 1.0
				}
				
				# Dummy values for required placeholders
				dummy_vals = np.zeros(max(2, batch_size), dtype=np.int32)
				feed_dict[self.uids] = dummy_vals
				feed_dict[self.iids] = dummy_vals
				feed_dict[self.uLocs_seq] = dummy_vals
				
				for k in range(args.graphNum):
					feed_dict[self.suids[k]] = dummy_vals
					feed_dict[self.siids[k]] = dummy_vals
					feed_dict[self.suLocs_seq[k]] = dummy_vals
				
				# Single forward pass for all embeddings
				items_embed, user_att = self.sess.run([self.final_item_vector, self.att_user], feed_dict=feed_dict)
				self._cached_item_embeddings = items_embed
				self._cached_user_embeddings = user_att
			except Exception as e:
				# Fallback to individual sampling
				return [self.sample_hard_negatives(batIds[i], user_interactions_batch[i], sampNums[i]) 
						for i in range(len(batIds))]
		else:
			items_embed = self._cached_item_embeddings
			# For user embeddings, use simplified computation when not refreshing
			user_att = np.zeros((len(batIds), items_embed.shape[1]))
			for i, user_id in enumerate(batIds):
				user_seq = self.handler.sequence[user_id][:-1]
				if len(user_seq) > 0:
					recent_items = user_seq[-min(5, len(user_seq)):]
					user_att[i] = np.mean(items_embed[recent_items], axis=0)
				else:
					user_att[i] = np.mean(items_embed, axis=0)
		
		# Process each user's hard negatives using cached embeddings
		all_hard_negs = []
		batch_size = len(batIds)
		for i, user_id in enumerate(batIds):
			if i >= batch_size:
				break
				
			user_embed = user_att[i]
			user_interactions = user_interactions_batch[i]
			sampNum = sampNums[i]
			
			# Fast similarity computation
			user_norm = np.linalg.norm(user_embed)
			items_norm = np.linalg.norm(items_embed, axis=1)
			user_norm = max(user_norm, 1e-10)
			items_norm = np.maximum(items_norm, 1e-10)
			similarities = np.dot(items_embed, user_embed) / (items_norm * user_norm)
			
			# Exclude interacted items
			interacted_items = np.where(user_interactions > 0)[0]
			similarities[interacted_items] = -np.inf
			if self.handler.tstInt[user_id] is not None:
				similarities[self.handler.tstInt[user_id]] = -np.inf
			
			# Fast top-k selection
			if args.hard_neg_top_k < len(similarities):
				partition_idx = len(similarities) - args.hard_neg_top_k
				hard_neg_indices = np.argpartition(similarities, partition_idx)[partition_idx:]
			else:
				hard_neg_indices = np.argsort(similarities)[-args.hard_neg_top_k:]
			
			# Handle sample requirements
			if sampNum > len(hard_neg_indices):
				additional_negs = negSamp(user_interactions, sampNum - len(hard_neg_indices), 
										args.item, [self.handler.sequence[user_id][-1], self.handler.tstInt[user_id]], 
										self.handler.item_with_pop)
				all_hard_negs.append(list(hard_neg_indices) + additional_negs)
			elif sampNum < len(hard_neg_indices):
				all_hard_negs.append(np.random.choice(hard_neg_indices, sampNum, replace=False).tolist())
			else:
				all_hard_negs.append(hard_neg_indices.tolist())
		
		return all_hard_negs

	def sampleSslBatch(self, batIds, labelMat, use_epsilon=True):
		temLabel=list()
		for k in range(args.graphNum):	
			temLabel.append(labelMat[k][batIds].toarray())
		batch = len(batIds)
		temlen = batch * 2 * args.sslNum
		uLocs = [[None] * temlen] * args.graphNum
		iLocs = [[None] * temlen] * args.graphNum
		uLocs_seq = [[None] * temlen] * args.graphNum

		for k in range(args.graphNum):	
			cur = 0				
			for i in range(batch):
				posset = np.reshape(np.argwhere(temLabel[k][i]!=0), [-1])
				sslNum = min(args.sslNum, len(posset)//2)
				if sslNum == 0:
					poslocs = [np.random.choice(args.item)]
					neglocs = [poslocs[0]]
				else:
					all = np.random.choice(posset, sslNum*2)
					poslocs = all[:sslNum]
					neglocs = all[sslNum:]
				for j in range(sslNum):
					posloc = poslocs[j]
					negloc = neglocs[j]			
					uLocs[k][cur] = uLocs[k][cur+1] = batIds[i]
					uLocs_seq[k][cur] = uLocs_seq[k][cur+1] = i
					iLocs[k][cur] = posloc
					iLocs[k][cur+1] = negloc
					cur += 2
			uLocs[k]=uLocs[k][:cur]
			iLocs[k]=iLocs[k][:cur]
			uLocs_seq[k]=uLocs_seq[k][:cur]
		return uLocs, iLocs, uLocs_seq

	def trainEpoch(self):
		# Clear hard negative sampling cache for fresh embeddings each epoch
		self.clear_hard_neg_cache()
		
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		epochContrastiveLoss = 0
		num = len(sfIds)
		sample_num_list=[40]		
		steps = int(np.ceil(num / args.batch))

		for s in range(len(sample_num_list)):
			for i in range(steps):
				st = i * args.batch
				ed = min((i+1) * args.batch, num)
				batIds = sfIds[st: ed]

				if args.use_hard_neg:
					target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.contrastive_loss, self.posPred, self.negPred, self.preds_one]
				else:
					target = [self.optimizer, self.preLoss, self.regLoss, self.loss, self.posPred, self.negPred, self.preds_one]
					
				feed_dict = {}
				uLocs, iLocs, sequence, mask, uLocs_seq= self.sampleTrainBatch(batIds, self.handler.trnMat, self.handler.timeMat, sample_num_list[s])
				suLocs, siLocs, suLocs_seq = self.sampleSslBatch(batIds, self.handler.subMat, False)
				feed_dict[self.uids] = uLocs
				feed_dict[self.iids] = iLocs
				feed_dict[self.sequence] = sequence
				feed_dict[self.mask] = mask
				feed_dict[self.is_train] = True
				feed_dict[self.uLocs_seq] = uLocs_seq
				
				for k in range(args.graphNum):
					feed_dict[self.suids[k]] = suLocs[k]
					feed_dict[self.siids[k]] = siLocs[k]
					feed_dict[self.suLocs_seq[k]] = suLocs_seq[k]
				feed_dict[self.keepRate] = args.keepRate

				# TF2-compatible session run without config_pb2
				res = self.sess.run(target, feed_dict=feed_dict)

				if args.use_hard_neg:
					preLoss, regLoss, loss, contrastiveLoss, pos, neg, pone = res[1:]
					epochContrastiveLoss += contrastiveLoss
					log('Step %d/%d: preloss = %.2f, REGLoss = %.2f, ConLoss = %.4f         ' % 
						(i+s*steps, steps*len(sample_num_list), preLoss, regLoss, contrastiveLoss), save=False, oneline=True)
				else:
					preLoss, regLoss, loss, pos, neg, pone = res[1:]
					log('Step %d/%d: preloss = %.2f, REGLoss = %.2f         ' % 
						(i+s*steps, steps*len(sample_num_list), preLoss, regLoss), save=False, oneline=True)
					
				epochLoss += loss
				epochPreLoss += preLoss
				
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		if args.use_hard_neg:
			ret['contrastiveLoss'] = epochContrastiveLoss / steps
		return ret

	def sampleTestBatch(self, batIds, labelMat):
		batch = len(batIds)
		temTst = self.handler.tstInt[batIds]
		temLabel = labelMat[batIds].toarray()
		temlen = batch * args.testSize
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		uLocs_seq = [None] * temlen
		tstLocs = [None] * batch
		sequence = [None] * args.batch
		mask = [None]*args.batch
		cur = 0
		val_list=[None]*args.batch

		for i in range(batch):
			if(args.test==True):
				posloc = temTst[i]
			else:
				posloc = self.handler.sequence[batIds[i]][-1]
				val_list[i]=posloc
			rdnNegSet = np.array(self.handler.test_dict[batIds[i]+1][:args.testSize-1])-1
			locset = np.concatenate((rdnNegSet, np.array([posloc])))
			tstLocs[i] = locset
			for j in range(len(locset)):
				uLocs[cur] = batIds[i]
				iLocs[cur] = locset[j]
				uLocs_seq[cur] = i
				cur += 1
			sequence[i]=np.zeros(args.pos_length,dtype=int)
			mask[i]=np.zeros(args.pos_length)
			if(args.test==True):
				posset=self.handler.sequence[batIds[i]]
			else:
				posset=self.handler.sequence[batIds[i]][:-1]

			if(len(posset)<=args.pos_length):
				sequence[i][-len(posset):]=posset
				mask[i][-len(posset):]=1
			else:
				sequence[i]=posset[-args.pos_length:]
				mask[i]+=1
		if(batch<args.batch):
			for i in range(batch,args.batch):
				sequence[i]=np.zeros(args.pos_length,dtype=int)
				mask[i]=np.zeros(args.pos_length)
		return uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list

	def testEpoch(self):
		epochHit, epochNdcg = [0] * 2
		epochHit5, epochNdcg5 = [0] * 2
		epochHit20, epochNdcg20 = [0] * 2
		epochHit1, epochNdcg1 = [0] * 2
		epochHit15, epochNdcg15 = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))

		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}
			uLocs, iLocs, temTst, tstLocs, sequence, mask, uLocs_seq, val_list = self.sampleTestBatch(batIds, self.handler.trnMat)
			suLocs, siLocs, _ = self.sampleSslBatch(batIds, self.handler.subMat, False)
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.is_train] = False
			feed_dict[self.sequence] = sequence
			feed_dict[self.mask] = mask
			feed_dict[self.uLocs_seq] = uLocs_seq

			for k in range(args.graphNum):
				feed_dict[self.suids[k]] = suLocs[k]
				feed_dict[self.siids[k]] = siLocs[k]
			feed_dict[self.keepRate] = 1.0

			# TF2-compatible session run
			preds = self.sess.run(self.preds, feed_dict=feed_dict)

			if(args.test==True):
				hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), temTst, tstLocs)
			else:
				hit, ndcg, hit5, ndcg5, hit20, ndcg20,hit1, ndcg1,  hit15, ndcg15= self.calcRes(np.reshape(preds, [ed-st, args.testSize]), val_list, tstLocs)
			epochHit += hit
			epochNdcg += ndcg
			epochHit5 += hit5
			epochNdcg5 += ndcg5
			epochHit20 += hit20
			epochNdcg20 += ndcg20
			epochHit15 += hit15
			epochNdcg15 += ndcg15
			epochHit1 += hit1
			epochNdcg1 += ndcg1
			log('Steps %d/%d: hit10 = %d, ndcg10 = %d' % (i, steps, hit, ndcg), save=False, oneline=True)

		ret = dict()
		ret['HR'] = epochHit / num
		ret['NDCG'] = epochNdcg / num
		return ret

	def calcRes(self, preds, temTst, tstLocs):
		hit = 0
		ndcg = 0
		hit1 = 0
		ndcg1 = 0
		hit5=0
		ndcg5=0
		hit20=0
		ndcg20=0
		hit15=0
		ndcg15=0
		for j in range(preds.shape[0]):
			predvals = list(zip(preds[j], tstLocs[j]))
			predvals.sort(key=lambda x: x[0], reverse=True)
			shoot = list(map(lambda x: x[1], predvals[:args.shoot]))
			if temTst[j] in shoot:
				hit += 1
				ndcg += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
			shoot = list(map(lambda x: x[1], predvals[:5]))
			if temTst[j] in shoot:
				hit5 += 1
				ndcg5 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))
			shoot = list(map(lambda x: x[1], predvals[:20]))	
			if temTst[j] in shoot:
				hit20 += 1
				ndcg20 += np.reciprocal(np.log2(shoot.index(temTst[j])+2))	
		return hit, ndcg, hit5, ndcg5, hit20, ndcg20, hit1, ndcg1, hit15, ndcg15
	
	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)
		saver = tf.compat.v1.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.compat.v1.train.Saver()
		saver.restore(self.sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded') 