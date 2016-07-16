from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from tf_seq2seq_chatbot.configs.config import FLAGS, BUCKETS
from tf_seq2seq_chatbot.lib import data_utils
from tf_seq2seq_chatbot.lib import seq2seq_model


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      source_vocab_size=FLAGS.vocab_size,
      target_vocab_size=FLAGS.vocab_size,
      buckets=BUCKETS,
      size=FLAGS.size,
      num_layers=FLAGS.num_layers,
      max_gradient_norm=FLAGS.max_gradient_norm,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
      use_lstm=False,
      forward_only=forward_only)

  ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
  if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def get_predicted_sentence_beam(input_sentence, vocab, rev_vocab, model, sess, beam_size = 20):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    ### beam search 
    log_beam_probs, beam_path  = [], []
    num_symbols = output_logits[0].shape[1]
    num_iters = 10 ## at most predict 50 tokens
    num_input = len(input_token_ids) 

    for i in xrange(num_iters):
      if i == 0:
        best_probs, indices =  tf.nn.top_k(output_logits[0], beam_size)
        indices = tf.stop_gradient(tf.squeeze(tf.reshape(indices, [-1, 1])))
        best_probs = tf.stop_gradient(tf.reshape(best_probs, [-1, 1]))

        indices = indices.eval().reshape(-1,).tolist()
        best_probs = np.log(best_probs.eval().reshape(-1,)).tolist()
        #print ('tf.log(output_logits[0]): ', tf.log(output_logits[0]).eval())

        #print('best_probs: ', best_probs, ' indices: ', indices)
        log_beam_probs += best_probs
        beam_path = [[ind] for ind in indices]
      else: 
        probs = []
        paths = []
        cnt = 0 
        inf = 20
        for j in xrange(beam_size):
          if beam_path[j][-1] == data_utils.EOS_ID:
            cnt += 1
            probs.append(log_beam_probs[j]+inf)
            paths.append(beam_path[j])
            '''
            for k in xrange(beam_size):
              probs.append(log_beam_probs[j]+inf)
              paths.append(beam_path[j])
            '''
          else:
            new_input = input_token_ids
            output = beam_path[j]
            feed_data = {bucket_id: [(input_token_ids, output)]}
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)
            _, _, opt_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)
            
            #new_best_probs, new_indices = tf.nn.top_k(opt_logits[0], beam_size)
            new_best_probs, new_indices = tf.nn.top_k(opt_logits[len(output)], beam_size)
            new_indices = tf.stop_gradient(tf.squeeze(tf.reshape(new_indices,[-1, 1])))
            new_best_probs =  tf.stop_gradient(tf.reshape(new_best_probs, [-1,1]))

            new_indices = new_indices.eval().reshape(-1,).tolist()
            new_best_probs = np.log(new_best_probs.eval().reshape(-1,)).tolist()
          
            ### probs will contain beam_size * beam_size probabilities
            ### paths will contain all the previous paths so far.
            for k in xrange(beam_size):
              probs.append(new_best_probs[k]+log_beam_probs[j])
              paths.append(beam_path[j]+[new_indices[k]])

        #print('step %d: ' % i)
        #print('probs: ', probs)
        #print('cnt: ', cnt)
        selected_best_probs, selected_indices = tf.nn.top_k(probs, beam_size) 
        selected_indices = tf.stop_gradient(tf.squeeze(tf.reshape(selected_indices, [-1,1])))
        selected_best_probs = tf.stop_gradient(tf.reshape(selected_best_probs, [-1, 1]))

 
        selected_indices = selected_indices.eval().reshape(-1,).tolist()
        selected_best_probs = selected_best_probs.eval().reshape(-1,).tolist()
        
        if cnt == beam_size:
          print('nothing to expand')
          break

        log_beam_probs = selected_best_probs ### size k
        beam_path = [paths[p] for p in selected_indices]  ### new paths
        #print('log_beam_probs: ', log_beam_probs)
        #print('beam_path', beam_path)

    res_probs, res_indices = tf.nn.top_k(log_beam_probs, 5)
    res_indices = tf.stop_gradient(tf.squeeze(tf.reshape(res_indices, [-1,1])))
    res_probs = tf.stop_gradient(tf.reshape(res_probs, [-1,1]))

    
    res_indices = res_indices.eval().reshape(-1,).tolist()
    res_probs = res_probs.eval().reshape(-1,).tolist()


    print('res_probs: ', res_probs, 'res_indices: ', res_indices)

    """
    ### top 1 
    output_logits = beam_path[res_indices[0]]

    outputs = []
    for logit in output_logits:
        if logit == data_utils.EOS_ID:
            break
        else:
            outputs.append(logit)
    

    # Forming output sentence on natural language
    output_sentence = ' '.join([rev_vocab[output] for output in outputs])
    """
    ### top 5
    output_sentence = []
    for ind in xrange(5):
      output_logits = beam_path[res_indices[ind]]
      outputs = []
      for logit in output_logits:
        if logit == data_utils.EOS_ID:
          break
        else:
          outputs.append(logit)
      print(ind, 'prediction: ', ' '.join([rev_vocab[output] for output in outputs]), 'prob: ', res_probs[ind])
      output_sentence.append(' '.join([rev_vocab[output] for output in outputs]))
    return output_sentence

def get_predicted_sentence(input_sentence, vocab, rev_vocab, model, sess):
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(BUCKETS)) if BUCKETS[b][0] > len(input_token_ids)])
    outputs = []

    feed_data = {bucket_id: [(input_token_ids, outputs)]}
    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)


    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)

    outputs = []
    #print('output_logits is ', output_logits)
    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    for logit in output_logits:
        #print('shape of logit: ', logit.shape)
        selected_token_id = int(np.argmax(logit, axis=1))

        if selected_token_id == data_utils.EOS_ID:
            break
        else:
            outputs.append(selected_token_id)

    # Forming output sentence on natural language
    output_sentence = ' '.join([rev_vocab[output] for output in outputs])

    return output_sentence