from utils import * 
import pickle 
import argparse
import tensorflow as tf 
from flask import Flask, request, render_template

app = Flask(__name__)

parser = argparse.ArgumentParser(description="Test URLNet model")

# data args
default_max_len_words = 200
parser.add_argument('--data.max_len_words', type=int, default=default_max_len_words, metavar="MLW",
  help="maximum length of url in words (default: {})".format(default_max_len_words))
default_max_len_chars = 200
parser.add_argument('--data.max_len_chars', type=int, default=default_max_len_chars, metavar="MLC",
  help="maximum length of url in characters (default: {})".format(default_max_len_chars))
default_max_len_subwords = 20
parser.add_argument('--data.max_len_subwords', type=int, default=default_max_len_subwords, metavar="MLSW",
  help="maxium length of word in subwords/ characters (default: {})".format(default_max_len_subwords))
parser.add_argument('--data.data_dir', type=str, default='test.txt', metavar="DATADIR",
  help="location of data file")
default_delimit_mode = 1
parser.add_argument("--data.delimit_mode", type=int, default=default_delimit_mode, metavar="DLMODE",
  help="0: delimit by special chars, 1: delimit by special chars + each char as a word (default: {})".format(default_delimit_mode))
parser.add_argument('--data.subword_dict_dir', type=str, default="runs/10000/subwords_dict.p", metavar="SUBWORD_DICT", 
    help="directory of the subword dictionary")
parser.add_argument('--data.word_dict_dir', type=str, default="runs/10000/words_dict.p", metavar="WORD_DICT",
    help="directory of the word dictionary")
parser.add_argument('--data.char_dict_dir', type=str, default="runs/10000/chars_dict.p", metavar="  CHAR_DICT",
    help="directory of the character dictionary")

# model args 
default_emb_dim = 32
parser.add_argument('--model.emb_dim', type=int, default=default_emb_dim, metavar="EMBDIM",
  help="embedding dimension size (default: {})".format(default_emb_dim))
default_emb_mode = 5
parser.add_argument('--model.emb_mode', type=int, default=default_emb_mode, metavar="EMBMODE",
  help="1: charCNN, 2: wordCNN, 3: char + wordCNN, 4: char-level wordCNN, 5: char + char-level wordCNN (default: {})".format(default_emb_mode))

# test args 
default_batch_size = 64
parser.add_argument('--test.batch_size', type=int, default=default_batch_size, metavar="BATCHSIZE",
  help="Size of each test batch (default: {})".format(default_batch_size))

# log args 
parser.add_argument('--log.output_dir', type=str, default="runs/10000/testLog", metavar="OUTPUTDIR",
  help="directory to save the test results")
parser.add_argument('--log.checkpoint_dir', type=str, default="runs/10000/checkpoints/", metavar="CHECKPOINTDIR",
    help="directory of the learned model")

def test_step(x, emb_mode):
    p = 1.0
    if emb_mode == 1: 
        feed_dict = {
            input_x_char_seq: x[0],
            dropout_keep_prob: p}  
    elif emb_mode == 2: 
        feed_dict = {
            input_x_word: x[0],
            dropout_keep_prob: p}
    elif emb_mode == 3: 
        feed_dict = {
            input_x_char_seq: x[0],
            input_x_word: x[1],
            dropout_keep_prob: p}
    elif emb_mode == 4: 
        feed_dict = {
            input_x_word: x[0],
            input_x_char: x[1],
            input_x_char_pad_idx: x[2],
            dropout_keep_prob: p}
    elif emb_mode == 5:  
        feed_dict = {
            input_x_char_seq: x[0],
            input_x_word: x[1],
            input_x_char: x[2],
            input_x_char_pad_idx: x[3],
            dropout_keep_prob: p}
    preds, s = sess.run([predictions, scores], feed_dict)
    return preds, s

FLAGS = vars(parser.parse_args())
for key, val in FLAGS.items():
    print("{}={}".format(key, val))

ngram_dict = pickle.load(open(FLAGS["data.subword_dict_dir"], "rb")) 
#print("Size of subword vocabulary (train): {}".format(len(ngram_dict)))
word_dict = pickle.load(open(FLAGS["data.word_dict_dir"], "rb"))
#print("size of word vocabulary (train): {}".format(len(word_dict)))
chars_dict = pickle.load(open(FLAGS["data.char_dict_dir"], "rb"))



######################## EVALUATION ########################### 

checkpoint_file = tf.train.latest_checkpoint(FLAGS["log.checkpoint_dir"])
graph = tf.Graph() 
with graph.as_default(): 
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True 
    sess = tf.Session(config=session_conf)
    with sess.as_default(): 
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file) 
        
        if FLAGS["model.emb_mode"] in [1, 3, 5]: 
            input_x_char_seq = graph.get_operation_by_name("input_x_char_seq").outputs[0]
        if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
            input_x_word = graph.get_operation_by_name("input_x_word").outputs[0]
        if FLAGS["model.emb_mode"] in [4, 5]:
            input_x_char = graph.get_operation_by_name("input_x_char").outputs[0]
            input_x_char_pad_idx = graph.get_operation_by_name("input_x_char_pad_idx").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0] 

        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
        scores = graph.get_operation_by_name("output/scores").outputs[0]


@app.route("/", methods=['GET', 'POST'])
def index():
    if True:
        while True:
            urls = []
            url = request.args.get('url')
            print ('checking: ' + url)
            if url.startswith("http://"):
                url=url[7:]
            if url.startswith("https://"):
                url=url[8:]
            if url.startswith("ftp://"):
                url=url[6:]
            if url.startswith("www."):
                url = url[4:]
            urls.append(url)
            x, word_reverse_dict = get_word_vocab(urls, FLAGS["data.max_len_words"]) 
            word_x = get_words(x, word_reverse_dict, FLAGS["data.delimit_mode"], urls) 
            ngramed_id_x, worded_id_x = ngram_id_x_from_dict(word_x, FLAGS["data.max_len_subwords"], ngram_dict, word_dict) 
            chared_id_x = char_id_x(urls, chars_dict, FLAGS["data.max_len_chars"])
        
            if FLAGS["model.emb_mode"] == 1: 
                batches = batch_iter(list(chared_id_x), FLAGS["test.batch_size"], 1, shuffle=False) 
            elif FLAGS["model.emb_mode"] == 2: 
                batches = batch_iter(list(worded_id_x), FLAGS["test.batch_size"], 1, shuffle=False) 
            elif FLAGS["model.emb_mode"] == 3: 
                batches = batch_iter(list(zip(chared_id_x, worded_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)
            elif FLAGS["model.emb_mode"] == 4: 
                batches = batch_iter(list(zip(ngramed_id_x, worded_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)
            elif FLAGS["model.emb_mode"] == 5: 
                batches = batch_iter(list(zip(ngramed_id_x, worded_id_x, chared_id_x)), FLAGS["test.batch_size"], 1, shuffle=False)    
        
            nb_batches = 1

            batch = next(batches)
            if FLAGS["model.emb_mode"] == 1: 
                x_char_seq = batch 
            elif FLAGS["model.emb_mode"] == 2: 
                x_word = batch 
            elif FLAGS["model.emb_mode"] == 3: 
                x_char_seq, x_word = zip(*batch) 
            elif FLAGS["model.emb_mode"] == 4: 
                x_char, x_word = zip(*batch)
            elif FLAGS["model.emb_mode"] == 5: 
                x_char, x_word, x_char_seq = zip(*batch)            

            x_batch = []    
            if FLAGS["model.emb_mode"] in[1, 3, 5]: 
                x_char_seq = pad_seq_in_word(x_char_seq, FLAGS["data.max_len_chars"]) 
                x_batch.append(x_char_seq)
            if FLAGS["model.emb_mode"] in [2, 3, 4, 5]:
                x_word = pad_seq_in_word(x_word, FLAGS["data.max_len_words"]) 
                x_batch.append(x_word)
            if FLAGS["model.emb_mode"] in [4, 5]:
                x_char, x_char_pad_idx = pad_seq(x_char, FLAGS["data.max_len_words"], FLAGS["data.max_len_subwords"], FLAGS["model.emb_dim"])
                x_batch.extend([x_char, x_char_pad_idx])
            
            batch_predictions, batch_scores = test_step(x_batch, FLAGS["model.emb_mode"])
            #print(batch_predictions[0]) # batch_predictions[0] la ket qua (0 la sach, 1 la doc hai)
            if (int)(batch_predictions[0]) == 0: 
                return "true"
            else:
                return "false"


if __name__ == "__main__":
    app.run(host='0.0.0.0')
