import util as util
import os
import re
import torch
from BiLSTM_CRF import BiLSTM_CRF
import codecs
import config

_, word_to_ix, tag_to_ix = util.data_prepare(config.TRAIN_DATA_PATH)
OrderedDict = torch.load(config.MODEL_PATH)
len_ = len(OrderedDict["word_embeds.weight"])

model = BiLSTM_CRF(len_, tag_to_ix, config.EMBEDDING_DIM, config.HIDDEN_DIM)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.eval()

ix_to_tag = {v: k for k, v in tag_to_ix.items()}
print(ix_to_tag)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
test_sent = '中国邮政10日在拉萨发行《川藏青藏公路建成通车六十五周年》纪念邮票1套2枚'
print(len(test_sent))
test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in test_sent]
test_res_ = torch.tensor(test_id, dtype=torch.long)
eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]
print(len(eval_res))
print(util.tag2word(test_sent, eval_res))

# for parent, dirnames, filenames in os.walk(NEWS_PATH):
#     for filename in filenames:
#         file_path_in = os.path.join(parent, filename)
#         print("transforming.....", file_path_in)
#         file_path_out = os.path.join(parent, r"..\\NERout\\ner_{}".format(filename))
#         test_d = codecs.open(file_path_in, 'r', 'utf-8').read()
#         test_d = strQ2B(test_d)
#         test_d = re.sub(r"\n\n", "", test_d)
#         test_d_Lst = cut_sentence(test_d)
#         result = set()
#
#         for sent in test_d_Lst:
#             sent = cleanSent(sent)
#             if len(sent) < 4: continue
#             test_id = [word_to_ix[i] if i in word_to_ix.keys() else 2 for i in sent]
#             test_res_ = torch.tensor(test_id, dtype=torch.long)
#             eval_res = [ix_to_tag[t] for t in model(test_res_)[1]]
#
#             tag2word_res = util.tag2word(sent,eval_res)
#             for s in tag2word_res:
#                 result.add(s)
#
#         with codecs.open(file_path_out, 'w+', 'utf-8') as surveyp:
#             surveyp.write(",\n".join(result))

# # #['O', 'O', 'O', 'O', 'B_PERSON', 'E_PERSON', 'O', 'B_PERSON', 'E_PERSON', 'O', 'O', 'B_LOCATION', 'E_LOCATION', 'O', 'B_LOCATION', 'M_LOCATION', 'M_LOCATION', 'E_LOCATION', 'O', 'O', 'O', 'O', 'O', 'O']