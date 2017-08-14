import numpy as np

sentence = "S I really really like you E"
vocab = ['S', 'I', 'really', 'like', 'you', 'E']
one_hot = np.array([
  [1, 0, 0, 0, 0, 0], # S
  [0, 1, 0, 0, 0, 0], # I
  [0, 0, 1, 0, 0, 0], # really
  [0, 0, 0, 1, 0, 0], # like
  [0, 0, 0, 0, 1, 0], # you
  [0, 0, 0, 0, 0, 1]  # E
])
sentence_by_one_hot_idx = [
  0, # S
  1, # I
  2, # really
  2, # really
  3, # like
  4, # you
  5  # E
]
vocab_by_one_hot_idx = [
  0, 1, 2, 3, 4, 5
]


v = len(vocab) # = len(input)
n = 3 # dimension of input word vector
c = 1 # window size
V = np.array([
  [0.01, 0.01, 0.01],
  [0.02, 0.02, 0.02],
  [0.03, 0.03, 0.03],
  [0.04, 0.04, 0.04],
  [0.05, 0.05, 0.05],
  [0.06, 0.06, 0.06]
])
U = np.array([
  [0.11, 0.11, 0.11],
  [0.12, 0.12, 0.12],
  [0.13, 0.13, 0.13],
  [0.14, 0.14, 0.14],
  [0.15, 0.15, 0.15],
  [0.16, 0.16, 0.16]
])

def make_combis(sentence):
  combis = []
  size = len(sentence)
  for idx, v in enumerate(sentence):
    if idx == 0 or idx == size - 1:
      continue
    combis.append([sentence[idx-1], sentence[idx]])
    combis.append([sentence[idx+1], sentence[idx]])
  return np.array(combis)

# very naive one
def get_prob(combis, U, V):
  exps = []
  for j in vocab_by_one_hot_idx:
    exp = []
    for i in vocab_by_one_hot_idx:
      exp.append(np.exp(np.dot(U[i], V[j])))
    exps.append(sum(exp))
  values = []
  for combi in combis:
    o, c = combi
    top = np.exp(np.dot(U[o], V[c]))
    bottom = exps[c]
    value = np.log(top/bottom)
    values.append(value)
  return sum(values)

def get_prob2(combis, U, V):
  UV = np.dot(U, V.T)
  loss = 0
  for combi in combis:
    o, c = combi
    top = np.exp(UV[o][c])
    bottom = sum(np.exp(UV.T[c]))
    each_loss = np.log(top/bottom)
    result = o, c, top, bottom, each_loss
    loss += each_loss
  return loss

combis = make_combis(sentence_by_one_hot_idx)

prob = get_prob(combis, U, V) # -17.9155
prob2 = get_prob2(combis, U, V) # -17.9155

np.allclose(prob, prob2)


x = np.concatenate((U, V), axis=0)
it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

while not it.finished:
  ix = it.multi_index
  print(ix)


















def exp_dot(w_o, w_c):
  v = np.dot(w_c,V)
  u = np.dot(w_o,U)
  print(v, u)
  dot = np.dot(v, u)
  print(dot)
  exp = np.exp(dot)
  print(exp)
  return exp



exp['I_S']            = exp_dot(oh_I, oh_S)
exp['I_really']       = exp_dot(oh_I, oh_really)

exp['really_I']       = exp_dot(oh_really, oh_I)
exp['really_really']  = exp_dot(oh_really, oh_really)

exp['really_really']  = exp_dot(oh_really, oh_really)
exp['really_like']    = exp_dot(oh_really, oh_like)

exp['like_really']    = exp_dot(oh_like, oh_really)
exp['like_you']       = exp_dot(oh_like, oh_you)

exp['you_like']       = exp_dot(oh_you, oh_like)
exp['you_E']          = exp_dot(oh_you, oh_E)

SS = sum(exp.values()) # 8.119

loss_each = {}

loss_each['I_S']          = np.log(exp['I_S']/SS)
loss_each['I_really']     = np.log(exp['I_really']/SS)

loss_each['really_I']     = np.log(exp['really_I']/SS)
loss_each['really_like']  = np.log(exp['really_like']/SS)

loss_each['like_really']     = np.log(exp['like_really']/SS)
loss_each['like_you']     = np.log(exp['like_you']/SS)

loss_each['you_like']     = np.log(exp['you_like']/SS)
loss_each['you_E']     = np.log(exp['you_E']/SS)

loss = sum(loss_each.values()) # -16.636

# This is too verbose, we simplify this using matrix.
def make_combis(one_hot):
  combis_word = []
  combis_idx = []
  for idx, v in enumerate(vocab):
    if idx == 0 or idx == len(one_hot) - 1:
      continue
    combis_word.append([vocab[idx+1], vocab[idx]])
    combis_idx.append([idx+1, idx])
    combis_word.append([vocab[idx-1], vocab[idx]])
    combis_idx.append([idx-1, idx])
  return np.array(combis_word), np.array(combis_idx)

combis_word, combis_idx = make_combis(one_hot)

def loss(one_hot):
  np.dot(np.dot(one_hot, V), U.T)
