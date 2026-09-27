import os, math
import pickle
from contextlib import nullcontext
import torch
import torch.nn as nn
import numpy as np

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
outDir = 'treino' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 4 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
# -----------------------------------------------------------------------------

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)

dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

## Self Attention ##
class MultiHead(nn.Module):
  """ 
    MultiHead otimizado. Todas as Heads computadas utilizando as mesmas matrizes.\n
    Tem um melhor desempenho.
  """

  def __init__(self):
    super().__init__()
    # Para otimizacao, as multiHeads pode ser agrupadas e simplificadas para: 
    self.attention = nn.Linear(E, 3*E, bias=BIAS)
    self.proj = nn.Linear(E, E)

    self.attentioDropout = nn.Dropout(DROPOUT)
    self.residualDropout = nn.Dropout(DROPOUT)

    self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    if not self.flash:
      print("WARNING: Flash Attention requer PyTorch >= 2.0. Versão mais lenta utilizada")
      # causal mask to ensure that attention is only applied to the left in the input sequence
      self.register_buffer("bias", torch.tril(torch.ones(C, C)).view(1, 1, C, C))

  def forward(self, x):
    b,c,e = x.shape
    k, q, v = self.attention(x).split(E, dim=2) # (B,C,E)

    k = k.view(b, c, numberHeads, H).transpose(1, 2) # (B, nh, C, H)
    q = q.view(b, c, numberHeads, H).transpose(1, 2) # (B, nh, C, H)
    v = v.view(b, c, numberHeads, H).transpose(1, 2) # (B, nh, C, H)
    
    if self.flash:
      # efficient attention using Flash Attention CUDA kernels
      out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
    else:
      # Implementacao Manual
      wei = q @ k.transpose(-2,-1) * H **-0.5 # (B, nh, C, H) x (B, nh, H, C) -> (B, nh, C, C)
      wei = wei.masked_fill(self.bias[:,:,:c,:c] == 0, float('-inf')) # (B, nh, C, C)
      wei = nn.softmax(wei, dim=-1) # (B, nh, C, C)
      wei = self.attentioDropout(wei)
      out = wei @ v # (B, nh, C, C) x (B, nh, C, H) -> (B, nh, C, H)

    # Coloca novamente o resultado de todas as Heads em sequencia
    out = out.transpose(1,2).contiguous().view(b,c,E) # (B, nh, C, H) -> (B, C, E)

    # Projecao final
    out = self.proj(out) # (B, C, E) -> (B, C, E)
    out = self.residualDropout(out)
    return out

class FeedFoward(nn.Module):
  '''Linear Layer folowed by non-linearity'''

  def __init__(self):
    super().__init__()
    self.fc = nn.Linear(E, 4 * E, bias=BIAS)
    self.activationFunction = nn.ReLU()
    self.proj = nn.Linear(4 * E, E, bias=BIAS) # Representar o Resíduo e evitar gradient vanishing
    self.dropout = nn.Dropout(DROPOUT)

    self.net = nn.Sequential(
      self.fc,
      self.activationFunction,
      self.proj,   
      self.dropout
    )

  def forward(self, x):
    return self.net(x)

class LayerNorm(nn.Module):
  '''
    Responsável pela normalização das "**linhas**" recebidas.
  '''
  def __init__(self, dim, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))
    self.bias = nn.Parameter(torch.zeros(dim)) if BIAS else None

  # def __call__(self, x):
  #   # calculate the forward pass
  #   xmean = x.mean(1, keepdim=True) # batch mean
  #   xvar = x.var(1, keepdim=True) # batch variance
  #   xhat = (x - xmean) / torch.sqrt(xvar + self.eps) # normalize to unit variance
  #   self.out = self.gamma * xhat + self.beta
  #   return self.out
  # def parameters(self):
  #   return [self.gamma, self.beta]

  def forward(self, input):
        return nn.functional.layer_norm(input, self.weight.shape, self.weight, self.bias, self.eps)
  
class Block(nn.Module):

  def __init__(self):
    super().__init__()
    self.heads = MultiHead()
    self.ffwd = FeedFoward()
    self.ln1 = nn.LayerNorm(E)
    self.ln2 = nn.LayerNorm(E)

  def forward(self, x):
    x = x + self.heads(self.ln1(x)) # (B, C, E) -> (B, C, H*) , H* = E
    x = x + self.ffwd(self.ln2(x)) # (B, C, E)
    return x

class LanguageModel(nn.Module):

  def __init__(self, vocabSize: int):
    super().__init__()

    self.transformer = nn.ModuleDict(dict(
      embeddingTable = nn.Embedding(vocabSize, E),
      positionEmbeddingTable = nn.Embedding(C, E),
      drop = nn.Dropout(DROPOUT),
      # Chamada sequencial do conjunto (SelfAttention -> FeedFoward)
      blocks = nn.ModuleList([Block() for _ in range(BLOCKLAYERS)]),
      ln = LayerNorm(E),
    ))

    self.lm_head = nn.Linear(E, vocabSize, bias=False)
    # Referencia em https://paperswithcode.com/method/weight-tying
    self.transformer.embeddingTable.weight = self.lm_head.weight 

    # Inicializar os pesos do modelo
    self.apply(self._init_weights)
    # Por causa das camadas residuais, aplicar um inicializador especial para elas
    for pn, p in self.named_parameters():
      if pn.endswith('proj.weight'):
        nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * BLOCKLAYERS))

    print("Numero de parametros: %.2fM" % (self.get_num_params()/1e6,))

  def forward(self, x, y=None):
    b, c = x.shape

    tok_emb = self.transformer.embeddingTable(x) # (B, C) -> (B, C, E)
    pos_emb = self.transformer.positionEmbeddingTable(torch.arange(c, dtype=torch.long, device=device)) # (C) -> (C, E)

    x = self.transformer.drop(tok_emb + pos_emb) # (B, C, E) + (C, E) -> (B, C, E)
    for block in self.transformer.blocks:
            x = block(x) # (B, C, E)
    x = self.transformer.ln(x) # (B, C, E)

    logits = self.lm_head(x) # (B, C, V)

    if y == None:
      erro = None
    else:
      # Ajusta a dimensionalidade do logits e do y para o calculo da entropia
      # O calculo de erro espera que o objeto de entrada tenha 2 dimensões: 
      #   Primeiro o nº de valores previstos 
      #   Segundo o número de classes/caracteres/tokens
      # Ou seja, batchs*context X vocabSize
      # O Y é um vetor de comprimento igual ao primeiro valor da entrada com cada valor no range do número de classes/caracteres/tokens
      # Ou seja, será o vetor de comprimento batchs*context com valores de 0 a vocabSize-1
      b, c, vocabSize = logits.shape
      logits = logits.view(b*c, vocabSize).to(device)
      y = y.view(b*c)
      erro = nn.functional.cross_entropy(logits, y).to(device)
    
    return logits, erro
  
  @torch.no_grad()
  def get_num_params(self, non_embedding=True):
    """
    Retorna o número de parâmetros do modelo.
    """
    n_params = sum(p.numel() for p in self.parameters())
    if non_embedding: # Remove os positional embeddings da contagem
      n_params -= self.transformer.positionEmbeddingTable.weight.numel()
    return n_params

  @torch.no_grad()
  def _init_weights(self, module):
    '''Inicializa os pesos das funções Lineares e de Embedding'''
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None:
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      nn.init.normal_(module.weight, mean=0.0, std=0.02)

  
  @torch.no_grad()
  def generate (self, x, newTokens: int):
    # X eh (B, C)
    for _ in range(newTokens):
      # Prediz o proximo caractere do contexto C (baseado no ultimo caractere)
      idx = x[:, -C:] # limita aos ultimos tokens
      logits, _ = self(idx)
      logits = logits[:, -1, :] # (B, C, V) -> (B, V)
      probs = torch.softmax(logits, dim=-1) # (B, V) 
      nextChar = torch.multinomial(probs, num_samples=1) # (B, 1) 
      x = torch.cat((x, nextChar), dim=1) # (B, C+1)
    return x

#### Funções auxiliares ####
dataDir = 'data'
outDir = 'treino'

## Get data dos arquivos binarios ##
def getDataBatchs(tipoDados):
  '''
    Obtêm **B** trecho(s) de tamanho **C** da entrada  do **tipoDados** definida
  '''
  if tipoDados == "treino":
    dados = np.memmap(os.path.join(dataDir, 'train.bin'), dtype=np.uint16, mode='r')
    #dados = torch.tensor(transformar(textoTreino, charToIndex), dtype=torch.long)
  else: 
    dados = np.memmap(os.path.join(dataDir, 'test.bin'), dtype=np.uint16, mode='r')
    #dados = torch.tensor(transformar(textoTest, charToIndex), dtype=torch.long) # textoTest
  n = torch.randint(0, len(dados) - C, (B,)) #Min, Max e quant. de inteiros
  x = torch.stack([torch.from_numpy((dados[i:i+C]).astype(np.int64)) for i in n])
  y = torch.stack([torch.from_numpy((dados[i+1:i+1+C]).astype(np.int64)) for i in n])

  # Caso não seja CUDA, remore pin_memory
  return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)

## Criando o codificador e decodificador ##
def transformar(entrada, padrao):
  '''
  Codifica/Decodifica a entrada baseado no padrão passado. 
  '''
  if (type(entrada) == list):
    return ''.join([padrao[c] for c in entrada])
  else:
    return [padrao[c] for c in entrada]

path = os.path.join(dataDir, 'meta.pkl')
vocabSize = None
if os.path.exists(path):
  with open(path, 'rb') as f:
    meta = pickle.load(f)
  vocabSize = meta['vocab_size']
  print(f"Vocabulário encontrado: {vocabSize} caracteres (inside {path})")

### Hiper parametros ###
# N. de Batchs (B) 
# Tamanho de Contexto (C)
# N. de Embeddings (E)
# HeadSize (H)
hiperParametros = {
  'C': 256,  # Números de caracteres utilizados no contexto
  'B': 64,  # Quantidade de amostras 
  'E': 384, # Dimensão de embedding
  'numberHeads': 6,
  'blockLayers': 6,   # Numero de camadas
  'dropout': 0.2,
  'bias': True
}


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# model
if init_from == 'resume':
  ckpt_path = os.path.join(outDir, 'ckpt.pt')
  checkpoint = torch.load(ckpt_path, map_location=device)
  checkpoint_model_args = checkpoint['hiperParametros']

  for k in ['blockLayers', 'numberHeads', 'B', 'E', 'C', 'bias', 'dropout']:
      hiperParametros[k] = checkpoint_model_args[k]

  B, C, E = hiperParametros['B'], hiperParametros['C'], hiperParametros['E']
  numberHeads = hiperParametros['numberHeads']
  H = E//numberHeads # head Size

  BLOCKLAYERS=hiperParametros['blockLayers']
  BIAS=hiperParametros['bias']
  DROPOUT=hiperParametros['dropout']

  model = LanguageModel(vocabSize)
  state_dict = checkpoint['model']

  ## Erro ja encntrado, mantido por facilidade
  # fix the keys of the state dictionary :(
  # honestly no idea how checkpoints sometimes get this prefix, have to debug more
  unwanted_prefix = '_orig_mod.'
  for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
      state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
  
  model.load_state_dict(state_dict)

model.eval()
model.to(device)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
meta_path = os.path.join('data', 'meta.pkl')
load_meta = os.path.exists(meta_path)

print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
# TODO want to make this more general to arbitrary encoder/decoder schemes
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# encode the beginning of the prompt
if start.startswith('FILE:'):
  with open(start[5:], 'r', encoding='utf-8') as f:
    start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
  with ctx:
    for k in range(num_samples):
      y = model.generate(x, max_new_tokens)
      print(decode(y[0].tolist()))
      print('---------------')
