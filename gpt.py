import torch
import torch.nn as nn
import numpy as np
import os, pickle, math, time
import inspect
from contextlib import nullcontext

# Define o valor para geração de dados aleatórios (testes iguais)
torch.manual_seed(23092026)
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
      out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=DROPOUT if TRAINING else 0, is_causal=True)
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

  def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    # Obtem todos os parametros e filtra os que precisam de gradiente
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    
    # Criar grupos de otimização. Parametros 2D+ terão weight decay, outros não.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
    otimizationGroups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]

    print(f"Parametros com weight decay: {len(decay_params)}, com {sum(p.numel() for p in decay_params):,} parametros.\nParametros sem weight decay: {len(nodecay_params)}, com {sum(p.numel() for p in nodecay_params):,} parametros.")

    # Cria AdamW optimizer e usa a versao fused, se disponivel
    use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters) and (device_type == 'cuda')
    extra_args = dict(fused=True) if use_fused else dict()
    print(f"Possível usar fused AdamW: {use_fused}")

    optimizer = torch.optim.AdamW(otimizationGroups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer
  
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

@torch.no_grad()
def estimate_loss():
  out = []
  model.eval()
  for split in ['treino', 'teste']:
    losses = torch.zeros(EVAL_ITERS)
    for k in range(EVAL_ITERS):
      X, Y = getDataBatchs(split)
      with ctx:
        logits, loss = model(X, Y)
      losses[k] = loss.item()
    out.append(losses.mean())
  model.train()
  return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
      return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
      return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

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
# adamw optimizer
learning_rate = 1e-4 # max learning rate
weight_decay = 1e-1
warmup_iters = 2000
lr_decay_iters = 4000
min_lr = 1e-6

beta1 = 0.9
beta2 = 0.99

TRAINING = True
compile = True
decay_lr = True 

iter_num = 0
best_val_loss = 1e9
tipoDeInicio = 'resume'

### Inicializando o modelo ###
print("Inicializando modelo...")
if vocabSize is None:
    raise Exception("Vocabulario nao encontrado!")

if tipoDeInicio == 'zero':
  B, C, E = hiperParametros['B'], hiperParametros['C'], hiperParametros['E']
  numberHeads = hiperParametros['numberHeads']
  H = E//numberHeads # head Size

  BLOCKLAYERS=hiperParametros['blockLayers']
  BIAS=hiperParametros['bias']
  DROPOUT=hiperParametros['dropout']

  model = LanguageModel(vocabSize)

  iter_num = 0
  best_val_loss = 1e3
else:
  print(f"Continuando treinamento...")
  # Continuar treinamento a partir do checkpoint.
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
  iter_num = checkpoint['iter_num']
  best_val_loss = checkpoint['best_val_loss']

model = model.to(device)

# Otimizador
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if tipoDeInicio != 'zero':
  optimizer.load_state_dict(checkpoint['optimizer'])

checkpoint = None # Libera Memoria

# Compila o modelo
if compile:
  print("Compilando o modelo... (leva aproximadamente 1 minuto)")
  model = torch.compile(model) # requires PyTorch 2.0

x, y = getDataBatchs('treino')  # X, Y para treino
logits, loss = model(x, y)
print("Erro: ", loss)

MAXSTEPS = 6000
EVAL_ITERS = 100
raw_model = model
t0 = time.time()
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

while True:
  lr = get_lr(iter_num) if decay_lr else learning_rate
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  
  # every once in a while evaluate the loss on train and val sets
  if iter_num % EVAL_ITERS == 0:
    losses = estimate_loss()
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    print(f"Step {iter_num}: Train loss: {losses[0]:.4f}, Test loss: {losses[1]:.4f}, Tempo: {dt*1000:.2f} ms, lr: {lr}")

    if losses[1] < best_val_loss:
      best_val_loss = losses[1]
      if iter_num > 0:
        checkpoint = {
          'model': raw_model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'hiperParametros': hiperParametros,
          'iter_num': iter_num,
          'best_val_loss': best_val_loss,
        }
        print(f"Salvando o checkpoint {outDir}")
        torch.save(checkpoint, os.path.join(outDir, 'ckpt.pt'))

  with ctx:
    logits, loss = model(x, y)

  x, y = getDataBatchs('treino')
  # backward pass, with gradient scaling if training in fp16
  scaler.scale(loss).backward()
  scaler.step(optimizer)
  scaler.update()
  optimizer.zero_grad(set_to_none=True)

  iter_num += 1

  if iter_num > MAXSTEPS:
    break

  ## Gerar texto após treinar o modelo ##
  # idx = torch.tensor([[0]], dtype=torch.long, device=device)
  # print(transformar(model.generate(idx, 1000)[0].tolist(), indexToChar))