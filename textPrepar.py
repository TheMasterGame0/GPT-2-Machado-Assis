# from datasets import load_dataset
# ds = load_dataset("giseldo/dataset-machado-assis")

# print(ds["train"])

# ## Textos do tipo conto ##
# with open("textos/conto/contosFluminenses.txt", "r", encoding="utf-8") as f:
#     contosFluminenses = f.read()
# with open("textos/conto/historiasMeiaNoite.txt", "r", encoding="utf-8") as f:
#     historiasMeiaNoite = f.read()
# with open("textos/conto/historiasSemData.txt", "r", encoding="utf-8") as f:
#     historiasSemData = f.read()
# with open("textos/conto/paginasRecolhidas.txt", "r", encoding="utf-8") as f:
#     paginasRecolhidas = f.read()
# with open("textos/conto/papeisAvulsos.txt", "r", encoding="utf-8") as f:
#     papeisAvulsos = f.read()
# with open("textos/conto/reliquias.txt", "r", encoding="utf-8") as f:
#     reliquias = f.read()
# with open("textos/conto/variasHistorias.txt", "r", encoding="utf-8") as f:
#     variasHistorias = f.read()

# ## Escreve no arquivo ##
# with open("textos/conto/textoCompleto.txt", "w", encoding="utf-8") as f:
#     f.write(contosFluminenses)
#     f.write(historiasMeiaNoite)
#     f.write(historiasSemData)
#     f.write(paginasRecolhidas)
#     f.write(papeisAvulsos)
#     f.write(reliquias)
#     f.write(variasHistorias)

# Textos de crítica #
# enderecos = ["textos/critica/cantosFantasias.txt","textos/critica/castroAlves.txt", "textos/critica/cenasVida.txt", "textos/critica/colombo.txt", "textos/critica/contituinteSombraLuz.txt", "textos/critica/contosSeletos.txt", "textos/critica/criticaTeatral.txt", "textos/critica/cultoDever.txt", "textos/critica/diarioRJ.txt", "textos/critica/discursosAcademia.txt", "textos/critica/ea.txt", "textos/critica/eduardo.txt", "textos/critica/fagundes.txt", "textos/critica/floresFrutos.txt", "textos/critica/garrett.txt", "textos/critica/guarani.txt", "textos/critica/guilhermeMalta.txt", "textos/critica/harmoniasErrantes.txt", "textos/critica/henriquieta.txt", "textos/critica/idealCritico.txt", "textos/critica/ideiasTeatro.txt", "textos/critica/inspiracoesClaustro.txt", "textos/critica/instintoNacionalidade.txt", "textos/critica/iracema.txt", "textos/critica/joaquim_1.txt", "textos/critica/lira20Anos.txt", "textos/critica/mae.txt", "textos/critica/magalhaes.txt", "textos/critica/meridionais.txt", "textos/critica/miragens.txt", "textos/critica/nevoasMatutinas.txt", "textos/critica/novaGeracao.txt", "textos/critica/oliveiraLimna.txt", "textos/critica/pareceresConservatorioDramatico.txt", "textos/critica/passadoPresenteFuturo.txt", "textos/critica/peregrinacao.txt", "textos/critica/primoBasilio.txt", "textos/critica/procelarias.txt", "textos/critica/proposito.txt", "textos/critica/revelacoes.txt", "textos/critica/revistaDramatica.txt", "textos/critica/revistaTeatros.txt", "textos/critica/sinfonias.txt", "textos/critica/suplicio.txt", "textos/critica/tiposQuadros.txt"]

# with open("textos/critica/textoCompleto.txt", "w", encoding="utf-8") as w:
#   for link in enderecos:
#     with open(link, "r", encoding="utf-8") as f:
#       texto = f.read()
#     f.close()

#     w.write(texto)
#     print("texto Escrito!")

## Textos de cronica ##
# enderecos = ['textos/cronica/aoAcaso.txt','textos/cronica/aquarelas.txt','textos/cronica/badaladas.txt','textos/cronica/balas.txt','textos/cronica/bonsDias.txt','textos/cronica/cartasFluminenses.txt','textos/cronica/cherchez.txt','textos/cronica/comentariosSemana.txt','textos/cronica/cronicasFuturo.txt','textos/cronica/drSemana.txt','textos/cronica/entre92-94.txt','textos/cronica/futuro.txt','textos/cronica/henrique.txt','textos/cronica/henriqueChaves.txt','textos/cronica/historia15dias.txt','textos/cronica/historia30dias.txt','textos/cronica/joaquim.txt','textos/cronica/jornalLivro.txt','textos/cronica/joseAlencar.txt','textos/cronica/notasSemanais.txt','textos/cronica/reforma.txt','textos/cronica/semana.txt','textos/cronica/velhoSenado.txt','textos/cronica/vicondeCastilho.txt']

# with open("textos/cronica/textoCompleto.txt", "w", encoding="utf-8") as w:
#   for link in enderecos:
#     with open(link, "r", encoding="utf-8") as f:
#       texto = f.read()
#     f.close()

#     w.write(texto)
#     print("texto Escrito!")

## Textos de miscelanea ##
# enderecos = ['textos/miscelanea/cartaBispoRJ.txt','textos/miscelanea/cartaImprensa.txt','textos/miscelanea/estatua.txt','textos/miscelanea/franciscoOtaviano.txt','textos/miscelanea/goncalvesDias.txt','textos/miscelanea/imortais.txt','textos/miscelanea/paixao.txt','textos/miscelanea/pedroLuis.txt','textos/miscelanea/quedaMulheres.txt','textos/miscelanea/secretariaAgricultura.txt']

# with open("textos/miscelanea/textoCompleto.txt", "w", encoding="utf-8") as w:
#   for link in enderecos:
#     with open(link, "r", encoding="utf-8") as f:
#       texto = f.read()
#     f.close()

#     w.write(texto)
#     print("texto Escrito!")

## Textos de poesia ##
# enderecos = ['textos/poesia/almada.txt','textos/poesia/americanas.txt','textos/poesia/crisalidas.txt','textos/poesia/dispersas.txt','textos/poesia/falenas.txt','textos/poesia/gazeta.txt','textos/poesia/ocidentais.txt']

# with open("textos/poesia/textoCompleto.txt", "w", encoding="utf-8") as w:
#   for link in enderecos:
#     with open(link, "r", encoding="utf-8") as f:
#       texto = f.read()
#     f.close()

#     w.write(texto)
#     print("texto Escrito!")

## Textos de romance ##
# enderecos = [
# 'textos/romance/casaVelha.txt',
# 'textos/romance/domCasmurro.txt',
# 'textos/romance/esau.txt',
# 'textos/romance/helena.txt',
# 'textos/romance/iaia.txt',
# 'textos/romance/maoLuva.txt',
# 'textos/romance/memorial-de-aires.txt',
# 'textos/romance/memoriasBras.txt',
# 'textos/romance/quincas.txt',
# 'textos/romance/ressurreicao.txt'
# ]

# with open("textos/romance/textoCompleto.txt", "w", encoding="utf-8") as w:
#   for link in enderecos:
#     with open(link, "r", encoding="utf-8") as f:
#       texto = f.read()
#     f.close()

#     w.write(texto)
#     print("texto Escrito!")


# export to bin files
import numpy as np
import os
import pickle

## Criando o codificador e decodificador ##
def transformar(entrada, padrao):
  '''
  Codifica/Decodifica a entrada baseado no padrão passado. 
  '''
  if (type(entrada) == list):
    return ''.join([padrao[c] for c in entrada])
  else:
    return [padrao[c] for c in entrada]

## Pegando texto para treinamento ##
with open("textos/conto/textoCompleto.txt", "r", encoding="utf-8") as f:
  texto1 = f.read()
f.close()
print("Contos!")

with open("textos/critica/textoCompleto.txt", "r", encoding="utf-8") as f:
  texto2 = f.read()
f.close()
print("Criticas!")

with open("textos/cronica/textoCompleto.txt", "r", encoding="utf-8") as f:
  texto3 = f.read()
f.close()
print("Cronica!")

with open("textos/miscelanea/textoCompleto.txt", "r", encoding="utf-8") as f:
  texto4 = f.read()
f.close()
print("Miscelanea!")

with open("textos/poesia/textoCompleto.txt", "r", encoding="utf-8") as f:
  texto5 = f.read()
f.close()
print("Poesia!")

with open("textos/romance/textoCompleto.txt", "r", encoding="utf-8") as f:
  texto6 = f.read()
f.close()
print("Romance!")


## Análise dos dados ##
textoTreino = ''
textoTest = ''

for texto in [texto1, texto2, texto3, texto4, texto5, texto6]:
  count = texto.count("\n")
  tmp = texto.split("\n")
  textoTreino += '\n'.join(tmp[:int(count*0.75)],)
  textoTest += '\n'.join(tmp[int(count*0.75)+1:])

print("Total de caracteres Treino: ", len(textoTreino))
print("Total de caracteres Teste: ", len(textoTest))
print("Total de linhas Treino: ", textoTreino.count("\n"))

caracteresUnicos = sorted(list(set(textoTreino+textoTest)))
print("Caracteres únicos: ", ''.join(caracteresUnicos), "\nTotal de caracteres únicos: ", len(caracteresUnicos))

## Dicionarios para one-hot enconding ##
charToIndex, indexToChar = dict(), dict()
for i, ch in enumerate(caracteresUnicos):
  charToIndex[ch] = i
  indexToChar[i] = ch

train_ids = np.array(transformar(textoTreino, charToIndex), dtype=np.uint16)
test_ids = np.array(transformar(textoTest, charToIndex), dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.join(os.path.dirname(__file__), 'data'), 'train.bin'))
test_ids.tofile(os.path.join(os.path.join(os.path.dirname(__file__), 'data'), 'test.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': len(caracteresUnicos),
    'itos': indexToChar,
    'stoi': charToIndex,
}
with open(os.path.join(os.path.dirname(__file__), 'data', 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)