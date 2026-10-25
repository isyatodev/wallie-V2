# Treino da Luna em GPU na nuvem

Este pacote leva o dataset da Luna para um runtime com GPU (Colab ou Kaggle),
treina um Piper `pt_BR-luna-medium` e devolve o par `.onnx` + `.onnx.json`
pronto para anexar no Wallie.

> **Antes de tudo:** confira `BUNDLE.json`. Se `trainable` for `false`, o
> dataset ainda está abaixo de **60 s** e o treino **vai recusar** — com razão:
> o Piper começa de pesos aleatórios e com poucos segundos só aprende ruído.
> O mínimo é 60 s; o recomendado é **10–30 min** de fala.

## 1. Preparar o dataset (na máquina com o Wallie)

O pacote atual é um retrato do que está em `voices/ported/luna-piper/dataset`.
Para acrescentar áudio, use o **Voice Lab → Convert to a local engine** e
importe mais clipes da Luna; o pacote é reconstruído. Depois gere o bundle de
novo:

```bash
python scripts/build_luna_cloud_bundle.py
```

Isso escreve `dist/luna-cloud/` e `dist/luna-cloud.zip`. O zip não inclui
checkpoints nem o `.onnx` antigo (o modelo de ruído); só o dataset, o manifesto,
o trainer e o núcleo de alinhamento.

## 2. Colab (T4 grátis)

1. Acesse <https://colab.research.google.com>, crie um notebook e faça upload
   de `dist/luna-cloud.zip` (ícone de pasta → Upload) e de
   `luna_cloud.ipynb` (File → Upload notebook).
2. **Runtime → Change runtime type → T4 GPU** e depois **Runtime → Restart**.
3. Rode as células do notebook na ordem. Elas instalam
   `piper-tts[train]==1.8.0` + `onnxscript`, descompactam o bundle, treinam e
   baixam o resultado.

Se preferir a linha de comando, dentro do Colab:

```bash
pip install -q -r /content/luna-cloud/requirements.txt
python /content/luna-cloud/train_luna.py \
  --bundle /content/luna-cloud \
  --out /content/luna-output \
  --epochs 150 --batch-size 16
```

O resultado é `/content/luna-output/pt_BR-luna-medium-cloud.zip`, com:

- `pt_BR-luna-medium.onnx`
- `pt_BR-luna-medium.onnx.json`
- `train.json` (clips, segundos, epochs, GPU, versão do Piper)

## 3. Kaggle (T4/P100)

1. Faça upload de `luna-cloud.zip` em **Datasets** (ou copie para
   `/kaggle/working`).
2. No notebook, ative **Settings → Accelerator → GPU T4 x2** (ou P100).
3. Rode o mesmo comando, trocando os caminhos:

```bash
pip install -q -r /kaggle/working/luna-cloud/requirements.txt
python /kaggle/working/luna-cloud/train_luna.py \
  --bundle /kaggle/working/luna-cloud \
  --out /kaggle/working/luna-output \
  --epochs 150 --batch-size 16
```

O zip sai em `/kaggle/working/luna-output/` e pode ser baixado pelo painel
**Output**.

## 4. Tempo esperado

O custo por epoch depende do **número de clipes**, não só dos minutos: o VITS
treina em segmentos fixos de 8192 amostras, então cada epoch tem
`clipes / batch` passos. Numa T4, 150 epochs com ~60 clipes costumam levar
**algumas dezenas de minutos**; com muitos clipes, 1 h ou mais. O
`train_luna.py` mostra o split e o batch efetivo antes de começar, e o log do
Lightning mostra o tempo real de cada epoch.

Se der **out of memory**, reduza `--batch-size 8`. Se a GPU não aparecer, o
script para com a instrução de trocar o runtime.

## 5. Trazer o modelo de volta e anexar

Baixe o `*-cloud.zip` e, na máquina do Wallie:

```bash
python scripts/attach_cloud_voice.py baixados/pt_BR-luna-medium-cloud.zip \
  --profile luna --activate
```

O helper:

- lê o `train.json` e **recusa** qualquer modelo treinado com menos de 60 s;
- copia `.onnx` e `.onnx.json` para `voices/`;
- registra a voz na biblioteca do perfil;
- com `--activate`, aponta `tts.voice_id` e `tts.piper_model_path` do perfil
  `luna` para o modelo novo.

Reinicie o dashboard depois para ele descartar a configuração antiga em
memória. Sem `--activate`, a voz fica salva na biblioteca e você escolhe no
Voice Lab.

## Solução de problemas

| Sintoma | Causa / correção |
|---|---|
| `this dataset has ... below 60s` | Falta áudio. Importe mais clipes e rode o builder de novo. |
| `no CUDA GPU is visible` | Runtime sem GPU; troque para T4/P100 e reinicie. |
| `ModuleNotFoundError: monotonic_align` | O bundle foi montado sem `monotonic_core.py`; rode o builder de novo. |
| Export falha no `torch.onnx.export` | O script aplica `dynamo=False` no torch ≥ 2.9; confirme `torch >= 2.9` e `onnxscript` instalado. |
| `CUDA out of memory` | `--batch-size 8` (ou 4). |
| O `.onnx` sai, mas o áudio é ruído | Dataset pequeno demais. Menos de 60 s nunca funciona; abaixo de 10 min ainda pode ficar áspero. |

## Privacidade

O zip contém áudio da sua voz. Não suba para um notebook público nem para um
dataset público do Kaggle; use o armazenamento privado da sua conta e apague o
arquivo depois de treinar.
