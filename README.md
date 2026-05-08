# CNN — GoogLeNet Inception V1: Classificação Binária de Pneumonia

![Status](https://img.shields.io/badge/Status-Concluído-success)
![Python](https://img.shields.io/badge/Python-3.x-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)

Adaptação da arquitetura **GoogLeNet Inception V1** para classificação binária de radiografias de tórax, distinguindo casos de **Pneumonia** e pulmões **Normais**. Desenvolvido como trabalho acadêmico de Deep Learning, executado inteiramente no **Kaggle** (GPU NVIDIA Tesla T4 ×2) e versionado via **GitHub**.

---

## Índice

- [Visão Geral](#visão-geral)
- [Dataset](#dataset)
- [Arquitetura](#arquitetura)
- [Modificações do Código Base](#modificações-do-código-base)
- [Regularização e Balanceamento](#regularização-e-balanceamento)
- [Métricas e Callbacks](#métricas-e-callbacks)
- [Histórico de Experimentos](#histórico-de-experimentos)
- [Resultado Final — O Modelo "Hipocondríaco"](#resultado-final--o-modelo-hipocondríaco)
- [Validação em Imagens Externas](#validação-em-imagens-externas)
- [Trabalhos Futuros](#trabalhos-futuros)
- [Como Reproduzir](#como-reproduzir)
- [Estrutura do Repositório](#estrutura-do-repositório)
- [Dependências](#dependências)
- [Referência do Dataset](#referência-do-dataset)

---

## Visão Geral

O projeto parte de uma implementação de referência do GoogLeNet (InceptionV1) fornecida em ambiente acadêmico, originalmente treinada no CIFAR-10 (10 classes). O objetivo foi adaptar essa arquitetura para um problema médico binário — detectar **Pneumonia** em radiografias de tórax — documentando sistematicamente **11 experimentos** que exploraram os efeitos de diferentes hiperparâmetros e estratégias de treinamento.

**Melhor resultado (Teste 10):**

| Métrica | Valor |
|---|---|
| Acurácia de validação | **84.13%** |
| AUC | **0.9224** |
| Loss de validação | **0.3187** |
| Falsos Negativos (FN) | **41** |
| Falsos Positivos (FP) | **58** |

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — Kermany et al., 2018

| Atributo | Detalhe |
|---|---|
| **Fonte** | [Kaggle — paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| **Total de imagens** | ~5.856 |
| **Classes** | `NORMAL` (0) e `PNEUMONIA` (1) |
| **Split** | `train` / `test` / `val` já fornecidos |
| **Resolução de entrada** | Redimensionado para `32×32×3` (RGB) |
| **Desbalanceamento** | ~74% Pneumonia / ~26% Normal no treino |

As imagens são radiografias frontais de tórax (posteroanterior), capturadas com o mesmo protocolo clínico — o que as torna um dataset de **baixa variabilidade visual**, ideal para testes de arquitetura.

---

## Arquitetura

A rede utilizada é a **GoogLeNet Inception V1**, adaptada para classificação binária. O modelo possui **duas saídas**:

- **Saída principal (`output`):** usada para inferência final
- **Saída auxiliar (`auxilliary_output_1`):** ativa apenas durante o treinamento para combater o vanishing gradient, com peso de contribuição de `0.3` na loss total

```
Input (32×32×3)
      │
      ├── Conv 3×3 / stride 1 (64 filtros)   ← kernel e stride modificados
      ├── MaxPool 3×3 / stride 2
      ├── Conv 1×1 → Conv 3×3 (192 filtros)
      ├── MaxPool 3×3 / stride 2
      ├── Inception 3a → Inception 3b
      ├── MaxPool 3×3 / stride 2
      │        │
      │   [Saída Auxiliar]
      │   AveragePool → Conv 1×1 → Dense(2048) → Dropout → Dense(1, sigmoid)
      │
      ├── Inception 5a → Inception 5b
      ├── GlobalAveragePooling
      ├── Dropout(0.2)
      └── Dense(1, sigmoid)  ← Saída Principal
```

Cada **Inception Module** concatena paralelamente convoluções `1×1`, `3×3`, `5×5` e um `MaxPool`, capturando características em múltiplas escalas espaciais.

---

## Modificações do Código Base

| # | O que foi alterado | Original | Novo valor | Justificativa |
|---|---|---|---|---|
| 1 | `kernel_size` (1ª Conv2D) | `(7, 7)` | `(3, 3)` | Kernel 7×7 cobre 22% de uma imagem 32×32 e descarta detalhes logo na 1ª camada |
| 2 | `stride` (1ª Conv2D) | `(2, 2)` | `(1, 1)` | Stride 1 preserva mais resolução espacial em imagens de entrada pequenas |
| 3 | `epochs` | `100` | `500` (com Early Stopping) | Permite aprendizado profundo sem risco de overfitting manual |
| 4 | Saída principal | `Dense(10, 'softmax')` | `Dense(1, 'sigmoid')` | Problema binário: 1 neurônio com sigmoid mapeia a saída para probabilidade em [0, 1] |
| 5 | Saída auxiliar | `Dense(10, 'softmax')` | `Dense(1, 'sigmoid')` | Mesma razão do item 4 |
| 6 | Função de perda | `categorical_crossentropy` | `binary_crossentropy` | Perda matematicamente correta para saída sigmoid binária |
| 7 | Dataset | `cifar10.load_data()` | `load_chestxray_data()` | Troca por dataset de imagens médicas com 2 classes |
| 8 | Regularização | Ausente | Data Augmentation + L2 (0.0005) | Prevenção de overfitting e melhoria de generalização |
| 9 | Métricas e callbacks | `accuracy` padrão | AUC, Precision, Recall + Early Stopping | Avaliação mais rica em contexto médico desbalanceado |
| 10 | Balanceamento | Ausente | `sample_weight` suavizado (×0.65) | Compensa o desbalanceamento 3:1 do dataset |

---

## Regularização e Balanceamento

### Data Augmentation

Aplicado via `ImageDataGenerator` em tempo real durante o treinamento, sem duplicar dados em disco. As três transformações utilizadas foram:

- **Flip** (`horizontal_flip=True`) — espelhamento horizontal, simulando orientações diferentes de captura
- **Rotation** (`rotation_range=10°`) — rotação leve, simulando variações no posicionamento do paciente
- **Zoom** (`zoom_range=0.1`) — zoom suave, simulando distâncias variadas do equipamento

O efeito prático é que o modelo nunca vê a mesma imagem duas vezes da mesma forma, forçando o aprendizado de características mais **generalizáveis** em vez de memorizar os pixels exatos do treino.

### Regularização L2 (0.0005)

Aplicada na camada `Dense` da saída auxiliar. Penaliza pesos extremos, distribuindo a responsabilidade de decisão entre mais neurônios.

O valor `0.0005` foi escolhido após o Teste 04 revelar que `0.001` combinado com Augmentation causava **underfitting severo** (37.66% de acurácia): a penalização era forte demais para uma rede já sobrecarregada com variações artificiais. Reduzir para `0.0005` suavizou a punição e permitiu que a rede aprendesse por até 96 épocas no Teste 10.

> **Regra geral:** quanto mais Augmentation, menor precisa ser o L2 — os dois mecanismos competem pelo controle da complexidade do modelo.

### Balanceamento com `sample_weight` suavizado

O `compute_class_weight('balanced')` gerava os pesos brutos, inversamente proporcionais à frequência de cada classe:

```
Pesos originais → NORMAL: 1.945 | PNEUMONIA: 0.673
```

Esses pesos causavam um viés extremo, gerando 156 Falsos Positivos para garantir apenas 11 Falsos Negativos (Teste 07). Multiplicar apenas o peso de NORMAL por **0.65** suavizou a correção:

```
Pesos ajustados → NORMAL: 1.264 | PNEUMONIA: 0.673
```

O resultado foi um modelo mais equilibrado, com FP caindo de 156 para 58, sem prejudicar significativamente o Recall.

---

## Métricas e Callbacks

### Métricas adicionadas

| Métrica | O que mede | Relevância médica |
|---|---|---|
| **AUC** | Capacidade de separação entre classes em todos os limiares possíveis | Não é afetada pelo desbalanceamento; 0.5 = aleatório, 1.0 = perfeito |
| **Recall** (Sensibilidade) | `TP / (TP + FN)` — dos doentes, quantos foram detectados | Crítico: um FN significa um paciente doente liberado sem tratamento |
| **Precision** | `TP / (TP + FP)` — das previsões positivas, quantas estavam certas | Controla o volume de alarmes falsos desnecessários |

### Callbacks

- **`EarlyStopping`** — monitora `val_auc` com `patience=5` e `restore_best_weights=True`: interrompe o treinamento quando a AUC de validação para de melhorar e restaura automaticamente os pesos da melhor época
- **`LearningRateScheduler`** — aplica decaimento exponencial da taxa de aprendizado a cada 5 épocas (`drop=0.96`), seguindo a configuração original do código base

---

## Histórico de Experimentos

| Teste | Épocas | Acc. Treino | Acc. Val. | AUC | Loss Val. | Observação |
|---|---|---|---|---|---|---|
| **01** Baseline adaptado | 20 | 90.31% | 79.49% | — | 0.6572 | 4/5 erros no top-5. Viés para classe majoritária. |
| **02** + 100 épocas | 100 | 97.32% | 74.68% | — | 1.1724 | Loss dobrou. **Overfitting claro.** |
| **03** + `sample_weight` | 15 | 78.21% | 63.94% | — | 0.8766 | Colapso. Errou 5/5. Pesos desequilibrados. |
| **04** + Augmentation + L2 | 15 | 58.57% | 37.66% | — | 0.7058 | **Underfitting severo.** Duas regularizações simultâneas. |
| **05** + Métricas + Early Stop | 15 | 48.26% | 55.61% | 0.77 | 0.6931 | AUC promissora, mas 271 FN. Viés invertido. |
| **06** + 100 épocas máximas | 100→29 | 58.99% | 77.08% | 0.87 | 0.5835 | **Primeiro equilíbrio real.** Stop na época 24. |
| **07** + URLs externas | 50→22 | 61.39% | 73.24% | 0.86 | 0.6101 | Apenas **11 FN**. Alta sensibilidade médica. 156 FP. |
| **08** Repetição T07 | 50→30 | 62.26% | 70.99% | 0.869 | 0.4992 | Reprodutível. 9 FN. Confirma perfil de triagem. |
| **09** Peso ×0.65 | 50→49 | 82.44% | 83.49% | 0.9224 | 0.3574 | **Modelo otimizado.** FP=37. Acurácia recorde. |
| **10** L2=0.0005 + 500 épocas | 500→96 | 88.75% | **84.13%** | **0.9224** | **0.3187** | **🏆 Melhor modelo.** FN=41, FP=58. Pico do projeto. |
| **11** Re-execução livre | 500→50 | 75.23% | 80.61% | 0.90 | 0.4780 | Alta variância estocástica. Recall=95.4%. 18 FN. |

---

## Resultado Final — O Modelo "Hipocondríaco"

O **Teste 10** produziu o melhor modelo do projeto, com Early Stopping ativando na época 96:

```
Matriz de Confusão — Teste 10 (conjunto de teste, 624 imagens)

                    PREDITO
               NORMAL    PNEUMONIA
REAL  NORMAL     176    │   58     ← 58 FP (saudáveis re-examinados)
      PNEUMONIA   41    │  349     ← 41 FN (doentes não detectados)
```

Ao longo dos experimentos, especialmente nos Testes 07 e 08, o modelo desenvolveu um comportamento que pode ser chamado de **"hipocondríaco"**: com `sample_weight` penalizando fortemente os erros na classe PNEUMONIA, ele aprendeu que **errar o diagnóstico de uma Pneumonia custa muito mais caro do que disparar um alarme falso**.

### O que isso significa na prática?

Na menor dúvida, o modelo prefere classificar como **PNEUMONIA**. Em números, o perfil mais sensível (Testes 07/08) atingiu apenas **9–11 Falsos Negativos** à custa de **156 Falsos Positivos**.

Em um cenário de **triagem médica real**, esse é exatamente o comportamento desejado:

> É muito melhor encaminhar **156 pessoas saudáveis para exames adicionais** do que mandar **11 pessoas com pneumonia para casa sem tratamento.**

Um Falso Negativo em diagnóstico de pneumonia pode resultar em progressão da doença, internação tardia ou óbito. Um Falso Positivo resulta em um exame adicional desnecessário — inconveniente, mas clinicamente seguro.

O modelo, sem ter sido explicitamente programado para isso, internalizou essa lógica médica através do balanceamento de classes por `sample_weight`. No Teste 10, o ajuste (×0.65) trouxe mais equilíbrio: **FN=41 e FP=58** — mantendo boa sensibilidade clínica com muito menos alarmes falsos.

---

## Validação em Imagens Externas

Ao final dos experimentos, o modelo foi testado em **7 radiografias de fontes externas** (Radiopaedia e Wikipedia), fora do dataset de treinamento:

| # | Fonte | Classe Real | Predito | Prob. Pneumonia | |
|---|---|---|---|---|---|
| 1 | Radiopaedia | NORMAL | NORMAL | 0.4082 | ✅ |
| 2 | Radiopaedia | PNEUMONIA | PNEUMONIA | 0.8041 | ✅ |
| 3 | Wikipedia | NORMAL | PNEUMONIA | 0.7014 | ❌ |
| 4 | Wikipedia | PNEUMONIA | NORMAL | 0.3694 | ❌ |
| 5 | Wikipedia | PNEUMONIA | PNEUMONIA | 0.7123 | ✅ |
| 6 | Wikipedia | NORMAL | NORMAL | 0.4331 | ✅ |
| 7 | Wikipedia | PNEUMONIA | NORMAL | 0.4623 | ❌ |

**Acurácia externa: 4/7 (57%)** — queda esperada por *distribution shift*: imagens externas possuem protocolos de captura, contraste e resolução originais muito diferentes do dataset de Kermany, e toda informação de alta frequência é perdida ao redimensionar para `32×32`.

> **Nota técnica:** As URLs da Wikimedia podem retornar **HTTP 429 (Too Many Requests)** em execuções consecutivas. O script inclui `time.sleep(10)` entre requisições para respeitar o *rate limit* dos servidores.

---

## Trabalhos Futuros

### 1. Aumentar a resolução de entrada

Testar entradas de **`224×224`** ou **`512×512`** em vez de `32×32`. A resolução atual descarta grande parte do detalhe clínico relevante — infiltrados e consolidações pulmonares sutis são praticamente invisíveis a 32 pixels. Maior resolução exige mais memória de GPU, mas tende a melhorar significativamente a capacidade diagnóstica.

### 2. Trocar o otimizador

Substituir o **SGD com momentum** por **Adam** ou **RMSprop**. O SGD exige ajuste cuidadoso da taxa de aprendizado; Adam e RMSprop adaptam o learning rate automaticamente por parâmetro, geralmente convergindo mais rápido e de forma mais estável em redes profundas.

### 3. Travar a semente aleatória (seed)

O Teste 11 demonstrou que a ausência de seed fixa causa variância expressiva entre execuções (acurácia variando em até 4%). Fixar a seed garante **reprodutibilidade total** dos experimentos:

```python
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

### 4. Experimento sistemático de `kernel_size` × `stride`

O objetivo central do trabalho é medir o impacto isolado desses dois hiperparâmetros na 1ª camada convolucional. A tabela abaixo define os 4 experimentos planejados, mantendo **todos os outros parâmetros fixos** (mesmas épocas, mesmo Early Stopping, mesmo balanceamento do Teste 10):

| Experimento | `kernel_size` | `stride` | Val Accuracy | AUC |
|---|---|---|---|---|
| A | `(3, 3)` | `(1, 1)` | ? | ? |
| B | `(5, 5)` | `(1, 1)` | ? | ? |
| C | `(3, 3)` | `(2, 2)` | ? | ? |
| D | `(5, 5)` | `(2, 2)` | ? | ? |

Os resultados desta tabela permitirão uma conclusão científica sobre o efeito isolado de cada variável — que é exatamente o argumento central esperado em um trabalho acadêmico de CNN.

---

## Como Reproduzir

### 1. Configurar o Kaggle

- Acesse [kaggle.com](https://www.kaggle.com) e crie um novo notebook
- Em **Settings → Accelerator**, selecione `GPU T4 x2`
- Em **Data → Add Data**, busque e adicione: `chest-xray-pneumonia` (de Paul Mooney)
- Certifique-se de que **Internet está habilitada** nas configurações

### 2. Clonar o repositório

```python
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
token   = secrets.get_secret("GITHUB_TOKEN")
user    = "SEU_USUARIO"
repo    = "SEU_REPOSITORIO"

os.system(f'git config --global user.name "{user}"')
os.system(f'git clone https://{user}:{token}@github.com/{user}/{repo}.git /kaggle/working/repo')
```

### 3. Executar

Rode todas as células em sequência (`Run All`). O treinamento do Teste 10 levou ~96 épocas com Early Stopping, em aproximadamente 15–20 minutos na GPU T4.

### 4. Fazer commit dos resultados

```python
import subprocess
from datetime import datetime

msg = f"exp: kernel=3x3, stride=1x1, acc=0.8413 [{datetime.now():%Y-%m-%d %H:%M}]"
cmds = [
    "cd /kaggle/working/repo && git add .",
    f'cd /kaggle/working/repo && git commit -m "{msg}"',
    "cd /kaggle/working/repo && git push origin main"
]
for cmd in cmds:
    subprocess.run(cmd, shell=True)
```

---

## Estrutura do Repositório

```
📁 /
├── cnn-dl-caio.ipynb                    # Notebook principal com os 11 experimentos
├── README.md                            # Este arquivo
└── inception_v1_binary_chestxray.h5     # Modelo salvo (Teste 10)
```

---

## Dependências

Todas já disponíveis no ambiente padrão do Kaggle (Python 3.12):

| Biblioteca | Uso |
|---|---|
| `tensorflow` / `keras` | Framework principal |
| `opencv-python` (`cv2`) | Leitura e redimensionamento de imagens |
| `numpy` | Manipulação de arrays |
| `matplotlib` | Visualização de métricas e imagens |
| `seaborn` | Matriz de confusão estilizada |
| `scikit-learn` | `compute_class_weight` e métricas auxiliares |
| `Pillow` (`PIL`) | Carregamento de imagens via URL |
| `requests` | Requisições HTTP para URLs externas |

---

## Referência do Dataset

> Kermany, D., Zhang, K., & Goldbaum, M. (2018). *Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification*. Mendeley Data. [https://doi.org/10.17632/rscbjbr9sj.3](https://doi.org/10.17632/rscbjbr9sj.3)
