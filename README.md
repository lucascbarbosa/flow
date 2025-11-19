# Mini-Projeto 1: Neural ODEs e Continuous Normalizing Flows

**Prazo:** 30/11/2025  
**Trabalho:** Individual ou duplas

## Objetivos de Aprendizagem

Ao completar este projeto, os alunos e alunas devem ser capazes de:

1. Implementar Neural ODEs usando torchdiffeq
2. Compreender continuous normalizing flows e change of variables
3. Aplicar trace estimation techniques (Hutchinson)
4. Analisar trade-offs computacionais de CNFs vs discrete flows
5. Preparar-se conceitualmente para Flow Matching (Módulo 2)

## Stack Tecnológico

- **torchdiffeq**: https://github.com/rtqichen/torchdiffeq
- **FFJORD repo (referência)**: https://github.com/rtqichen/ffjord
  - Apenas para consulta, não usar diretamente
  - Entender a implementação, mas escrever seu próprio código

## Estrutura do Projeto

```
mini-project-1/
├── README.md
├── requirements.txt
├── src/
│   ├── models/
│   │   ├── vector_field.py      # VectorField architectures
│   │   ├── neural_ode.py        # Neural ODE básico
│   │   ├── cnf.py               # CNF com trace exato
│   │   └── ffjord.py            # FFJORD com Hutchinson
│   ├── utils/
│   │   ├── datasets.py          # Data loading
│   │   ├── training.py          # Training loops
│   │   ├── trace.py             # Trace computation utilities
│   │   └── visualization.py     # Plotting
│   └── experiments/
│       ├── exp1_ode_solvers.py
│       ├── exp2_regularization.py
│       └── exp3_architectures.py
├── notebooks/
│   ├── 01_neural_ode_2d.ipynb
│   ├── 02_cnf_trace_comparison.ipynb
│   └── 03_ffjord_mnist.ipynb
└── results/
    ├── figures/
    └── checkpoints/
```

## Instalação

```bash
pip install -r requirements.txt
```

## Milestones

### Milestone 1: Neural ODE Básico (Semana 1)
- Implementar VectorField e NeuralODE
- Treinar em dataset 2D sintético
- Visualizar trajetórias e vector fields
- Analisar NFEs e comparar solvers

### Milestone 2: CNF com Trace Exato (Semana 1-2)
- Implementar divergence_exact e CNF
- Treinar em dados 2D e MNIST reduzido
- Comparar com Real NVP baseline
- Analisar escalabilidade

### Milestone 3: FFJORD com Hutchinson Estimator (Semana 2)
- Implementar Hutchinson trace estimator
- Escalar CNF para alta dimensão
- Treinar em MNIST completo

## Uso

Ver notebooks individuais para exemplos de uso de cada componente.

