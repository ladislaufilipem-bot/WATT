# Framework de Treinamento PPO com Webots

Um framework modular, eficiente e organizado para treinar agentes PPO (Proximal Policy Optimization) usando simulação física Webots. Cada agente é isolado em seu próprio diretório com bibliotecas compartilhadas para máximo reaproveitamento de código e manutenibilidade.

## Características

- **Arquitetura Modular**: Cada agente em seu próprio diretório (`agents/agent_XXX/`)
- **Config-Driven**: Todos hiperparâmetros em YAML com herança de configuração
- **Integração Webots**: Wrapper nativo de ambiente Webots com comunicação via Supervisor
- **Implementação PPO**: PPO limpo e eficiente com GAE, loss clippado, bônus de entropia
- **Redes Neurais**: Arquiteturas flexíveis (MLP, CNN, RNN, Transformer, encoders LiDAR)
- **Treino Multi-Agente**: Orquestrador para treino paralelo com curriculum learning
- **Reprodutibilidade**: Seeding completo, checkpointing, versionamento de configs
- **Observabilidade**: TensorBoard, Weights & Biases, logs estruturados JSONL

## Início Rápido

### Pré-requisitos

1. **Python 3.10+**
2. **Webots R2023b+** - Baixe de [cyberbotics.com](https://cyberbotics.com/)
3. **CUDA** (opcional, para treino em GPU)

```bash
# Defina a variável de ambiente Webots
# Windows PowerShell:
$env:WEBOTS_HOME = "C:\Program Files\Webots"

# Linux/macOS:
export WEBOTS_HOME=/usr/local/webots
```

### Instalação

```bash
cd ia

# Dependências principais
pip install -r requirements.txt

# Dependências de desenvolvimento (testes, linting, visualização)
pip install -r requirements-dev.txt

# Instalar hooks pre-commit
pre-commit install
```

### Crie Seu Primeiro Agente

```bash
# Listar templates disponíveis
python scripts/create_agent.py --list

# Criar agente walker
python scripts/create_agent.py --name agent_001 --template walker --world walker-training.wbt

# Ou agente customizado
python scripts/create_agent.py --name agent_002 --template arm --description "Manipulação braço 6-DOF"
```

### Treine um Agente

```bash
# Navegue para o diretório do agente
cd agents/agent_001

# Teste rápido (1 episódio)
python train.py --dry-run

# Treino completo
python train.py

# Retomar de checkpoint
python train.py --resume checkpoints/latest.pt

# Com Weights & Biases
python train.py --wandb --wandb-project meu-projeto
```

### Avalie um Agente

```bash
# Avaliar melhor checkpoint
python scripts/evaluate.py --agents agent_001 --episodes 50

# Comparar múltiplos agentes
python scripts/evaluate.py --agents agent_001,agent_002,agent_003 --compare --output comparacao.json
```

### Visualize Treinamento

```bash
# Plotar agente único
python scripts/visualize.py --agents agent_001 --metric episode_reward

# Comparar agentes
python scripts/visualize.py --compare agent_001,agent_002 --metric episode_reward --output comparacao.png

# Relatório HTML interativo
python scripts/visualize.py --compare agent_001,agent_002 --format html
```

## Estrutura do Projeto

```
ia/
├── CLAUDE.md              # Guia do projeto para Claude Code
├── init/                  # OBRIGATÓRIO: Leia rules.md primeiro!
│   ├── rules.md           # Regras: eficiência, organização, modularidade
│   ├── structure.md       # Padrões de estrutura de diretórios
│   └── workflow.md        # Procedimentos de treinamento
├── config/                # Configurações globais
│   ├── ppo_config.yaml    # Padrões algoritmo PPO
│   ├── neural_config.yaml # Arquiteturas redes neurais
│   ├── webots_config.yaml # Configuração simulação Webots
│   └── training_config.yaml # Pipeline treinamento
├── core/                  # Bibliotecas compartilhadas (SEM imports de agents)
│   ├── ppo/               # Algoritmo PPO (agent, buffer, loss, update)
│   ├── neural/            # Redes (actor, critic, encoders)
│   ├── env/               # Wrappers ambiente (Gym, Webots)
│   └── utils/             # Utilitários (config, seeding, logging, checkpoint, metrics)
├── agents/                # Agentes individuais (isolados)
│   ├── template/          # Template para novos agentes
│   ├── agent_001/         # Exemplo: walker
│   └── agent_002/         # Exemplo: arm
├── webots/                # Assets Webots
│   ├── worlds/            # Arquivos .wbt
│   ├── controllers/       # Controladores robô (Python)
│   ├── proto/             # Definições PROTO
│   └── assets/            # Malhas, texturas, sons
├── training/              # Orquestração
│   ├── orchestrator.py    # Coordenador multi-agente
│   ├── evaluator.py       # Utilitários avaliação
│   └── curriculum.py      # Curriculum learning
├── scripts/               # CLI utilities
│   ├── create_agent.py    # Criar agente do template
│   ├── run_training.py    # Pipeline treinamento
│   ├── evaluate.py        # Avaliar agentes
│   └── visualize.py       # Plotar logs, comparar
├── tests/                 # Testes unitários e integração
├── docs/                  # Documentação
├── pyproject.toml         # Config projeto
├── requirements.txt       # Dependências principais
├── requirements-dev.txt   # Dependências dev
└── README.md              # Este arquivo
```

## Estrutura do Agente

Cada agente é completamente auto-contido:

```
agents/agent_XXX/
├── config.yaml          # Config completa (herda global + overrides)
├── train.py             # Entry point treinamento
├── eval.py              # Entry point avaliação
├── model/
│   └── __init__.py      # Definições modelo customizado (opcional)
├── logs/                # Logs treinamento (JSONL, TensorBoard)
├── checkpoints/         # Checkpoints modelo
│   ├── best.pt          # Melhor performance avaliação
│   ├── latest.pt        # Checkpoint mais recente
│   └── step_XXXXXX.pt   # Checkpoints periódicos
└── videos/              # Vídeos avaliação (opcional)
```

## Configuração

### Herança de Configuração

```
config/ppo_config.yaml       # Padrões PPO base
    ↓
config/neural_config.yaml    # Padrões rede base
    ↓
config/webots_config.yaml    # Padrões Webots base
    ↓
config/training_config.yaml  # Padrões treino base
    ↓
agents/agent_XXX/config.yaml # Overrides ESPECÍFICOS do agente
    ↓
CLI args (--key value)       # Overrides runtime
```

### Exemplo Config Agente

```yaml
# agents/agent_001/config.yaml
version: "1.0.0"

agent:
  name: "agent_001"
  description: "Bipedal walker em terreno plano"

ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01

neural:
  actor:
    hidden_sizes: [256, 256]
    activation: "tanh"
  critic:
    hidden_sizes: [256, 256]
    activation: "tanh"

env:
  world: "training.wbt"
  observation_keys: [joint_pos, joint_vel, imu, lidar]
  action_keys: [motor_pos]
  reward_weights:
    forward_vel: 1.0
    energy: -0.01
    alive: 0.1

training:
  total_timesteps: 5_000_000
  eval_freq: 50_000
  n_eval_episodes: 10
```

## Integração Webots

### Setup do Mundo

1. Crie mundo no Webots em `webots/worlds/seu_mundo.wbt`
2. Adicione robô com nome DEF "PPO_ROBOT"
3. Adicione sensores: Lidar, InertialUnit, Gyro, Accelerometer, GPS, PositionSensors
4. Adicione atuadores: Motors com controle de posição
5. Adicione Supervisor para controle ambiente
6. Defina controller do robô como "ppo_agent"
7. Defina controller supervisor como "supervisor"

### Robô Customizado

Defina seu robô em `webots/proto/robots/SeuRobo.proto`:

```proto
PROTO SeuRobo [ ... ] {
  Robot {
    name "PPO_ROBOT"
    controller "ppo_agent"
    children [ ... sensores e atuadores ... ]
  }
}
```

## Fluxo de Trabalho

### 1. Setup Ambiente (Uma vez)
```bash
# Verificar integração Webots
webots --mode=fast --no-rendering webots/worlds/training.wbt
python -m core.utils.validate_config config/ppo_config.yaml
```

### 2. Criar Agente
```bash
python scripts/create_agent.py --name agent_003 --template walker --world walker-rough.wbt
```

### 3. Personalizar Função Recompensa
Edite `agents/agent_003/train.py`:
```python
def custom_reward_fn(obs, action, next_obs, info):
    reward = 0.0
    # Sua lógica recompensa aqui
    return reward, breakdown
```

### 4. Dry Run
```bash
python agents/agent_003/train.py --dry-run --verbose
```

### 5. Treinar
```bash
python agents/agent_003/train.py
# Monitorar: tensorboard --logdir agents/agent_003/logs/tensorboard
```

### 6. Avaliar
```bash
python scripts/evaluate.py --agent agent_003 --episodes 100 --output eval_resultados.json
```

## Orquestração Multi-Agente

```bash
# Treinar múltiplos agentes com curriculum
python training/orchestrator.py \
    --config config/training_config.yaml \
    --agents agent_001,agent_002,agent_003 \
    --parallel 2
```

Configuração em `config/training_config.yaml`:
```yaml
orchestration:
  agents: ["agent_001", "agent_002"]
  max_parallel_agents: 2
  population:
    enabled: true
  curriculum:
    enabled: true
    stages: [...]
```

## Exportação Modelo

```bash
# Exportar para ONNX deploy
python scripts/export_onnx.py --agent agent_001 --checkpoint best.pt --output modelo.onnx
```

## Testes

```bash
# Todos testes
pytest tests/ -v

# Apenas unitários
pytest tests/unit/ -v

# Com coverage
pytest tests/ --cov=core --cov-report=html
```

## Regras de Desenvolvimento

**LEIA `init/rules.md` ANTES DE FAZER ALTERAÇÕES**

Princípios-chave:
1. **Eficiência**: Reutilize `core/`, configs > valores hardcoded
2. **Organização**: 1 agente = 1 pasta, código compartilhado SÓ em `core/`
3. **Modularidade**: Responsabilidade única por módulo
4. **Reprodutibilidade**: Configs versionadas, seeding, checkpointing
5. **Observabilidade**: Logs estruturados, métricas, visualização

## Problemas Comuns

| Problema | Solução |
|----------|---------|
| Webots não encontrado | Defina `WEBOTS_HOME`, verifique executável |
| Erros import | Rode da raiz do projeto, PYTHONPATH correto |
| CUDA OOM | Reduza `batch_size`, `n_steps`, enable gradient checkpointing |
| Treino diverge | Reduza learning_rate 10x, normalize reward/obs |
| Webots crash | Verifique `controller.log`, reduza timestep physics |

## Licença

MIT License - veja arquivo LICENSE para detalhes.

## Citação

Se usar este framework em pesquisa, cite:
```bibtex
@software{ppo_webots,
  title = {PPO Neural Network Training Framework with Webots},
  year = {2024}
}
```