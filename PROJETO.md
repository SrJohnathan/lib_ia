# PROJETO — lib_ia, lib-rust e drill

Este documento descreve os **três níveis** do repositório:

1. **A lib** (`src/`, C++17 + ABI C) — o motor de IA local (execução de LLMs GGUF via llama.cpp, imagens, voz e outros módulos).
2. **`lib-rust`** (`client-rust/lib-rust/`) — bindings seguros de Rust sobre a ABI C + o *agent framework* que orquestra turnos, fila e ferramentas.
3. **`drill`** (`client-rust/drill/`) — cliente TUI em Rust (agente de código no terminal) que **consome** a lib através do `lib-rust`.

> **Nota importante:** a lib **não foi criada para o drill**. Ela é um componente independente
> e reutilizável (bliblioteca compartilhada, com ABI C estável). O drill é apenas um dos
> clientes que a consomem, via `lib-rust`. A lib também tem exemplos próprios em Rust
> (`client-rust/inference`) que não passam pelo drill.

---

## 1. A lib (`lib_ia`)

Camada nativa em **C++** com **header de integração C** (`src/integration.h`), compilada com
CMake como `liblib_ia.so` (e variantes de build `debug`/`release`).

Módulos principais:

| Fonte        | Responsabilidade |
|--------------|------------------|
| `runtime.cpp` | Um `Runtime` que encapsula a execução de um modelo GGUF (llama.cpp): contexto, geração, multimodal (mmproj), histórico de mídia (imagens/áudio/vídeo), cancelamento e propriedades (flash-attn, cache KV, threads, razão crítica). |
| `integration.cpp/.h` | A **fronteira C estável** — todas as funções `libia_*` que o Rust chama. |
| `library.cpp` | Carga/descarga dos modelos e controle do runtime. |
| `image-create.cpp` | Geração de imagem (schedulers, difusão) — vetor `LIBIA_IMAGE_SCHEDULER_*`. |
| `sst.cpp` | Speech-to-text (transcrição). |

A lib não conhece o drill nem o Rust: expõe somente a ABI C.

Build:

```sh
cmake -B build-release -DCMAKE_BUILD_TYPE=Release   # (vid: crie na pasta build-release)
cmake --build build-release -j
```

## 2. `lib-rust` (bindings + agente)

Workspace `client-rust` (crates: `lib-rust`, `drill`, `inference`, `sst-inference`).

`lib-rust/src`:

| Fonte             | Conteúdo |
|-------------------|----------|
| `ffi.rs`          | Declaração do FFI: enums de scheduler/geração/erros e funções `libia_*`. |
| `runtime/mod.rs`  | `RuntimeLlama` — wrapper seguro: `try_new`, geração, `add_media_file`/`base64` (imagens, áudio, vídeo), `supports_image/audio/video`, propriedades, cancelamento global. |
| `agents/`         | **AgentOrchestrator** (turnos solo + subagentes, histórico, sessão, persona), **AgentManager** (fila serial de execução, troca de engine a quente via `install_engine`, turnos remotos `run_remote_turn`) e **MAESTRO_AGENT_ID** (`drill`). |
| `tools/`          | Catálogo de ferramentas (`TOOLS_JSON`: read_file, write_file, list_dir, run_command, fetch_url, …). |
| `sst.rs` / `image.rs` / `traits.rs` | Voz, imagem e contratos compartilhados. |

O **IR (intermediate representation)** entre o drill e a lib é agnóstico de vendor: mensagens
`ChatMessage { role, content, reasoning_content }` no formato OpenAI, e closures
`Fn(Vec<Value>, &mut dyn FnMut(RemoteEvent))` para o provider remoto.

### Importante sobre a compilação

`lib-rust` carrega a lib nativa em tempo de execução; para **compilar** é preciso
apontar onde está a lib:

```sh
LIBIA_BUILD_DIR=/caminho/para/lib_ia/build-release cargo build -p drill
```

## 3. `drill` (cliente TUI)

Agente de código rodando no terminal (Rust, `crossterm` + `ratatui`). Características
centrais:

- **Providers** (`/provider`): lista *local* (GGUF via `RuntimeLlama`) e *cloud*
  (`openai` / `anthropic` / `google`, com base URL, modelo, API key e router). A escolha
  é persistida no `config.json`.
  - Wizard **`a`** no overlay adiciona um provider cloud novo (tipo → base URL →
    modelo → API key) e o deixa ativo sem reiniciar.
  - Cloud: `build_generator` traduz o IR para o wire de cada vendor
    (chat/responses, Messages API, Generative Language) e faz streaming + tool calls.
- **Modelo local** (`/model` + modal no boot): se o GGUF está ausente, a TUI sobrevive
  e pede o caminho (e o `mmproj` opcional) em um modal. **Hot-swap**: `install_engine`
  troca o runtime a quente (nação sem reiniciar).
- **Multimodal local**: se `mmproj` está configurado, o engine suporta imagens; o
  toggle do modal liga/desliga no boot.
- **Menção de arquivos** (`@`): popup de arquivos do diretório de trabalho
  (`scan_project_files`). O que acontece com cada menção:
  - **não-imagem** (ou arquivo ausente): mantém o token literal (fluxo original);
  - **imagem** (`png` / `jpg` / `jpeg`) **+ multimodal ativo**: a imagem é de fato
    anexada — em local via `add_media_file`; em cloud via partes `image_url` (data URI
    base64) traduzidas para `input_image`/`image`/`inlineData` conforme o vendor;
  - **imagem + multimodal desativado**: mantém o token e **injeta um aviso no prompt**
    (`[AVISO] ... imagem não processada`) e mostra um aviso na UI.
- **Modo maestro** (`Ctrl+A` / `--mode agents`): o drill assume `MAESTRO_AGENT_ID`
  `drill`, carrega subagentes (`drill.<nome>.agent.json`) e os executa **sempre no
  runtime local**, mesmo com provider cloud.
- **Histórico**: persistência por sessão em LanceDB (`drill-<sessão>`), listas `/session`,
  retomada com `resume`.
- **Permissões**: ferramentas sensíveis bloqueiam a geração até aprovação do usuário.

Arquivos relevantes em `client-rust/drill/src`:

| Arquivo        | Conteúdo |
|----------------|----------|
| `main.rs`      | CLI, build do engine, worker de mensagens, headless, `EngineParams`. |
| `agent.rs`     | `Agent` (frente do drill): turnos local/cloud, `prepare_turn` (menções `@` + imagens), procedures de setup. |
| `provider.rs`  | `ProviderStore` (swap/`add_cloud`/persistência), geradores de cada vendor e tradução do IR. |
| `config.rs`    | `DrillConfig`, `LocalProvider`/`CloudProvider`, defaults. |
| `tui.rs`       | TUI: overlays `/provider`, `/models`, `/model`, modais, popup de menção. |
| `history.rs` / `tools.rs` / `subagents.rs` | Persistência, permissão de tools e carregamento de subagentes. |

### Config

- `config.json` padrão em `~/.drill/config.json` (autocriado), sobrescrito por CLI
  (`--model`, `--mmproj`, `--ngl`, `--ctx`, `--dir`, `--persona`, `--mode`, `--headless`, …).
- `DRILL_CONFIG=<path>` troca o arquivo de config; `HOME` afeta o diretório `~/.drill`.
- **Nunca versionar chaves de API** do provider cloud (`api_key`); usar env vars do vendor
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`) quando preferível.

### Testes

```sh
LIBIA_BUILD_DIR=/.../build-release cargo check -p drill
LD_LIBRARY_PATH=/.../build-release cargo test -p drill   # cobre agent + provider
LD_LIBRARY_PATH=/.../build-release cargo test -p lib-rust
# E2E cloud: subir um mock HTTP e rodar a TUI com provider remoto (PTY).
```

---

Ver também: `README.md` (raiz) — doc original *em inglês* focada na lib nativa.