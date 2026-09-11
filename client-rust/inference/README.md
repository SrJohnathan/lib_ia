# inference

Binário de teste do `lib-rust` para validar o runtime `llama.cpp` via Rust.

Ele foi feito para:

- inferência de texto
- inferência com raciocínio separado da resposta
- testes com multimodalidade usando `mmproj`
- testes com áudio quando o modelo suporta essa modalidade
- testes com vídeo quando o build/helper suporta essa modalidade
- testes com speculative decoding e MTP
- streaming no terminal com cor diferente para `Reasoning`

## Como rodar

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --prompt "O que é Rust language?" \
  --reasoning-format auto \
  --reasoning on
```

Se você quiser deixar a saída mais curta ou mais longa, ajuste `--n-predict`:

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --prompt "Explique ownership em Rust" \
  --reasoning-format auto \
  --reasoning on \
  --n-predict 1024
```

## Exemplo de inferência normal

Use quando quiser só texto, sem imagem:

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --prompt "Explique o que é uma trait em Rust" \
  --reasoning off \
  --n-predict 512
```

Com `--reasoning off`, o runtime tenta responder direto, sem imprimir bloco de pensamento.

## Exemplo com raciocínio

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --prompt "O que é Rust language?" \
  --reasoning-format auto \
  --reasoning on \
  --n-predict -1
```

Neste modo:

- `Reasoning` sai em ciano no terminal
- `Content` sai normal
- o fluxo tenta respeitar o template do modelo

## Exemplo com MTP

Quando o modelo principal tiver um drafter auxiliar `mtp-*.gguf`, ative `draft-mtp` e passe o arquivo auxiliar explicitamente:

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --spec-type draft-mtp \
  --spec-draft-model /home/johnathan/models/mtp-gemma-4-E4B-it.gguf \
  --spec-draft-n-max 8 \
  --spec-draft-n-min 0 \
  --spec-draft-p-split 0.10 \
  --spec-draft-p-min 0.00 \
  --reasoning-format auto \
  --reasoning on \
  --prompt "Explique ownership em Rust"
```

Parâmetros úteis:

- `--spec-type`: escolhe o tipo de speculative decoding
- `--spec-draft-model`: caminho do drafter auxiliar
- `--spec-draft-n-max`: limite máximo de tokens do draft
- `--spec-draft-n-min`: mínimo de tokens do draft
- `--spec-draft-p-split`: probabilidade de split do draft
- `--spec-draft-p-min`: probabilidade mínima para parar cedo
- `--spec-draft-backend-sampling`: liga ou desliga sampling no backend
- `--spec-draft-ngl`: camadas do drafter que vão para GPU/Vulkan

## Exemplo com imagens

O modelo multimodal precisa de dois arquivos:

- o modelo principal `.gguf`
- o projector `mmproj-*.gguf`

Exemplo de configuração esperada:

```text
/home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf
/home/johnathan/models/mmproj-F16.gguf
```

Fluxo recomendado:

1. passar o `mmproj` no `RuntimeLlama::new(...)`
2. carregar uma ou mais mídias com `add_media_file(...)` ou `add_media_files(...)`
3. perguntar algo sobre a mídia

Exemplo de uso da lib, em nível baixo:

```rust
let mut engine = RuntimeLlama::new(
    "/home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf".to_string(),
    4096,
    4096,
    512,
    128,
    1.0,
    0.95,
    64,
    Some("/home/johnathan/models/mmproj-F16.gguf".to_string()),
    None,
    None,
    Some("Voce e um assistente tecnico.".to_string()),
);

engine.load().unwrap();
engine.add_media_file("/home/johnathan/Imagens/Capturas de tela/Captura de tela de 2026-06-16 22-18-33.png").unwrap();
engine.charge_prop("prompt".to_string(), "O que tem nesta imagem?".to_string());
```

No binário `inference`, você também pode passar mídia direto por flags:

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --mmproj /home/johnathan/models/mmproj-F16.gguf \
  --image /home/johnathan/Imagens/Capturas\ de\ tela/Captura\ de\ tela\ de\ 2026-06-16\ 22-18-33.png \
  --prompt "O que tem nesta imagem?"
```

Vários arquivos, separados por vírgula:

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --mmproj /home/johnathan/models/mmproj-F16.gguf \
  --audio /home/johnathan/Audio/parte1.wav,/home/johnathan/Audio/parte2.mp3 \
  --prompt "Compare estes áudios"
```

Base64 também aceita lista separada por vírgula:

```bash
cargo run -p inference -- \
  --model /home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf \
  --mmproj /home/johnathan/models/mmproj-F16.gguf \
  --audio-b64 "data:audio/wav;base64,AAAA...,data:audio/mp3;base64,BBBB..." \
  --prompt "Transcreva estes áudios"
```

Múltiplas imagens por caminho:

```rust
let images = vec![
    "/home/johnathan/Imagens/img1.png".to_string(),
    "/home/johnathan/Imagens/img2.png".to_string(),
];

engine.add_media_files(&images).unwrap();
engine.charge_prop("prompt".to_string(), "Compare estas imagens".to_string());
```

Base64 de clipboard ou payload bruto:

```rust
let image_b64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...";
engine.add_media_base64(image_b64).unwrap();
engine.charge_prop("prompt".to_string(), "O que há nesta imagem?".to_string());
```

Várias imagens em base64:

```rust
let images_b64 = vec![
    "data:image/png;base64,AAAA...".to_string(),
    "data:image/jpeg;base64,BBBB...".to_string(),
];

engine.add_media_base64s(&images_b64).unwrap();
engine.charge_prop("prompt".to_string(), "Analise todas as imagens".to_string());
```

Áudio funciona pelo mesmo pipeline multimodal:

```rust
if engine.supports_audio().unwrap_or(false) {
    engine.add_audio_file("/home/johnathan/Audio/exemplo.wav").unwrap();
    engine.charge_prop("prompt".to_string(), "Transcreva e resuma este áudio".to_string());
}
```

Vários áudios:

```rust
let audios = vec![
    "/home/johnathan/Audio/parte1.wav".to_string(),
    "/home/johnathan/Audio/parte2.mp3".to_string(),
];

engine.add_audio_files(&audios).unwrap();
engine.charge_prop("prompt".to_string(), "Compare estes áudios".to_string());
```

Áudio em base64:

```rust
let audio_b64 = "data:audio/wav;base64,UklGRiQAAABXQVZFZm10...";
engine.add_audio_base64(audio_b64).unwrap();
engine.charge_prop("prompt".to_string(), "Transcreva este áudio".to_string());
```

Vários áudios em base64:

```rust
let audios_b64 = vec![
    "data:audio/wav;base64,AAAA...".to_string(),
    "data:audio/mp3;base64,BBBB...".to_string(),
];

engine.add_audio_base64s(&audios_b64).unwrap();
engine.charge_prop("prompt".to_string(), "Analise todos os áudios".to_string());
```

Vídeo segue o mesmo padrão:

```rust
if engine.supports_video().unwrap_or(false) {
    engine.add_video_file("/home/johnathan/Videos/exemplo.mp4").unwrap();
    engine.charge_prop("prompt".to_string(), "Descreva este vídeo".to_string());
}
```

Também é possível usar `--video` e `--video-b64` no binário.

Se você quiser usar o fluxo com imagem em uma prova rápida, a sequência é:

```text
1. carregar o modelo
2. carregar o mmproj
3. anexar a imagem com `add_media_file(...)`
4. gerar a resposta
```

### Base64

Hoje o wrapper Rust expõe anexar arquivo local por caminho.

Se você quiser base64, pode usar `add_media_base64(...)`, `add_media_base64s(...)`, `add_audio_base64(...)`, `add_audio_base64s(...)`, `add_video_base64(...)` ou `add_video_base64s(...)`.

Isso funciona bem para fluxo de clipboard, quando o sistema já entrega a imagem como `data:image/...;base64,...` ou como string base64 pura.

Se você estiver usando MTP, o binário também aceita esses parâmetros por flags. O runtime Rust apenas repassa para `llama.cpp`, então o comportamento deve ser o mesmo do CLI nativo.

## Props usadas por este binário

Essas são as props mais importantes que o `inference` usa hoje.

| Prop | O que faz | Valor atual |
| --- | --- | --- |
| `--model` | Caminho do `.gguf` principal | `/home/johnathan/models/gemma-4-E4B-it-Q6_K.gguf` |
| `--prompt` | Texto de entrada do usuário | `O que é Rust language?` |
| `--reasoning-format` | Define como o raciocínio é tratado/extraído | `auto` |
| `--reasoning` | Liga, desliga ou deixa automático | `on` |
| `--n-predict` | Quantidade máxima de tokens gerados; `-1` = sem limite | `-1` |
| `--fit` | Ajusta ou não parâmetros para caber em memória de GPU | `off` |
| `--reasoning-budget` | Orçamento de tokens de pensamento; `-1` = sem limite | `-1` |
| `--reasoning-budget-message` | Mensagem injetada quando o orçamento de pensamento termina | texto curto em português |
| `--spec-type` | Tipo de speculative decoding | `draft-mtp` |
| `--spec-draft-model` | Caminho do drafter auxiliar MTP | tenta auto-descobrir `mtp-*.gguf` |
| `--spec-draft-n-max` | Máximo de tokens para draft | `4` |
| `--spec-draft-n-min` | Mínimo de tokens para draft | `0` |
| `--spec-draft-p-split` | Probabilidade de split | `0.10` |
| `--spec-draft-p-min` | Probabilidade mínima do draft | `0.00` |
| `--spec-draft-backend-sampling` | Sampling do backend para o drafter | `on` |
| `--spec-draft-ngl` | Número de camadas do drafter em GPU/Vulkan | `-1` |
| `--jinja` | Liga o engine de templates Jinja | `on` via código |
| `--image-min-tokens` | Menor orçamento para uma imagem | vem do modelo / runtime |
| `--image-max-tokens` | Maior orçamento para uma imagem | vem do modelo / runtime |

## Defaults internos do teste

O binário já nasce com estes valores:

- `ctx = 4096`
- `n_predict = 4096` no construtor, depois pode ser sobrescrito por `--n-predict`
- `n_batch = 512`
- `n_ubatch = 128`
- `temp = 1.0`
- `top_p = 0.95`
- `top_k = 64`
- `system_prompt = "Voce e um assistente tecnico. Responda em portugues, com clareza, objetividade e completude."`

## Saída no terminal

O stream imprime assim:

- `Reasoning` em ciano
- `Content` na cor normal

Isso ajuda a ver se o modelo está:

- pensando demais
- entrando em resposta final
- truncando cedo

## Dicas de manutenção

- Se a resposta vier só em pensamento, o problema costuma ser `reasoning_budget`, `reasoning_format` ou o template do modelo.
- Se o modelo não subir, confirme se o `mmproj` está correto e se a GPU/Vulkan está conseguindo alocar memória.
- Para ver todas as props suportadas pela lib, consulte `RuntimeLlama::get_all_props()` no `lib-rust`.
