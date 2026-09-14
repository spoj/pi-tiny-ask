# pi-tiny-ask

A tiny pi extension that adds an `ask` tool for sending local media to another configured model or generating an image.

The main agent chooses the model from behavioral guidance in `extensions/ask.ts`; there are no settings.

## Try it

```bash
pi -e .
```

Then ask pi to inspect an image, PDF, audio file, or video. The tool accepts standard `provider/model` IDs and paths relative to pi's working directory. The optional `api` parameter overrides only the request serializer; it does not change the provider endpoint, authentication, or model ID. When omitted, the registered model API remains authoritative. An explicit `api` also permits models absent from the local catalog. The configured endpoint must accept the selected serializer.

```json
{
  "model": "my-gateway/gemini-3.5-flash-lite",
  "api": "openai-completions",
  "prompt": "Transcribe this audio.",
  "files": ["voice.oga"]
}
```

Each call sends one prompt and all files in one native SDK request. Anthropic Messages accepts images and PDFs, OpenAI Responses accepts images and PDFs, OpenAI Chat Completions sends images, audio, and PDFs using their standard content blocks, and Google AI or Vertex sends images, PDFs, audio, and video inline. Video is not represented by the OpenAI Chat Completions serializer. Provider/model support still determines whether a request succeeds; this extension does not switch models or retry with another format. Audio is sent without transcoding; `.oga` is normalized to the `ogg` format name. Files are sent inline, so this extension is intended for small files.

Requests use the provider's configured endpoint; providers without one are rejected rather than falling back to an SDK default, except Google Vertex, which derives its own endpoint. Azure OpenAI and Codex transports are not supported; use Pi directly for those providers. Truncated, filtered, or refused responses return any partial text plus an explicit `[ask: ...]` status note.

Text responses up to 50 KiB / 2,000 lines are returned directly. Larger responses return a head preview and the path to the complete response in a temporary file; use `read` with `offset` and `limit` to inspect sections. This limits model-context output, not generation. Provider truncation or refusal notices remain visible beside the preview. Response files remain available until normal operating-system temp cleanup.

The tool can also generate an image with pi's configured Google, Google Vertex, OpenAI, or OpenRouter authentication. Set `output` to the workspace-relative path where the image should be saved. Examples are `google/gemini-3.1-flash-image`, `google-vertex/gemini-3.1-flash-image`, `openai/gpt-image-2`, and `openrouter/bytedance-seed/seedream-4.5`. All four accept optional reference images through `files`; OpenAI uses its image edit API, while OpenRouter uses its image endpoint only when no explicit `api` override is supplied. Vertex generation supports a configured API key or ADC with project and location.

## Install locally

```bash
pi install github.com/spoj/pi-tiny-ask
```
