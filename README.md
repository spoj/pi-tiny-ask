# pi-tiny-ask

A tiny pi extension that adds an `ask` tool for sending local media to another configured model or generating an image.

The main agent chooses the model from behavioral guidance in `extensions/ask.ts`; there are no settings.

## Try it

```bash
pi -e .
```

Then ask pi to inspect an image, PDF, audio file, or video. The tool accepts exact `provider/model` IDs and paths relative to pi's working directory.

Images work with image-capable adapters. PDFs work with Google, Anthropic, and OpenAI Responses adapters. Audio and video are limited to Google adapters. Files are sent inline, so this extension is intended for small files.

The tool can also generate an image with pi's configured Google, Google Vertex, OpenAI, or OpenRouter authentication. Set `output` to the workspace-relative path where the image should be saved. Examples of model IDs are `google/gemini-3.1-flash-image`, `google-vertex/gemini-3.1-flash-image`, `openai/gpt-image-2`, and `openrouter/google/gemini-3.1-flash-image`. Google, Google Vertex, and OpenRouter accept optional reference images through `files`; native OpenAI generation is currently text-to-image only. Vertex generation uses `GOOGLE_CLOUD_PROJECT` (or `GCLOUD_PROJECT`) and `GOOGLE_CLOUD_LOCATION` with ADC, or a configured `GOOGLE_CLOUD_API_KEY`.

## Install locally

```bash
pi install github.com/spoj/pi-tiny-ask
```
