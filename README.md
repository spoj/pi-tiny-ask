# pi-tiny-ask

A tiny pi extension that adds an `ask` tool for sending local media to another configured model or generating an image.

The main agent chooses the model from behavioral guidance in `extensions/ask.ts`; there are no settings.

## Try it

```bash
pi -e .
```

Then ask pi to inspect an image, PDF, audio file, or video. The tool accepts exact `provider/model` IDs and paths relative to pi's working directory.

Images work with image-capable adapters. PDFs work with Google, Anthropic, and OpenAI Responses adapters. Audio and video are limited to Google adapters. Files are sent inline, so this extension is intended for small files.

The tool can also generate an image with pi's configured Google, OpenAI, or OpenRouter authentication. Set `output` to the workspace-relative path where the image should be saved. Examples of model IDs are `google/gemini-3.1-flash-image`, `openai/gpt-image-2`, and `openrouter/google/gemini-3.1-flash-image`. Google and OpenRouter accept optional reference images through `files`; native OpenAI generation is currently text-to-image only.

## Install locally

```bash
pi install github.com/spoj/pi-tiny-ask
```
