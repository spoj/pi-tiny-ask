# pi-tiny-multimodal

A tiny pi extension that adds an `ask` tool for sending local media to another configured model.

The main agent chooses the model from behavioral guidance in `extensions/ask.ts`; there are no settings.

## Try it

```bash
pi -e .
```

Then ask pi to inspect an image, PDF, audio file, or video. The tool accepts exact `provider/model` IDs and paths relative to pi's working directory.

Images work with image-capable adapters. PDFs work with Google, Anthropic, and OpenAI Responses adapters. Audio and video are limited to Google adapters. Files are sent inline, so this extension is intended for small files.

## Install locally

```bash
pi install /absolute/path/to/pi-tiny-multimodal
```
