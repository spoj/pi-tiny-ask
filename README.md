# pi-tiny-multimodal

A tiny pi extension that adds an `ask` tool for sending local media to another configured model.

The main agent chooses the model from behavioral guidance in `extensions/ask.ts`; there are no settings.

## Try it

```bash
pi -e .
```

Then ask pi to inspect an image, PDF, audio file, or video. The tool accepts exact `provider/model` IDs and paths relative to pi's working directory.

Images work with image-capable adapters. Audio, video, and PDFs are limited to Google adapters because pi-ai represents binary chat input as image content and Google serializes its MIME type as generic inline data.

## Install locally

```bash
pi install /absolute/path/to/pi-tiny-multimodal
```
