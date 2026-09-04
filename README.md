# pi-tiny-ask

A tiny pi extension that adds an `ask` tool for sending local media to another configured model or generating an image.

The main agent chooses the model from behavioral guidance in `extensions/ask.ts`; there are no settings.

## Try it

```bash
pi -e .
```

Then ask pi to inspect an image, PDF, audio file, or video. The tool accepts exact `provider/model` IDs and paths relative to pi's working directory.

Each call sends one prompt and all files in one native SDK request. Routing follows the selected model's pi API format: Anthropic Messages accepts images and PDFs, OpenAI Responses accepts images and PDFs, OpenAI Chat Completions accepts images, and Google AI or Vertex accepts images, PDFs, audio, and video. Files are sent inline, so this extension is intended for small files.

The tool can also generate an image with pi's configured Google, Google Vertex, or OpenAI authentication. Set `output` to the workspace-relative path where the image should be saved. Examples are `google/gemini-3.1-flash-image`, `google-vertex/gemini-3.1-flash-image`, and `openai/gpt-image-2`. All three accept optional reference images through `files`; OpenAI uses its image edit API when references are present. Vertex generation supports a configured API key or ADC with project and location.

## Install locally

```bash
pi install github.com/spoj/pi-tiny-ask
```
