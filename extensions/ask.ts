import { readFile } from "node:fs/promises";
import path from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { Api, Context, ImageContent, Model, TextContent } from "@earendil-works/pi-ai";
import { Type } from "typebox";

const MEDIA = {
  ".png": ["image/png", "image"],
  ".jpg": ["image/jpeg", "image"],
  ".jpeg": ["image/jpeg", "image"],
  ".webp": ["image/webp", "image"],
  ".gif": ["image/gif", "image"],
  ".pdf": ["application/pdf", "document"],
  ".oga": ["audio/ogg", "audio"],
  ".ogg": ["audio/ogg", "audio"],
  ".opus": ["audio/opus", "audio"],
  ".mp3": ["audio/mpeg", "audio"],
  ".wav": ["audio/wav", "audio"],
  ".m4a": ["audio/mp4", "audio"],
  ".aac": ["audio/aac", "audio"],
  ".mp4": ["video/mp4", "video"],
  ".webm": ["video/webm", "video"],
  ".mov": ["video/quicktime", "video"],
} as const;

function resolveModel(ctx: ExtensionContext, spec: string): Model<Api> {
  const slash = spec.indexOf("/");
  if (slash < 1) throw new Error("model must use provider/model format");

  const model = ctx.modelRegistry.find(spec.slice(0, slash), spec.slice(slash + 1));
  if (!model) throw new Error(`Model not found: ${spec}`);
  if (!ctx.modelRegistry.hasConfiguredAuth(model)) throw new Error(`Model has no configured authentication: ${spec}`);
  return model;
}

function isGoogle(model: Model<Api>): boolean {
  return model.api === "google-generative-ai" || model.api === "google-vertex";
}

export default function (pi: ExtensionAPI): void {
  pi.registerTool({
    name: "ask",
    label: "Ask",
    description: "Ask a configured multimodal model to inspect local image, audio, video, or PDF files. Model must be an exact provider/model ID. Images work with image-capable models; audio, video, and PDFs require a Google adapter.",
    promptSnippet: "Ask another configured model to inspect local media files",
    promptGuidelines: [
      "Use ask when a task needs image, audio, video, or PDF understanding that would benefit from a multimodal model or a second opinion.",
      "When using ask, choose an exact configured provider/model ID; prefer a Google Gemini model for audio, video, and PDFs, and a strong image-capable model for images.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Exact provider/model ID" }),
      prompt: Type.String({ description: "What the other model should do" }),
      files: Type.Array(Type.String(), { description: "Media paths relative to the workspace" }),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const model = resolveModel(ctx, params.model);
      const content: Array<TextContent | ImageContent> = [{ type: "text", text: params.prompt }];

      for (const file of params.files) {
        const filePath = path.resolve(ctx.cwd, file.replace(/^@/, ""));
        const media = MEDIA[path.extname(filePath).toLowerCase() as keyof typeof MEDIA];
        if (!media) throw new Error(`Unsupported file: ${file}`);

        const [mimeType, kind] = media;
        if (!model.input.includes("image")) throw new Error(`${params.model} does not accept media input`);
        if (kind !== "image" && !isGoogle(model)) throw new Error(`${kind} input requires a Google adapter`);

        content.push({
          type: "image",
          data: (await readFile(filePath)).toString("base64"),
          mimeType,
        });
      }

      const completionContext: Context = {
        systemPrompt: "Follow the user's request and accurately analyze the provided media.",
        messages: [{ role: "user", content, timestamp: Date.now() }],
      };
      const response = await ctx.modelRegistry.complete(model, completionContext, { signal });
      if (response.stopReason === "error") throw new Error(response.errorMessage ?? "Model completion failed");

      const answer = response.content
        .filter((part): part is TextContent => part.type === "text")
        .map((part) => part.text)
        .join("\n")
        .trim();

      return {
        content: [{ type: "text", text: answer }],
        details: { model: params.model, files: params.files },
        usage: response.usage,
      };
    },
  });
}
