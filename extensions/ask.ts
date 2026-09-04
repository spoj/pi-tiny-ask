import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { GoogleGenAI, Modality, ResourceScope } from "@google/genai";
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

function isOpenAIResponses(model: Model<Api>): boolean {
  return model.api === "openai-responses" ||
    model.api === "openai-codex-responses" ||
    model.api === "azure-openai-responses";
}

function supports(model: Model<Api>, kind: (typeof MEDIA)[keyof typeof MEDIA][1]): boolean {
  if (!model.input.includes("image")) return false;
  if (kind === "image") return true;
  if (kind === "document") {
    return isGoogle(model) || model.api === "anthropic-messages" || isOpenAIResponses(model);
  }
  return isGoogle(model);
}

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined;
}

type RequestAuth = {
  auth: {
    apiKey?: string;
    headers?: Record<string, string | null>;
    baseUrl?: string;
  };
  env?: Record<string, string>;
};

async function readJson(response: Response): Promise<Record<string, unknown>> {
  const text = await response.text();
  const payload = record(JSON.parse(text));
  if (!response.ok) {
    const error = record(payload?.error);
    throw new Error(typeof error?.message === "string" ? error.message : `${response.status} ${response.statusText}`);
  }
  if (!payload) throw new Error("Image provider returned an invalid response");
  return payload;
}

async function generateImage(
  ctx: ExtensionContext,
  spec: string,
  prompt: string,
  files: string[],
  outputPath: string,
  signal: AbortSignal | undefined,
): Promise<void> {
  const slash = spec.indexOf("/");
  if (slash < 1) throw new Error("model must use provider/model format");
  const providerId = spec.slice(0, slash);
  const modelId = spec.slice(slash + 1);
  if (providerId !== "google" && providerId !== "google-vertex" && providerId !== "openai" && providerId !== "openrouter") {
    throw new Error("Image generation supports google, google-vertex, openai, and openrouter providers");
  }
  if (path.isAbsolute(outputPath)) throw new Error("output must be workspace-relative");
  const workspace = path.resolve(ctx.cwd);
  const output = path.resolve(workspace, outputPath);
  const relative = path.relative(workspace, output);
  if (relative.startsWith(`..${path.sep}`) || relative === ".." || path.isAbsolute(relative)) {
    throw new Error("output must stay within the workspace");
  }

  const resolved = await ctx.modelRegistry.getProviderAuth(providerId) as RequestAuth | undefined;
  if (!resolved) throw new Error(`Provider has no configured authentication: ${providerId}`);
  const provider = ctx.modelRegistry.getProvider(providerId);
  const headers = new Headers({ "content-type": "application/json" });
  for (const [name, value] of Object.entries({ ...provider?.headers, ...resolved.auth.headers })) {
    if (value !== null) headers.set(name, value);
  }
  if (resolved.auth.apiKey) {
    if (providerId === "google") headers.set("x-goog-api-key", resolved.auth.apiKey);
    if (providerId === "openai" || providerId === "openrouter") {
      headers.set("authorization", `Bearer ${resolved.auth.apiKey}`);
    }
  }

  const images = await Promise.all(files.map(async (file): Promise<ImageContent> => {
    const filePath = path.resolve(ctx.cwd, file.replace(/^@/, ""));
    const media = MEDIA[path.extname(filePath).toLowerCase() as keyof typeof MEDIA];
    if (!media || media[1] !== "image") throw new Error(`Image generation input must be an image: ${file}`);
    return { type: "image", data: (await readFile(filePath)).toString("base64"), mimeType: media[0] };
  }));

  if (providerId === "google-vertex") {
    const env = { ...process.env, ...resolved.env };
    const apiKey = resolved.auth.apiKey && resolved.auth.apiKey !== "gcp-vertex-credentials" ? resolved.auth.apiKey : undefined;
    const project = env.GOOGLE_CLOUD_PROJECT ?? env.GCLOUD_PROJECT;
    const location = env.GOOGLE_CLOUD_LOCATION;
    if (!apiKey && !project) throw new Error("Vertex image generation requires GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT");
    if (!apiKey && !location) throw new Error("Vertex image generation requires GOOGLE_CLOUD_LOCATION");
    const configuredBaseUrl = resolved.auth.baseUrl ?? provider?.baseUrl;
    const baseUrl = configuredBaseUrl?.includes("{location}") ? undefined : configuredBaseUrl;
    const client = new GoogleGenAI({
      vertexai: true,
      ...(apiKey ? { apiKey } : {
        project,
        location,
        ...(env.GOOGLE_APPLICATION_CREDENTIALS ? {
          googleAuthOptions: { keyFilename: env.GOOGLE_APPLICATION_CREDENTIALS },
        } : {}),
      }),
      apiVersion: "v1",
      httpOptions: {
        headers: Object.fromEntries(headers),
        ...(baseUrl ? { baseUrl, baseUrlResourceScope: ResourceScope.COLLECTION } : {}),
      },
    });
    const response = await client.models.generateContent({
      model: modelId,
      contents: [{ role: "user", parts: [
        { text: prompt },
        ...images.map((image) => ({ inlineData: { mimeType: image.mimeType, data: image.data } })),
      ] }],
      config: { responseModalities: [Modality.TEXT, Modality.IMAGE], abortSignal: signal },
    });
    let data: Buffer | undefined;
    for (const candidate of response.candidates ?? []) {
      for (const part of candidate.content?.parts ?? []) {
        if (part.inlineData?.data) data = Buffer.from(part.inlineData.data, "base64");
      }
    }
    if (!data || data.length === 0) throw new Error("Image provider returned no image");
    await mkdir(path.dirname(output), { recursive: true });
    await writeFile(output, data);
    return;
  }

  let url: string;
  let body: Record<string, unknown>;
  if (providerId === "google") {
    const baseUrl = resolved.auth.baseUrl ?? provider?.baseUrl ?? "https://generativelanguage.googleapis.com/v1beta";
    url = `${baseUrl.replace(/\/$/, "")}/models/${encodeURIComponent(modelId)}:generateContent`;
    body = {
      contents: [{ role: "user", parts: [
        { text: prompt },
        ...images.map((image) => ({ inlineData: { mimeType: image.mimeType, data: image.data } })),
      ] }],
      generationConfig: { responseModalities: ["TEXT", "IMAGE"] },
    };
  } else {
    if (providerId === "openai" && images.length) {
      throw new Error("Native OpenAI image generation does not accept reference files through ask; use Google or OpenRouter");
    }
    const baseUrl = resolved.auth.baseUrl ?? provider?.baseUrl ??
      (providerId === "openrouter" ? "https://openrouter.ai/api/v1" : "https://api.openai.com/v1");
    url = `${baseUrl.replace(/\/$/, "")}/${providerId === "openrouter" ? "images" : "images/generations"}`;
    body = {
      model: modelId,
      prompt,
      n: 1,
      ...(providerId === "openrouter" && images.length ? {
        input_references: images.map((image) => ({
          type: "image_url",
          image_url: { url: `data:${image.mimeType};base64,${image.data}` },
        })),
      } : {}),
    };
  }

  const payload = await readJson(await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
    signal,
  }));

  let data: Buffer | undefined;
  if (providerId === "google") {
    const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
    for (const candidateValue of candidates) {
      const parts = record(record(candidateValue)?.content)?.parts;
      if (!Array.isArray(parts)) continue;
      for (const partValue of parts) {
        const inlineData = record(record(partValue)?.inlineData ?? record(partValue)?.inline_data);
        if (typeof inlineData?.data === "string") data = Buffer.from(inlineData.data, "base64");
      }
    }
  } else {
    const results = Array.isArray(payload.data) ? payload.data : [];
    const image = record(results[0]);
    if (typeof image?.b64_json === "string") {
      data = Buffer.from(image.b64_json, "base64");
    } else if (typeof image?.url === "string") {
      const response = await fetch(image.url, { signal });
      if (!response.ok) throw new Error(`Image provider returned ${response.status} ${response.statusText}`);
      data = Buffer.from(await response.arrayBuffer());
    }
  }
  if (!data || data.length === 0) throw new Error("Image provider returned no image");
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, data);
}

function normalizePayload(payload: unknown, model: Model<Api>): unknown {
  const body = record(payload);

  if (model.api === "anthropic-messages" && Array.isArray(body?.messages)) {
    for (const messageValue of body.messages) {
      const message = record(messageValue);
      if (!Array.isArray(message?.content)) continue;
      for (const blockValue of message.content) {
        const block = record(blockValue);
        const source = record(block?.source);
        if (block?.type === "image" && source?.media_type === "application/pdf") {
          block.type = "document";
        }
      }
    }
  }

  if (isOpenAIResponses(model) && Array.isArray(body?.input)) {
    for (const itemValue of body.input) {
      const item = record(itemValue);
      if (!Array.isArray(item?.content)) continue;
      item.content = item.content.map((partValue) => {
        const part = record(partValue);
        const imageUrl = part?.image_url;
        if (part?.type !== "input_image" || typeof imageUrl !== "string" ||
          !imageUrl.startsWith("data:application/pdf;")) return partValue;
        return { type: "input_file", file_data: imageUrl };
      });
    }
  }

  return payload;
}

export default function (pi: ExtensionAPI): void {
  pi.registerTool({
    name: "ask",
    label: "Tiny Ask",
    description: "Ask a configured model to inspect local media or generate an image. Set output to generate with Google, Google Vertex, OpenAI, or OpenRouter and save the returned image.",
    promptSnippet: "Inspect media or generate an image with a configured model",
    promptGuidelines: [
      "Use ask when a task needs image, audio, video, or PDF understanding that would benefit from another model.",
      "To generate an image, set output to a workspace-relative image path and use a Google, Google Vertex, OpenAI, or OpenRouter image model.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Exact provider/model ID" }),
      prompt: Type.String({ description: "What the other model should do" }),
      files: Type.Optional(Type.Array(Type.String(), { description: "Media paths relative to the workspace" })),
      output: Type.Optional(Type.String({ description: "Workspace-relative path for a generated image" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const files = params.files ?? [];
      if (params.output) {
        await generateImage(ctx, params.model, params.prompt, files, params.output, signal);
        return {
          content: [{ type: "text", text: `Image saved to ${params.output}` }],
          details: { model: params.model, files, output: params.output },
        };
      }

      const model = resolveModel(ctx, params.model);
      const content: Array<TextContent | ImageContent> = [{ type: "text", text: params.prompt }];

      for (const file of files) {
        const filePath = path.resolve(ctx.cwd, file.replace(/^@/, ""));
        const media = MEDIA[path.extname(filePath).toLowerCase() as keyof typeof MEDIA];
        if (!media) throw new Error(`Unsupported file: ${file}`);

        const [mimeType, kind] = media;
        if (!supports(model, kind)) {
          throw new Error(`${params.model} does not support ${kind} input through its ${model.api} adapter`);
        }

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
      const response = await ctx.modelRegistry.complete(model, completionContext, {
        signal,
        onPayload: (payload, requestModel) => normalizePayload(payload, requestModel),
      });
      if (response.stopReason === "error") throw new Error(response.errorMessage ?? "Model completion failed");

      const answer = response.content
        .filter((part): part is TextContent => part.type === "text")
        .map((part) => part.text)
        .join("\n")
        .trim();

      return {
        content: [{ type: "text", text: answer }],
        details: { model: params.model, files },
        usage: response.usage,
      };
    },
  });
}
