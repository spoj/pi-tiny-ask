import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { GoogleGenAI, Modality, ResourceScope } from "@google/genai";
import type { Api, Model } from "@earendil-works/pi-ai";
import OpenAI, { toFile } from "openai";
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

type MediaKind = (typeof MEDIA)[keyof typeof MEDIA][1];
type MediaFile = {
  data: Buffer;
  name: string;
  mimeType: string;
  kind: MediaKind;
};
type RequestAuth = {
  auth: {
    apiKey?: string;
    headers?: Record<string, string | null>;
    baseUrl?: string;
  };
  env?: Record<string, string>;
};
type Request = {
  api: Api;
  apiKey?: string;
  baseUrl?: string;
  env: Record<string, string | undefined>;
  files: MediaFile[];
  headers: Record<string, string>;
  maxTokens?: number;
  modelId: string;
  output?: string;
  prompt: string;
  signal?: AbortSignal;
};

function generationApi(providerId: string, model: Model<Api> | undefined, providerModels: readonly Model<Api>[]): Api {
  if (model) return model.api;
  const apis = [...new Set(providerModels.map((candidate) => candidate.api))];
  if (apis.length === 1) return apis[0];
  if (providerId === "google") return "google-generative-ai";
  if (providerId === "google-vertex") return "google-vertex";
  if (providerId === "openai") return "openai-responses";
  throw new Error(`Cannot infer the API serialization format for unregistered model: ${providerId}`);
}

function assertKinds(files: MediaFile[], allowed: MediaKind[], api: Api): void {
  const unsupported = files.find((file) => !allowed.includes(file.kind));
  if (unsupported) throw new Error(`${api} does not support ${unsupported.kind} input: ${unsupported.name}`);
}

async function saveImage(output: string, data: Buffer): Promise<void> {
  if (data.length === 0) throw new Error("Image provider returned an empty image");
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, data);
}

async function callAnthropic(request: Request): Promise<string> {
  if (request.output) throw new Error("Anthropic does not support image generation");
  assertKinds(request.files, ["image", "document"], request.api);
  const content: Anthropic.Messages.ContentBlockParam[] = [{ type: "text", text: request.prompt }];
  for (const file of request.files) {
    content.push(file.kind === "document"
      ? {
        type: "document",
        source: { type: "base64", media_type: "application/pdf", data: file.data.toString("base64") },
      }
      : {
        type: "image",
        source: {
          type: "base64",
          media_type: file.mimeType as "image/jpeg" | "image/png" | "image/gif" | "image/webp",
          data: file.data.toString("base64"),
        },
      });
  }
  const client = new Anthropic({
    apiKey: request.apiKey,
    baseURL: request.baseUrl,
    defaultHeaders: request.headers,
  });
  const response = await client.messages.create({
    model: request.modelId,
    max_tokens: request.maxTokens ?? 16_384,
    messages: [{ role: "user", content }],
  }, { signal: request.signal });
  return response.content
    .filter((block) => block.type === "text")
    .map((block) => block.text)
    .join("\n")
    .trim();
}

async function callOpenRouterImage(request: Request): Promise<void> {
  const headers = new Headers(request.headers);
  headers.set("authorization", `Bearer ${request.apiKey}`);
  headers.set("content-type", "application/json");
  const response = await fetch(`${(request.baseUrl ?? "https://openrouter.ai/api/v1").replace(/\/$/, "")}/images`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: request.modelId,
      prompt: request.prompt,
      n: 1,
      ...(request.files.length ? {
        input_references: request.files.map((file) => ({
          type: "image_url",
          image_url: { url: `data:${file.mimeType};base64,${file.data.toString("base64")}` },
        })),
      } : {}),
    }),
    signal: request.signal,
  });
  if (!response.ok) throw new Error(`Image provider returned ${response.status} ${response.statusText}: ${await response.text()}`);
  const payload = await response.json() as { data?: Array<{ b64_json?: string }> };
  const data = payload.data?.[0]?.b64_json;
  if (!data) throw new Error("Image provider returned no image");
  await saveImage(request.output!, Buffer.from(data, "base64"));
}

async function callOpenAI(request: Request): Promise<string | undefined> {
  const client = new OpenAI({
    apiKey: request.apiKey ?? "pi-auth",
    baseURL: request.baseUrl,
    defaultHeaders: request.headers,
  });

  if (request.output) {
    assertKinds(request.files, ["image"], request.api);
    const unsupported = request.files.find((file) => file.mimeType === "image/gif");
    if (unsupported) throw new Error(`OpenAI image generation does not support GIF input: ${unsupported.name}`);
    const response = request.files.length
      ? await client.images.edit({
        model: request.modelId,
        prompt: request.prompt,
        image: await Promise.all(request.files.map((file) => toFile(file.data, file.name, { type: file.mimeType }))),
      }, { signal: request.signal })
      : await client.images.generate({ model: request.modelId, prompt: request.prompt }, { signal: request.signal });
    const image = response.data?.[0];
    if (image?.b64_json) {
      await saveImage(request.output, Buffer.from(image.b64_json, "base64"));
      return;
    }
    if (image?.url) {
      const result = await fetch(image.url, { signal: request.signal });
      if (!result.ok) throw new Error(`Image provider returned ${result.status} ${result.statusText}`);
      await saveImage(request.output, Buffer.from(await result.arrayBuffer()));
      return;
    }
    throw new Error("Image provider returned no image");
  }

  if (request.api === "openai-completions") {
    assertKinds(request.files, ["image"], request.api);
    const content: OpenAI.Chat.Completions.ChatCompletionContentPart[] = [{ type: "text", text: request.prompt }];
    for (const file of request.files) {
      content.push({
        type: "image_url",
        image_url: { url: `data:${file.mimeType};base64,${file.data.toString("base64")}` },
      });
    }
    const response = await client.chat.completions.create({
      model: request.modelId,
      messages: [{ role: "user", content }],
    }, { signal: request.signal });
    return response.choices.map((choice) => choice.message.content ?? "").join("\n").trim();
  }

  assertKinds(request.files, ["image", "document"], request.api);
  const content: OpenAI.Responses.ResponseInputContent[] = [{ type: "input_text", text: request.prompt }];
  for (const file of request.files) {
    content.push(file.kind === "document"
      ? {
        type: "input_file",
        filename: file.name,
        file_data: `data:${file.mimeType};base64,${file.data.toString("base64")}`,
      }
      : {
        type: "input_image",
        detail: "auto",
        image_url: `data:${file.mimeType};base64,${file.data.toString("base64")}`,
      });
  }
  const response = await client.responses.create({
    model: request.modelId,
    input: [{ role: "user", content }],
  }, { signal: request.signal });
  return response.output_text.trim();
}

async function callGoogle(request: Request): Promise<string | undefined> {
  const vertex = request.api === "google-vertex";
  const project = request.env.GOOGLE_CLOUD_PROJECT ?? request.env.GCLOUD_PROJECT;
  const location = request.env.GOOGLE_CLOUD_LOCATION;
  if (vertex && !request.apiKey && !project) {
    throw new Error("Vertex requires GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT");
  }
  if (vertex && !request.apiKey && !location) throw new Error("Vertex requires GOOGLE_CLOUD_LOCATION");
  const baseUrl = request.baseUrl?.includes("{location}") ? undefined : request.baseUrl;
  const versionedBaseUrl = baseUrl && new URL(baseUrl).pathname.split("/").some((part) => /^v\d+(?:beta\d*)?$/.test(part));
  const client = new GoogleGenAI(vertex
    ? {
      vertexai: true,
      ...(request.apiKey ? { apiKey: request.apiKey } : {
        project,
        location,
        ...(request.env.GOOGLE_APPLICATION_CREDENTIALS ? {
          googleAuthOptions: { keyFilename: request.env.GOOGLE_APPLICATION_CREDENTIALS },
        } : {}),
      }),
      apiVersion: "v1",
      httpOptions: {
        headers: request.headers,
        ...(baseUrl ? { baseUrl, baseUrlResourceScope: ResourceScope.COLLECTION } : {}),
        ...(versionedBaseUrl ? { apiVersion: "" } : {}),
      },
    }
    : {
      apiKey: request.apiKey,
      apiVersion: "v1beta",
      httpOptions: {
        headers: request.headers,
        ...(baseUrl ? { baseUrl } : {}),
        ...(versionedBaseUrl ? { apiVersion: "" } : {}),
      },
    });
  const response = await client.models.generateContent({
    model: request.modelId,
    contents: [{ role: "user", parts: [
      { text: request.prompt },
      ...request.files.map((file) => ({
        inlineData: { mimeType: file.mimeType, data: file.data.toString("base64") },
      })),
    ] }],
    config: {
      ...(request.output ? { responseModalities: [Modality.TEXT, Modality.IMAGE] } : {}),
      abortSignal: request.signal,
    },
  });
  if (!request.output) return response.text?.trim() ?? "";
  let data: string | undefined;
  for (const candidate of response.candidates ?? []) {
    for (const part of candidate.content?.parts ?? []) data = part.inlineData?.data ?? data;
  }
  if (!data) throw new Error("Image provider returned no image");
  await saveImage(request.output, Buffer.from(data, "base64"));
}

export default function (pi: ExtensionAPI): void {
  pi.registerTool({
    name: "ask",
    label: "Tiny Ask",
    description: "Send one prompt and local media files to a configured Anthropic, OpenAI-compatible, Google, or Google Vertex model. Set output to generate an image.",
    promptSnippet: "Inspect local media or generate an image with a configured model",
    promptGuidelines: [
      "Use ask when a task needs image, audio, video, or PDF understanding that would benefit from another model.",
      "To generate an image, set output to a workspace-relative path and use an OpenAI, OpenRouter, Google, or Google Vertex image model.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Exact provider/model ID" }),
      prompt: Type.String({ description: "What the other model should do" }),
      files: Type.Optional(Type.Array(Type.String(), { description: "Media paths relative to the workspace" })),
      output: Type.Optional(Type.String({ description: "Workspace-relative path for a generated image" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      const slash = params.model.indexOf("/");
      if (slash < 1) throw new Error("model must use provider/model format");
      const providerId = params.model.slice(0, slash);
      const modelId = params.model.slice(slash + 1);
      const model = ctx.modelRegistry.find(providerId, modelId);
      if (!model && !params.output) throw new Error(`Model not found: ${params.model}`);
      const provider = ctx.modelRegistry.getProvider(providerId);
      if (!provider) throw new Error(`Provider not found: ${providerId}`);
      const resolved = await ctx.modelRegistry.getProviderAuth(providerId) as RequestAuth | undefined;
      if (!resolved) throw new Error(`Provider has no configured authentication: ${providerId}`);

      let output: string | undefined;
      if (params.output) {
        if (path.isAbsolute(params.output)) throw new Error("output must be workspace-relative");
        const workspace = path.resolve(ctx.cwd);
        output = path.resolve(workspace, params.output);
        const relative = path.relative(workspace, output);
        if (relative.startsWith(`..${path.sep}`) || relative === ".." || path.isAbsolute(relative)) {
          throw new Error("output must stay within the workspace");
        }
      }

      const files = await Promise.all((params.files ?? []).map(async (file): Promise<MediaFile> => {
        const filePath = path.resolve(ctx.cwd, file.replace(/^@/, ""));
        const media = MEDIA[path.extname(filePath).toLowerCase() as keyof typeof MEDIA];
        if (!media) throw new Error(`Unsupported file: ${file}`);
        return {
          data: await readFile(filePath),
          name: path.basename(filePath),
          mimeType: media[0],
          kind: media[1],
        };
      }));
      const api = output && providerId === "openrouter"
        ? "openai-completions"
        : output ? generationApi(providerId, model, provider.getModels()) : model!.api;
      if (output) assertKinds(files, ["image"], api);

      const headers: Record<string, string> = {};
      for (const [name, value] of Object.entries({
        ...provider.headers,
        ...model?.headers,
        ...resolved.auth.headers,
      })) {
        if (value !== null) headers[name] = value;
      }
      const request: Request = {
        api,
        apiKey: resolved.auth.apiKey === "gcp-vertex-credentials" ? undefined : resolved.auth.apiKey,
        baseUrl: resolved.auth.baseUrl ?? model?.baseUrl ?? provider.baseUrl,
        env: { ...process.env, ...resolved.env },
        files,
        headers,
        maxTokens: model?.maxTokens,
        modelId,
        output,
        prompt: params.prompt,
        signal,
      };

      let answer: string | undefined;
      if (request.output && providerId === "openrouter") await callOpenRouterImage(request);
      else if (request.api === "anthropic-messages") answer = await callAnthropic(request);
      else if (request.api === "google-generative-ai" || request.api === "google-vertex") {
        answer = await callGoogle(request);
      } else if (request.api === "openai-completions" || request.api === "openai-responses" ||
        request.api === "openai-codex-responses" || request.api === "azure-openai-responses") {
        answer = await callOpenAI(request);
      } else {
        throw new Error(`Unsupported API serialization format: ${request.api}`);
      }

      return {
        content: [{ type: "text", text: output ? `Image saved to ${params.output}` : answer ?? "" }],
        details: { model: params.model, files: params.files ?? [], output: params.output },
      };
    },
  });
}
