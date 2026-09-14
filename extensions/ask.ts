import { mkdir, mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import { truncateHead, type ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { FinishReason, GoogleGenAI, Modality, ResourceScope } from "@google/genai";
import { StringEnum, type Api, type Model, type OpenAICompletionsCompat } from "@earendil-works/pi-ai";
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

const MEDIA_KINDS = {
  "anthropic-messages": ["image", "document"],
  "openai-completions": ["image", "audio", "document"],
  "openai-responses": ["image", "document"],
  "google-generative-ai": ["image", "document", "audio", "video"],
  "google-vertex": ["image", "document", "audio", "video"],
} satisfies Record<string, readonly MediaKind[]>;
const SUPPORTED_APIS = Object.keys(MEDIA_KINDS) as (keyof typeof MEDIA_KINDS)[];

type MediaFile = {
  data: Buffer;
  name: string;
  mimeType: string;
  kind: MediaKind;
};
type Request = {
  api: Api;
  apiKey?: string;
  baseUrl?: string;
  env: Record<string, string | undefined>;
  files: MediaFile[];
  headers: Record<string, string | null>;
  maxTokens?: number;
  providerId: string;
  modelId: string;
  output?: string;
  prompt: string;
  routing: Record<string, unknown>;
  samplingParams?: Record<string, unknown>;
  signal?: AbortSignal;
};
type Answer = { text: string; status?: string };

function generationApi(providerId: string, model: Model<Api> | undefined, providerModels: readonly Model<Api>[]): Api {
  if (model) return model.api;
  const apis = [...new Set(providerModels.map((candidate) => candidate.api))];
  if (apis.length === 1) return apis[0];
  if (providerId === "google") return "google-generative-ai";
  if (providerId === "google-vertex") return "google-vertex";
  if (providerId === "openai") return "openai-responses";
  throw new Error(`Cannot infer the API serialization format for unregistered model: ${providerId}`);
}

async function saveImage(output: string, data: Buffer): Promise<void> {
  if (data.length === 0) throw new Error("Image provider returned an empty image");
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(output, data);
}

function dataUri(file: MediaFile): string {
  return `data:${file.mimeType};base64,${file.data.toString("base64")}`;
}

async function callAnthropic(request: Request): Promise<Answer> {
  if (request.output) throw new Error("Anthropic does not support image generation");
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
  // Anthropic OAuth requires the Claude Code identity as well as bearer auth.
  const oauth = request.apiKey?.includes("sk-ant-oat") ?? false;
  const bearer = oauth || request.providerId === "github-copilot";
  const client = new Anthropic({
    apiKey: bearer ? null : request.apiKey ?? "pi-auth",
    authToken: bearer ? request.apiKey ?? null : null,
    baseURL: request.baseUrl,
    defaultHeaders: {
      ...(oauth ? {
        "anthropic-beta": "claude-code-20250219,oauth-2025-04-20",
        "user-agent": "claude-cli/2.1.75",
        "x-app": "cli",
      } : {}),
      ...request.headers,
    },
    maxRetries: 0,
  });
  const response = await client.messages.stream({
    model: request.modelId,
    max_tokens: request.maxTokens ?? 16_384,
    ...(oauth ? { system: "You are Claude Code, Anthropic's official CLI for Claude." } : {}),
    messages: [{ role: "user", content }],
    ...request.routing,
  }, { signal: request.signal }).finalMessage();
  const status = response.stop_reason === "max_tokens"
    ? "response truncated (max_tokens)"
    : response.stop_reason === "refusal" ? "model refused the request" : undefined;
  return {
    text: response.content
      .filter((block) => block.type === "text")
      .map((block) => block.text)
      .join("\n")
      .trim(),
    status,
  };
}

async function callOpenRouterImage(request: Request): Promise<void> {
  const headers = new Headers();
  if (request.apiKey) headers.set("authorization", `Bearer ${request.apiKey}`);
  for (const [name, value] of Object.entries(request.headers)) {
    if (value === null) headers.delete(name);
    else headers.set(name, value);
  }
  headers.set("content-type", "application/json");
  const response = await fetch(`${request.baseUrl!.replace(/\/$/, "")}/images`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: request.modelId,
      prompt: request.prompt,
      n: 1,
      ...request.routing,
      ...(request.files.length ? {
        input_references: request.files.map((file) => ({
          type: "image_url",
          image_url: { url: dataUri(file) },
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

async function callOpenAI(request: Request): Promise<Answer | undefined> {
  const client = new OpenAI({
    apiKey: request.apiKey ?? "pi-auth",
    baseURL: request.baseUrl,
    defaultHeaders: request.headers,
    maxRetries: 0,
  });

  if (request.output) {
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
    const content: Array<OpenAI.Chat.Completions.ChatCompletionContentPart | {
      type: "input_audio";
      input_audio: { data: string; format: string };
    }> = [{ type: "text", text: request.prompt }];
    for (const file of request.files) {
      if (file.kind === "audio") {
        const extension = path.extname(file.name).slice(1).toLowerCase();
        content.push({
          type: "input_audio",
          input_audio: { data: file.data.toString("base64"), format: extension === "oga" ? "ogg" : extension },
        });
      } else if (file.kind === "document") {
        content.push({
          type: "file",
          file: { filename: file.name, file_data: dataUri(file) },
        });
      } else {
        content.push({ type: "image_url", image_url: { url: dataUri(file) } });
      }
    }
    const response = await client.chat.completions.create({
      model: request.modelId,
      messages: [{
        role: "user",
        // Compatible endpoints can accept audio formats beyond the SDK's WAV/MP3 types.
        content: content as OpenAI.Chat.Completions.ChatCompletionContentPart[],
      }],
      ...request.routing,
      ...request.samplingParams,
    }, { signal: request.signal });
    const choice = response.choices[0];
    const status = choice?.message.refusal
      ? "model refused the request"
      : choice?.finish_reason === "length"
        ? "response truncated (length)"
        : choice?.finish_reason === "content_filter" ? "response blocked by content filter" : undefined;
    return {
      text: response.choices.map((choice) => choice.message.content ?? choice.message.refusal ?? "").join("\n").trim(),
      status,
    };
  }

  const content: OpenAI.Responses.ResponseInputContent[] = [{ type: "input_text", text: request.prompt }];
  for (const file of request.files) {
    content.push(file.kind === "document"
      ? {
        type: "input_file",
        filename: file.name,
        file_data: dataUri(file),
      }
      : {
        type: "input_image",
        detail: "auto",
        image_url: dataUri(file),
      });
  }
  const response = await client.responses.create({
    model: request.modelId,
    input: [{ role: "user", content }],
    ...request.routing,
    ...request.samplingParams,
  }, { signal: request.signal });
  const refusal = response.output
    .filter((item) => item.type === "message")
    .flatMap((item) => item.content)
    .find((content) => content.type === "refusal");
  const status = refusal
    ? "model refused the request"
    : response.status === "incomplete"
      ? `response incomplete (${response.incomplete_details?.reason ?? "unknown"})`
      : response.status === "failed"
        ? (response.error?.message ? `response failed: ${response.error.message}` : "response failed")
        : response.status === "cancelled" ? "response cancelled" : undefined;
  const text = response.output_text ?? "";
  return { text: text || refusal?.refusal || "", status };
}

async function callGoogle(request: Request): Promise<Answer | undefined> {
  const vertex = request.api === "google-vertex";
  const project = request.env.GOOGLE_CLOUD_PROJECT ?? request.env.GCLOUD_PROJECT;
  const location = request.env.GOOGLE_CLOUD_LOCATION;
  const baseUrl = request.baseUrl?.includes("{location}") ? undefined : request.baseUrl;
  const vertexApiKey = request.apiKey ?? (vertex && baseUrl && (!project || !location) ? "pi-auth" : undefined);
  if (vertex && !baseUrl && !vertexApiKey && !project) {
    throw new Error("Vertex requires GOOGLE_CLOUD_PROJECT or GCLOUD_PROJECT");
  }
  if (vertex && !baseUrl && !vertexApiKey && !location) throw new Error("Vertex requires GOOGLE_CLOUD_LOCATION");
  const versionedBaseUrl = baseUrl && new URL(baseUrl).pathname.split("/").some((part) => /^v\d+(?:beta\d*)?$/.test(part));
  const headers: Record<string, string> = {};
  for (const [name, value] of Object.entries(request.headers)) {
    if (value === null) {
      if (["authorization", "x-goog-api-key", "content-type", "user-agent", "x-goog-api-client"].includes(name)) {
        throw new Error(`Google SDK cannot suppress configured header: ${name}`);
      }
      continue;
    }
    headers[name === "user-agent" ? "User-Agent" : name === "content-type" ? "Content-Type" : name] = value;
  }
  const httpOptions = {
    headers,
    retryOptions: { attempts: 1 },
    ...(baseUrl ? {
      baseUrl,
      ...(vertex ? { baseUrlResourceScope: ResourceScope.COLLECTION } : {}),
    } : {}),
    ...(versionedBaseUrl ? { apiVersion: "" } : {}),
  };
  const client = new GoogleGenAI({
    vertexai: vertex,
    ...(vertex
      ? vertexApiKey ? { apiKey: vertexApiKey } : {
        project,
        location,
        ...(request.env.GOOGLE_APPLICATION_CREDENTIALS ? {
          googleAuthOptions: { keyFilename: request.env.GOOGLE_APPLICATION_CREDENTIALS },
        } : {}),
      }
      : { apiKey: request.apiKey ?? "pi-auth" }),
    apiVersion: vertex ? "v1" : "v1beta",
    httpOptions,
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
  const finishReason = response.candidates
    ?.map((candidate) => candidate.finishReason)
    .find((reason) => reason !== undefined && reason !== FinishReason.STOP && reason !== FinishReason.FINISH_REASON_UNSPECIFIED);
  const status = finishReason
    ? finishReason === FinishReason.MAX_TOKENS ? "response truncated (MAX_TOKENS)" : `response stopped early (${finishReason})`
    : response.promptFeedback?.blockReason ? `prompt blocked (${response.promptFeedback.blockReason})` : undefined;
  if (!request.output) return { text: response.text?.trim() ?? "", status };
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
    description: "Send one prompt and local media files to a configured Anthropic, OpenAI-compatible, Google, or Google Vertex model. Set output to generate an image. Text previews are limited to 50 KiB or 2,000 lines; larger responses are saved to a temp file for reading in sections.",
    promptSnippet: "Inspect local media or generate an image with a configured model",
    promptGuidelines: [
      "Use ask when a task needs image, audio, video, or PDF understanding that would benefit from another model.",
      "To generate an image, set output to a workspace-relative path and use an OpenAI, OpenRouter, Google, or Google Vertex image model.",
    ],
    parameters: Type.Object({
      model: Type.String({ description: "Exact provider/model ID" }),
      api: Type.Optional(StringEnum(SUPPORTED_APIS, { description: "Override the API serializer; with output, OpenAI values select the Images API" })),
      prompt: Type.String({ description: "What the other model should do" }),
      files: Type.Optional(Type.Array(Type.String(), { description: "Media paths relative to the workspace" })),
      output: Type.Optional(Type.String({ description: "Workspace-relative path for a generated image" })),
    }),
    async execute(_toolCallId, params, signal, _onUpdate, ctx) {
      signal?.throwIfAborted();
      const slash = params.model.indexOf("/");
      if (slash < 1) throw new Error("model must use provider/model format");
      const providerId = params.model.slice(0, slash);
      const modelId = params.model.slice(slash + 1);
      const model = ctx.modelRegistry.find(providerId, modelId);
      if (!model && !params.api && !params.output) throw new Error(`Model not found: ${params.model}`);
      const provider = ctx.modelRegistry.getProvider(providerId);
      if (!provider) throw new Error(`Provider not found: ${providerId}`);
      let resolved: { apiKey?: string; baseUrl?: string; headers?: Record<string, string | null>; env?: Record<string, string> };
      if (model) {
        const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
        if (!auth.ok) throw new Error(auth.error);
        resolved = auth;
      } else {
        const auth = await ctx.modelRegistry.getProviderAuth(providerId);
        if (!auth) throw new Error(`Provider has no configured authentication: ${providerId}`);
        resolved = { ...auth.auth, env: auth.env };
      }

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

      const openRouterImage = output !== undefined && providerId === "openrouter" && !params.api;
      const api = params.api ?? (openRouterImage
        ? "openai-completions"
        : output ? generationApi(providerId, model, provider.getModels()) : model!.api);
      if (!(SUPPORTED_APIS as readonly string[]).includes(api)) {
        throw new Error(api === "openai-codex-responses" || api === "azure-openai-responses"
          ? `ask does not support the ${api} transport`
          : `Unsupported API serialization format: ${api}`);
      }
      if (!model && provider.getModels().some((candidate) => {
        const compat = candidate.compat as OpenAICompletionsCompat | undefined;
        return compat?.openRouterRouting || compat?.vercelGatewayRouting;
      })) {
        throw new Error(`Register ${params.model} to resolve its gateway routing; ask cannot infer it from other models`);
      }
      const compat = model?.compat as OpenAICompletionsCompat | undefined;
      const routing = {
        ...(compat?.openRouterRouting ? { provider: compat.openRouterRouting } : {}),
        ...(compat?.vercelGatewayRouting ? { providerOptions: { gateway: compat.vercelGatewayRouting } } : {}),
      };
      const openAIText = !output && (api === "openai-completions" || api === "openai-responses");
      if (openAIText && Object.keys(routing).some((key) => key in (model?.samplingParams ?? {}))) {
        throw new Error("Configure gateway routing in compat or samplingParams, not both");
      }
      const routedText = openAIText || (!output && api === "anthropic-messages");
      if (Object.keys(routing).length && !routedText) {
        if (!openRouterImage || compat?.vercelGatewayRouting) {
          throw new Error(`ask cannot preserve configured gateway routing for ${output ? "image generation with " : ""}${api}`);
        }
        const unsupported = Object.keys(compat?.openRouterRouting ?? {}).find((key) =>
          !["only", "order", "ignore", "sort", "allow_fallbacks"].includes(key));
        if (unsupported) throw new Error(`OpenRouter images do not support configured routing option: ${unsupported}`);
      }
      const env = { ...process.env, ...resolved.env };
      let baseUrl = resolved.baseUrl ?? model?.baseUrl ?? provider.baseUrl;
      if (providerId === "cloudflare-ai-gateway" || providerId === "cloudflare-workers-ai") {
        for (const name of ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_GATEWAY_ID"]) {
          if (!baseUrl?.includes(`{${name}}`)) continue;
          if (!env[name]) throw new Error(`Provider endpoint requires ${name}`);
          baseUrl = baseUrl.replaceAll(`{${name}}`, env[name]);
        }
      }
      const nativeVertex = !params.api && api === "google-vertex";
      if (!baseUrl && !nativeVertex) throw new Error(`Provider has no configured endpoint: ${providerId}`);

      const allowed: readonly MediaKind[] = output ? ["image"] : MEDIA_KINDS[api as keyof typeof MEDIA_KINDS];
      const files = await Promise.all((params.files ?? []).map(async (file): Promise<MediaFile> => {
        const filePath = path.resolve(ctx.cwd, file.replace(/^@/, ""));
        const media = MEDIA[path.extname(filePath).toLowerCase() as keyof typeof MEDIA];
        if (!media) throw new Error(`Unsupported file: ${file}`);
        const name = path.basename(filePath);
        const [mimeType, kind] = media;
        if (!allowed.includes(kind)) {
          throw new Error(`ask ${api} serializer does not support ${kind} input: ${name}`);
        }
        if (output && !openRouterImage && mimeType === "image/gif" && (api === "openai-completions" || api === "openai-responses")) {
          throw new Error(`OpenAI image generation does not support GIF input: ${name}`);
        }
        return {
          data: await readFile(filePath),
          name,
          mimeType,
          kind,
        };
      }));

      const headers: Record<string, string | null> = providerId === "github-copilot" ? {
        "x-initiator": "agent",
        "openai-intent": "conversation-edits",
        ...(files.some((file) => file.kind === "image") ? { "copilot-vision-request": "true" } : {}),
      } : {};
      for (const [name, value] of Object.entries({ ...provider.headers, ...resolved.headers })) {
        headers[name.toLowerCase()] = value;
      }
      const request: Request = {
        api,
        apiKey: resolved.apiKey,
        baseUrl,
        env,
        providerId,
        files,
        headers,
        maxTokens: model?.maxTokens,
        modelId,
        output,
        prompt: params.prompt,
        routing,
        samplingParams: model?.samplingParams,
        signal,
      };

      signal?.throwIfAborted();
      let answer: Answer | undefined;
      if (openRouterImage) await callOpenRouterImage(request);
      else if (request.api === "anthropic-messages") answer = await callAnthropic(request);
      else if (request.api === "google-generative-ai" || request.api === "google-vertex") {
        answer = await callGoogle(request);
      } else if (request.api === "openai-completions" || request.api === "openai-responses") {
        answer = await callOpenAI(request);
      }

      const text = answer?.text ?? "";
      const status = answer?.status;
      let result = text;
      let fullOutputPath: string | undefined;
      const preview = truncateHead(text);
      if (preview.truncated) {
        const directory = await mkdtemp(path.join(tmpdir(), "pi-ask-"));
        fullOutputPath = path.join(directory, "response.txt");
        await writeFile(fullOutputPath, text, "utf8");
        result = `${preview.content}\n\n[ask: response preview truncated. Full response: ${fullOutputPath}. Use read with offset and limit to read sections.]`;
      }
      if (status) result = result ? `${result}\n\n[ask: ${status}]` : `[ask: ${status}]`;
      return {
        content: [{ type: "text", text: output ? `Image saved to ${params.output}` : result }],
        details: { model: params.model, files: params.files ?? [], output: params.output, status, fullOutputPath },
      };
    },
  });
}
