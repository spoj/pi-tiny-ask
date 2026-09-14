import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import type { ExtensionAPI, ExtensionContext, ToolDefinition } from "@earendil-works/pi-coding-agent";
import register from "../extensions/ask.ts";
import { anthropicResponse } from "./anthropic-response.ts";

type HeaderMap = Record<string, string | null>;
type FixtureOptions = {
  baseUrl?: string;
  providerHeaders?: HeaderMap;
  modelHeaders?: HeaderMap;
  modelOverrideHeaders?: HeaderMap;
  auth?: { apiKey?: string; headers?: HeaderMap };
  authError?: string;
  respond?: (request: Request) => Response | Promise<Response>;
};

function setEnv(t: TestContext, name: string, value: string): void {
  const previous = process.env[name];
  process.env[name] = value;
  t.after(() => {
    if (previous === undefined) delete process.env[name];
    else process.env[name] = previous;
  });
}

function textOf(result: { content: Array<unknown> }): string {
  return (result.content[0] as { text: string }).text;
}

async function fixture(
  t: TestContext,
  modelApi = "openai-completions",
  registered = true,
  providerId = "custom",
  options: FixtureOptions = {},
) {
  const cwd = await mkdtemp(path.join(tmpdir(), "tiny-ask-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  for (const name of ["photo.png", "anim.gif", "voice.oga", "voice.OGG", "voice.opus", "voice.mp3", "voice.wav", "voice.m4a", "voice.aac", "report.pdf", "clip.mp4"]) {
    await writeFile(path.join(cwd, name), "media");
  }
  const requests: Array<{ url: string; headers: Headers; body: any }> = [];
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push({ url: request.url, headers: request.headers, body: await request.json() });
    if (options.respond) return options.respond(request);
    if (requests.at(-1)!.body.stream) return anthropicResponse({ text: "ok" });
    return Response.json({
      object: "response",
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
      output: [{ type: "message", content: [{ type: "output_text", text: "ok", annotations: [] }] }],
      content: [{ type: "text", text: "ok" }],
      candidates: [{ content: { parts: [{ text: "ok" }] } }],
      data: [{ b64_json: Buffer.from("image").toString("base64") }],
    });
  });
  let tool!: ToolDefinition;
  register({ registerTool(definition: ToolDefinition) { tool = definition; } } as ExtensionAPI);
  const baseUrl = options.baseUrl ?? "https://gateway.test/v1";
  const providerHeaders = options.providerHeaders ?? {};
  const modelHeaders = options.modelHeaders ?? { "x-model": "configured" };
  const modelOverrideHeaders = options.modelOverrideHeaders ?? {};
  const apiKey = options.auth ? options.auth.apiKey : "test-key";
  const authHeaders = options.auth?.headers ?? {};
  const model = { api: modelApi, headers: modelHeaders };
  const ctx = {
    cwd,
    modelRegistry: {
      find: () => registered ? model : undefined,
      getProvider: (id: string) => id === providerId ? {
        baseUrl,
        headers: providerHeaders,
        getModels: () => registered ? [model] : [],
      } : undefined,
      getApiKeyAndHeaders: async () => options.authError
        ? { ok: false, error: options.authError }
        : { ok: true, apiKey, headers: { ...providerHeaders, ...authHeaders, ...modelHeaders, ...modelOverrideHeaders }, baseUrl: options.baseUrl },
      getProviderAuth: async () => ({ auth: { apiKey, headers: authHeaders, baseUrl: options.baseUrl } }),
    },
  } as unknown as ExtensionContext;
  return {
    cwd, requests, tool,
    run: (params: Record<string, unknown>, signal?: AbortSignal) => tool.execute("test", {
      model: `${providerId}/test-model`, prompt: "inspect these", ...params,
    }, signal, undefined, ctx),
  };
}

test("uses the registered API and serializes chat image, audio, and PDF inputs", async (t) => {
  const { run, requests, tool } = await fixture(t);
  assert.ok(Reflect.get(tool.parameters, "properties").api.enum.includes("openai-completions"));
  assert.deepEqual(Reflect.get(tool.parameters, "required"), ["model", "prompt"]);
  await run({ files: ["photo.png", "voice.oga", "report.pdf"] });
  assert.equal(requests[0].url, "https://gateway.test/v1/chat/completions");
  assert.equal(requests[0].headers.get("authorization"), "Bearer test-key");
  assert.equal(requests[0].headers.get("x-model"), "configured");
  assert.deepEqual(requests[0].body, {
    model: "test-model",
    messages: [{ role: "user", content: [
      { type: "text", text: "inspect these" },
      { type: "image_url", image_url: { url: "data:image/png;base64,bWVkaWE=" } },
      { type: "input_audio", input_audio: { data: "bWVkaWE=", format: "ogg" } },
      { type: "file", file: { filename: "report.pdf", file_data: "data:application/pdf;base64,bWVkaWE=" } },
    ] }],
  });
});

test("preserves audio formats without transcoding", async (t) => {
  const { run, requests } = await fixture(t);
  await run({ files: ["voice.OGG", "voice.opus", "voice.mp3", "voice.wav", "voice.m4a", "voice.aac"] });
  assert.deepEqual(requests[0].body.messages[0].content.slice(1).map((part: any) => part.input_audio),
    ["ogg", "opus", "mp3", "wav", "m4a", "aac"].map((format) => ({ data: "bWVkaWE=", format })));
});

test("explicit API overrides a registered model without changing its endpoint or ID", async (t) => {
  const { run, requests } = await fixture(t, "openai-responses");
  await run({ api: "openai-completions", files: ["voice.oga"] });
  assert.equal(requests[0].url, "https://gateway.test/v1/chat/completions");
  assert.equal(requests[0].body.model, "test-model");
  assert.equal(requests[0].headers.get("authorization"), "Bearer test-key");
});

test("unregistered models require an explicit API and preserve slashes in the model ID", async (t) => {
  const { run, requests } = await fixture(t, "openai-completions", false);
  await assert.rejects(run({ model: "custom/vendor/new-model" }), /Model not found/);
  assert.equal(requests.length, 0);
  await run({ model: "custom/vendor/new-model", api: "openai-completions" });
  assert.equal(requests[0].body.model, "vendor/new-model");
});

test("rejects unsupported chat video locally and preserves provider errors", async (t) => {
  const { run, requests } = await fixture(t);
  await assert.rejects(run({ files: ["clip.mp4"] }), /ask openai-completions serializer does not support video input/);
  assert.equal(requests.length, 0);
  t.mock.method(globalThis, "fetch", async () => Response.json({
    error: { message: "This model does not support document input" },
  }, { status: 400 }));
  await assert.rejects(run({ files: ["report.pdf"] }), /This model does not support document input/);
});

test("validates media kinds before reading file contents", async (t) => {
  const { run, requests } = await fixture(t);
  await assert.rejects(run({ files: ["missing.mp4"] }), /ask openai-completions serializer does not support video input: missing\.mp4/);
  await assert.rejects(run({ output: "out.png", files: ["missing.pdf"] }), /ask openai-completions serializer does not support document input: missing\.pdf/);
  await assert.rejects(run({ api: "google-generative-ai", output: "out.png", files: ["missing.mp4"] }), /ask google-generative-ai serializer does not support video input: missing\.mp4/);
  assert.equal(requests.length, 0);
});

test("Google serializers accept every supported media kind inline", async (t) => {
  const { run, requests } = await fixture(t);
  await run({ api: "google-generative-ai", files: ["clip.mp4", "voice.oga", "report.pdf", "photo.png"] });
  assert.deepEqual(requests[0].body.contents[0].parts.slice(1), [
    { inlineData: { mimeType: "video/mp4", data: "bWVkaWE=" } },
    { inlineData: { mimeType: "audio/ogg", data: "bWVkaWE=" } },
    { inlineData: { mimeType: "application/pdf", data: "bWVkaWE=" } },
    { inlineData: { mimeType: "image/png", data: "bWVkaWE=" } },
  ]);
});

test("rejects GIF input only for OpenAI image generation", async (t) => {
  const { run, requests } = await fixture(t);
  await run({ files: ["anim.gif"] });
  assert.equal(requests[0].body.messages[0].content[1].image_url.url, "data:image/gif;base64,bWVkaWE=");
  await assert.rejects(run({ output: "out.png", files: ["anim.gif"] }), /OpenAI image generation does not support GIF input: anim\.gif/);
  assert.equal(requests.length, 1);

  const openrouter = await fixture(t, "openai-completions", false, "openrouter");
  await openrouter.run({ output: "out.png", files: ["anim.gif"] });
  assert.equal(openrouter.requests[0].url, "https://gateway.test/v1/images");
});

test("preserves Responses PDF serialization without an override", async (t) => {
  const { run, requests } = await fixture(t, "openai-responses");
  await run({ files: ["report.pdf"] });
  assert.equal(requests[0].url, "https://gateway.test/v1/responses");
  assert.deepEqual(requests[0].body.input[0].content[1], {
    type: "input_file", filename: "report.pdf", file_data: "data:application/pdf;base64,bWVkaWE=",
  });
});

test("explicit Google serializer sends native inline audio", async (t) => {
  const { run, requests } = await fixture(t);
  await run({ api: "google-generative-ai", files: ["voice.oga"] });
  assert.match(requests[0].url, /^https:\/\/gateway\.test\/v1\/models\/test-model:generateContent/);
  assert.deepEqual(requests[0].body.contents[0].parts[1], {
    inlineData: { mimeType: "audio/ogg", data: "bWVkaWE=" },
  });
});

test("explicit API overrides OpenRouter image routing; omission preserves it", async (t) => {
  const { run, requests, cwd } = await fixture(t, "openai-completions", false, "openrouter");
  await run({ output: "default.png" });
  await run({ output: "explicit.png", api: "openai-responses" });
  assert.equal(requests[0].url, "https://gateway.test/v1/images");
  assert.equal(requests[1].url, "https://gateway.test/v1/images/generations");
  assert.equal(await readFile(path.join(cwd, "explicit.png"), "utf8"), "image");
});

test("honors model-scoped auth headers and fails closed on unresolved auth", async (t) => {
  const { run, requests } = await fixture(t, "openai-completions", true, "custom", {
    providerHeaders: { "x-provider": "p", "x-shared": "provider" },
    modelOverrideHeaders: { "x-model-override": "mo", "x-shared": "model" },
  });
  await run({ files: ["photo.png"] });
  assert.equal(requests[0].headers.get("x-provider"), "p");
  assert.equal(requests[0].headers.get("x-model"), "configured");
  assert.equal(requests[0].headers.get("x-model-override"), "mo");
  assert.equal(requests[0].headers.get("x-shared"), "model");

  const failing = await fixture(t, "openai-completions", true, "custom", { authError: 'No API key found for "custom"' });
  await assert.rejects(failing.run({ files: ["photo.png"] }), /No API key found/);
  assert.equal(failing.requests.length, 0);
});

test("does not send ambient Anthropic credentials to header-authenticated providers", async (t) => {
  setEnv(t, "ANTHROPIC_API_KEY", "sk-ant-ambient");
  setEnv(t, "ANTHROPIC_AUTH_TOKEN", "ambient-auth-token");
  const { run, requests } = await fixture(t, "anthropic-messages", true, "custom", {
    auth: { headers: { authorization: "Bearer real-token" } },
    respond: () => anthropicResponse({ text: "ok" }),
  });
  await run({ files: ["photo.png"] });
  assert.notEqual(requests[0].headers.get("x-api-key"), "sk-ant-ambient");
  assert.notEqual(requests[0].headers.get("authorization"), "Bearer ambient-auth-token");
  assert.equal(requests[0].headers.get("authorization"), "Bearer real-token");
});

test("does not send ambient Google credentials to header-authenticated providers", async (t) => {
  setEnv(t, "GEMINI_API_KEY", "ambient-gemini");
  const { run, requests } = await fixture(t, "google-generative-ai", true, "custom", {
    auth: { headers: { "x-proxy-key": "proxy-secret" } },
  });
  await run({ files: ["photo.png"] });
  assert.notEqual(requests[0].headers.get("x-goog-api-key"), "ambient-gemini");
  assert.equal(requests[0].headers.get("x-goog-api-key"), "pi-auth");
  assert.equal(requests[0].headers.get("x-proxy-key"), "proxy-secret");
});

test("preserves Google header overrides without duplicating SDK defaults", async (t) => {
  const { run, requests } = await fixture(t, "google-generative-ai", true, "custom", {
    providerHeaders: { "X-Remove": "stale" },
    auth: { apiKey: "test-key", headers: { "user-agent": "ask-test", "content-type": "application/json", "x-remove": null } },
  });
  await run({});
  assert.equal(requests[0].headers.get("user-agent"), "ask-test");
  assert.equal(requests[0].headers.get("content-type"), "application/json");
  assert.equal(requests[0].headers.get("x-remove"), null);
});

for (const header of ["authorization", "x-goog-api-key", "content-type", "user-agent", "x-goog-api-client"]) {
  test(`rejects unsupported Google SDK header removal: ${header}`, async (t) => {
    const { run, requests } = await fixture(t, "google-generative-ai", true, "custom", {
      auth: { apiKey: "test-key", headers: { [header]: null } },
    });
    await assert.rejects(run({}), /Google SDK cannot suppress configured header/);
    assert.equal(requests.length, 0);
  });
}

test("keeps non-Vertex Google serialization when ambient vertex flags are set", async (t) => {
  setEnv(t, "GOOGLE_GENAI_USE_VERTEXAI", "true");
  const { run, requests } = await fixture(t, "google-generative-ai");
  await run({ files: ["photo.png"] });
  assert.match(requests[0].url, /^https:\/\/gateway\.test\/v1\/models\/test-model:generateContent/);
});

test("preserves a resolved authorization header for OpenRouter image requests", async (t) => {
  const { run, requests, cwd } = await fixture(t, "openai-completions", false, "openrouter", {
    auth: { headers: { authorization: "Bearer oauth-style" } },
  });
  await run({ output: "image.png" });
  assert.equal(requests[0].headers.get("authorization"), "Bearer oauth-style");
  assert.equal(await readFile(path.join(cwd, "image.png"), "utf8"), "image");
});

test("preserves an explicit OpenRouter authorization header when a key is also resolved", async (t) => {
  const { run, requests, cwd } = await fixture(t, "openai-completions", false, "openrouter", {
    auth: { apiKey: "resolved-key", headers: { authorization: "Bearer explicit-header" } },
  });
  await run({ output: "image.png" });
  assert.equal(requests[0].headers.get("authorization"), "Bearer explicit-header");
  assert.equal(await readFile(path.join(cwd, "image.png"), "utf8"), "image");
});

for (const api of ["anthropic-messages", "openai-completions", "openai-responses"]) {
  test(`preserves case-insensitive auth header removals for ${api}`, async (t) => {
    const { run, requests } = await fixture(t, api, true, "custom", {
      providerHeaders: { Authorization: "Bearer stale", "X-API-Key": "stale" },
      auth: { apiKey: "resolved-key", headers: { authorization: null, "x-api-key": null, "x-proxy-key": "proxy-key" } },
    });
    await run({});
    assert.equal(requests[0].headers.get("authorization"), null);
    assert.equal(requests[0].headers.get("x-api-key"), null);
    assert.equal(requests[0].headers.get("x-proxy-key"), "proxy-key");
  });
}

test("preserves explicit authorization removal for OpenRouter images", async (t) => {
  const { run, requests } = await fixture(t, "openai-completions", false, "openrouter", {
    providerHeaders: { Authorization: "Bearer stale" },
    auth: { apiKey: "resolved-key", headers: { authorization: null, "x-proxy-key": "proxy-key" } },
  });
  await run({ output: "image.png" });
  assert.equal(requests[0].headers.get("authorization"), null);
  assert.equal(requests[0].headers.get("x-proxy-key"), "proxy-key");
});

test("rejects Azure and Codex transports but keeps them out of the API enum", async (t) => {
  const codex = await fixture(t, "openai-codex-responses");
  assert.ok(!Reflect.get(codex.tool.parameters, "properties").api.enum.includes("openai-codex-responses"));
  await assert.rejects(codex.run({}), /ask does not support the openai-codex-responses transport/);
  assert.equal(codex.requests.length, 0);

  const azure = await fixture(t, "azure-openai-responses");
  assert.ok(!Reflect.get(azure.tool.parameters, "properties").api.enum.includes("azure-openai-responses"));
  await assert.rejects(azure.run({}), /ask does not support the azure-openai-responses transport/);
  assert.equal(azure.requests.length, 0);
});

test("requires an endpoint for every non-Vertex serializer", async (t) => {
  const overridden = await fixture(t, "openai-completions", true, "azure", { baseUrl: "" });
  await assert.rejects(overridden.run({ api: "anthropic-messages" }), /Provider has no configured endpoint: azure/);
  assert.equal(overridden.requests.length, 0);

  const vertexOverride = await fixture(t, "azure-openai-responses", true, "azure", { baseUrl: "" });
  await assert.rejects(vertexOverride.run({ api: "google-vertex" }), /Provider has no configured endpoint: azure/);
  assert.equal(vertexOverride.requests.length, 0);

  const configured = await fixture(t, "openai-completions", true, "custom", { baseUrl: "" });
  await assert.rejects(configured.run({ files: ["photo.png"] }), /Provider has no configured endpoint: custom/);
  assert.equal(configured.requests.length, 0);
});

test("uses a built-in-style configured endpoint when no override is given", async (t) => {
  const { run, requests } = await fixture(t, "openai-completions", true, "openai", { baseUrl: "https://api.openai.com/v1" });
  await run({ files: ["photo.png"] });
  assert.equal(requests[0].url, "https://api.openai.com/v1/chat/completions");
});

test("allows header-authenticated Vertex gateways with a configured endpoint", async (t) => {
  setEnv(t, "GOOGLE_CLOUD_PROJECT", "");
  setEnv(t, "GOOGLE_CLOUD_LOCATION", "");
  setEnv(t, "GOOGLE_APPLICATION_CREDENTIALS", "");
  const { run, requests } = await fixture(t, "google-vertex", true, "custom", {
    auth: { headers: { "x-proxy-key": "proxy-secret" } },
  });
  await run({ files: ["photo.png"] });
  assert.equal(requests[0].headers.get("x-proxy-key"), "proxy-secret");
  assert.equal(requests[0].headers.get("x-goog-api-key"), "pi-auth");
  assert.match(requests[0].url, /^https:\/\/gateway\.test\/v1\/.*generateContent/);
});

test("allows Vertex native endpoint derivation without a configured endpoint", async (t) => {
  const { run, requests } = await fixture(t, "google-vertex", true, "google-vertex", { baseUrl: "" });
  const result = await run({ files: ["photo.png"] });
  assert.equal(textOf(result), "ok");
  assert.match(requests[0].url, /^https:\/\/aiplatform\.googleapis\.com\//);
});

for (const [name, answer] of [
  ["line limit", Array.from({ length: 2001 }, (_, i) => `line ${i + 1}`).join("\n")],
  ["UTF-8 byte limit", Array.from({ length: 100 }, () => "終".repeat(200)).join("\n")],
]) {
  test(`spills the complete response when its preview exceeds the ${name}`, async (t) => {
    const { run, requests } = await fixture(t, "openai-completions", true, "custom", {
      respond: () => Response.json({ choices: [{ message: { content: answer }, finish_reason: "length" }] }),
    });
    const result = await run({});
    const { fullOutputPath } = result.details as { fullOutputPath: string };
    t.after(() => rm(path.dirname(fullOutputPath), { recursive: true, force: true }));
    assert.equal(await readFile(fullOutputPath, "utf8"), answer);
    const preview = textOf(result).split("\n\n[ask: response preview truncated.")[0];
    assert.ok(answer.startsWith(preview));
    assert.ok(Buffer.byteLength(preview) <= 50 * 1024);
    assert.ok(preview.split("\n").length <= 2000);
    assert.match(textOf(result), /Use read with offset and limit/);
    assert.ok(textOf(result).includes(fullOutputPath));
    assert.match(textOf(result), /\[ask: response truncated \(length\)\]$/);
    assert.equal(requests[0].body.max_tokens, undefined);
    assert.equal(requests[0].body.max_completion_tokens, undefined);
  });
}

for (const [name, answer] of [
  ["short", "complete answer"],
  ["exact byte limit", "x".repeat(50 * 1024)],
  ["exact line limit", Array.from({ length: 2000 }, () => "line").join("\n")],
]) {
  test(`returns a ${name} response without spilling`, async (t) => {
    const { run } = await fixture(t, "openai-completions", true, "custom", {
      respond: () => Response.json({ choices: [{ message: { content: answer }, finish_reason: "stop" }] }),
    });
    const result = await run({});
    assert.equal(textOf(result), answer);
    assert.equal((result.details as { fullOutputPath?: string }).fullOutputPath, undefined);
  });
}

test("reports truncated and refused provider responses with an explicit status", async (t) => {
  const chat = await fixture(t, "openai-completions", true, "custom", {
    respond: () => Response.json({ choices: [{ message: { content: "partial" }, finish_reason: "length" }] }),
  });
  const chatResult = await chat.run({ files: ["photo.png"] });
  assert.match(textOf(chatResult), /partial/);
  assert.match(textOf(chatResult), /\[ask: response truncated \(length\)\]/);
  assert.equal((chatResult.details as { status?: string }).status, "response truncated (length)");

  const responses = await fixture(t, "openai-responses", true, "custom", {
    respond: () => Response.json({
      id: "resp_1", object: "response", created_at: 0, status: "incomplete",
      incomplete_details: { reason: "max_output_tokens" }, error: null, model: "test-model",
      output: [{
        type: "message", id: "msg_1", role: "assistant", status: "incomplete",
        content: [{ type: "output_text", text: "partial", annotations: [] }],
      }],
    }),
  });
  const responsesResult = await responses.run({ files: ["report.pdf"] });
  assert.match(textOf(responsesResult), /partial/);
  assert.match(textOf(responsesResult), /\[ask: response incomplete \(max_output_tokens\)\]/);

  const anthropic = await fixture(t, "anthropic-messages", true, "custom", {
    respond: () => anthropicResponse({ text: "partial", stopReason: "max_tokens" }),
  });
  const anthropicResult = await anthropic.run({ files: ["photo.png"] });
  assert.match(textOf(anthropicResult), /\[ask: response truncated \(max_tokens\)\]/);

  const google = await fixture(t, "google-generative-ai", true, "custom", {
    respond: () => Response.json({
      candidates: [{ content: { role: "model", parts: [{ text: "partial" }] }, finishReason: "MAX_TOKENS" }],
    }),
  });
  const googleResult = await google.run({ files: ["photo.png"] });
  assert.match(textOf(googleResult), /\[ask: response truncated \(MAX_TOKENS\)\]/);
});

test("reports explicit model refusals", async (t) => {
  const chat = await fixture(t, "openai-completions", true, "custom", {
    respond: () => Response.json({ choices: [{ message: { content: null, refusal: "I can't help with that." }, finish_reason: "stop" }] }),
  });
  const chatResult = await chat.run({ files: ["photo.png"] });
  assert.match(textOf(chatResult), /I can't help with that\./);
  assert.match(textOf(chatResult), /\[ask: model refused the request\]/);

  const responses = await fixture(t, "openai-responses", true, "custom", {
    respond: () => Response.json({
      id: "resp_1", object: "response", created_at: 0, status: "completed",
      incomplete_details: null, output_text: "", error: null, model: "test-model",
      output: [{
        type: "message", id: "msg_1", role: "assistant", status: "completed",
        content: [{ type: "refusal", refusal: "I can't help with that." }],
      }],
    }),
  });
  const responsesResult = await responses.run({ files: ["report.pdf"] });
  assert.match(textOf(responsesResult), /I can't help with that\./);
  assert.match(textOf(responsesResult), /\[ask: model refused the request\]/);

  const anthropic = await fixture(t, "anthropic-messages", true, "custom", {
    respond: () => anthropicResponse({ stopReason: "refusal" }),
  });
  const anthropicResult = await anthropic.run({ files: ["photo.png"] });
  assert.equal(textOf(anthropicResult), "[ask: model refused the request]");
});

test("does not retry Google SDK requests", async (t) => {
  const google = await fixture(t, "google-generative-ai", true, "custom", {
    respond: () => Response.json({ error: { message: "server error" } }, { status: 500 }),
  });
  await assert.rejects(google.run({ files: ["photo.png"] }));
  assert.equal(google.requests.length, 1);
});

test("does not retry SDK requests and honors a pre-aborted signal", async (t) => {
  const openai = await fixture(t, "openai-completions", true, "custom", {
    respond: () => Response.json({ error: { message: "server error" } }, { status: 500 }),
  });
  await assert.rejects(openai.run({ files: ["photo.png"] }));
  assert.equal(openai.requests.length, 1);

  const anthropic = await fixture(t, "anthropic-messages", true, "custom", {
    respond: () => Response.json({ type: "error", error: { type: "api_error", message: "server error" } }, { status: 500 }),
  });
  await assert.rejects(anthropic.run({ files: ["photo.png"] }));
  assert.equal(anthropic.requests.length, 1);

  const aborted = await fixture(t, "google-generative-ai");
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(aborted.run({ api: "google-generative-ai", files: ["voice.oga"] }, controller.signal), /abort/i);
  assert.equal(aborted.requests.length, 0);
});
