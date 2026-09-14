import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import { InMemoryCredentialStore, InMemoryModelsStore, type Credential } from "@earendil-works/pi-ai";
import { ModelRegistry, ModelRuntime, type ExtensionAPI, type ExtensionContext, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import register from "../extensions/ask.ts";
import { anthropicResponse } from "./anthropic-response.ts";

async function fixture(
  t: TestContext,
  providerId: string,
  credential: Credential = { type: "api_key", key: "test-key" },
  config?: Record<string, unknown>,
) {
  const cwd = await mkdtemp(path.join(tmpdir(), "tiny-ask-providers-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  await writeFile(path.join(cwd, "photo.png"), "image");
  await writeFile(path.join(cwd, "report.pdf"), "pdf");
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(providerId, async () => credential);
  const modelsPath = path.join(cwd, "models.json");
  if (config) await writeFile(modelsPath, JSON.stringify({ providers: { [providerId]: config } }));
  const runtime = await ModelRuntime.create({
    credentials, modelsPath, modelsStore: new InMemoryModelsStore(), refreshOnCreate: false,
  });
  const modelRegistry = new ModelRegistry(runtime);
  let tool!: ToolDefinition;
  register({ registerTool(definition: ToolDefinition) { tool = definition; } } as ExtensionAPI);
  const requests: Array<{ url: string; headers: Headers; body: any }> = [];
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    const request = new Request(input, init);
    const body = await request.json();
    requests.push({ url: request.url, headers: request.headers, body });
    return body.stream ? anthropicResponse({ text: "ok" }) : Response.json({
      choices: [{ message: { content: "ok" }, finish_reason: "stop" }],
      output: [{ type: "message", content: [{ type: "output_text", text: "ok", annotations: [] }] }],
      data: [{ b64_json: Buffer.from("image").toString("base64") }],
    });
  });
  return {
    runtime, modelRegistry, requests,
    run: (modelId: string, params: Record<string, unknown> = {}) => tool.execute("test", {
      model: `${providerId}/${modelId}`, prompt: "inspect these", ...params,
    }, undefined, undefined, { cwd, modelRegistry } as ExtensionContext),
  };
}

for (const modelId of ["claude-sonnet-4-5", "claude-opus-4-6"]) {
  test(`streams the catalog token limit for anthropic/${modelId}`, async (t) => {
    const { run, requests, modelRegistry } = await fixture(t, "anthropic");
    const model = modelRegistry.find("anthropic", modelId)!;
    assert.ok(model.maxTokens >= 64_000);
    const result = await run(modelId, { files: ["photo.png", "report.pdf"] });
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, "https://api.anthropic.com/v1/messages");
    assert.equal(requests[0].body.max_tokens, model.maxTokens);
    assert.equal(requests[0].body.stream, true);
    assert.deepEqual(requests[0].body.messages[0].content.map((block: any) => block.type), ["text", "image", "document"]);
    assert.deepEqual(result.content, [{ type: "text", text: "ok" }]);
  });
}

test("uses resolved Anthropic OAuth as bearer auth with the required identity", async (t) => {
  const token = "sk-ant-oat-test-token";
  const { run, requests } = await fixture(t, "anthropic", {
    type: "oauth", access: token, refresh: "unused", expires: Date.now() + 3_600_000,
  });
  await run("claude-sonnet-4-5", { files: ["photo.png"] });
  assert.equal(requests[0].headers.get("authorization"), `Bearer ${token}`);
  assert.equal(requests[0].headers.get("x-api-key"), null);
  assert.match(requests[0].headers.get("anthropic-beta")!, /oauth-2025-04-20/);
  assert.match(requests[0].headers.get("user-agent")!, /^claude-cli\//);
  assert.equal(requests[0].body.system, "You are Claude Code, Anthropic's official CLI for Claude.");
});

for (const [api, route] of [
  ["anthropic-messages", "anthropic/v1/messages"],
  ["openai-completions", "compat/chat/completions"],
  ["openai-responses", "openai/responses"],
]) {
  test(`preserves Copilot bearer auth and vision headers for ${api}`, async (t) => {
    const { run, requests, runtime } = await fixture(t, "github-copilot");
    const model = runtime.getModels("github-copilot").find((model) => model.api === api)!;
    assert.ok(model);
    await run(model.id, { files: ["photo.png"] });
    assert.equal(requests[0].headers.get("authorization"), "Bearer test-key");
    assert.equal(requests[0].headers.get("x-api-key"), null);
    assert.equal(requests[0].headers.get("copilot-vision-request"), "true");
    assert.equal(requests[0].headers.get("x-initiator"), "agent");
    assert.equal(requests[0].headers.get("editor-version"), model.headers!["Editor-Version"]);
  });

  test(`materializes Cloudflare gateway endpoints and suppresses upstream auth for ${api}`, async (t) => {
    const { run, requests, runtime } = await fixture(t, "cloudflare-ai-gateway", {
      type: "api_key", key: "cloudflare-token",
      env: { CLOUDFLARE_ACCOUNT_ID: "account-123", CLOUDFLARE_GATEWAY_ID: "gateway-123" },
    });
    const model = runtime.getModels("cloudflare-ai-gateway").find((model) => model.api === api)!;
    assert.ok(model);
    await run(model.id, { files: ["photo.png"] });
    assert.equal(requests[0].url, `https://gateway.ai.cloudflare.com/v1/account-123/gateway-123/${route}`);
    assert.equal(requests[0].headers.get("cf-aig-authorization"), "Bearer cloudflare-token");
    assert.equal(requests[0].headers.get("authorization"), null);
    assert.equal(requests[0].headers.get("x-api-key"), null);
  });
}

test("materializes Cloudflare Workers AI endpoints without suppressing bearer auth", async (t) => {
  const { run, requests, runtime } = await fixture(t, "cloudflare-workers-ai", {
    type: "api_key", key: "cloudflare-token", env: { CLOUDFLARE_ACCOUNT_ID: "account-123" },
  });
  const model = runtime.getModels("cloudflare-workers-ai")[0];
  await run(model.id);
  assert.equal(requests[0].url, "https://api.cloudflare.com/client/v4/accounts/account-123/ai/v1/chat/completions");
  assert.equal(requests[0].headers.get("authorization"), "Bearer cloudflare-token");
});

for (const api of ["openai-completions", "openai-responses"]) {
  test(`preserves configured OpenRouter routing for ${api}`, async (t) => {
    const routing = { only: ["google-ai-studio"], allow_fallbacks: false, data_collection: "deny", zdr: true };
    const { run, requests, runtime } = await fixture(t, "openrouter", undefined, {
      compat: { openRouterRouting: routing },
    });
    const model = runtime.getModels("openrouter").find((model) => model.api === "openai-completions")!;
    await run(model.id, { api, files: ["photo.png"] });
    assert.deepEqual(requests[0].body.provider, routing);
    assert.equal(requests[0].body.model, model.id);
  });

  test(`preserves configured OpenAI sampling parameters for ${api}`, async (t) => {
    const { run, requests } = await fixture(t, "openai", undefined, {
      modelOverrides: { "gpt-4o": { samplingParams: { temperature: 0.25, top_p: 0.8 } } },
    });
    await run("gpt-4o", { api });
    assert.equal(requests[0].body.temperature, 0.25);
    assert.equal(requests[0].body.top_p, 0.8);
  });
}

test("preserves OpenRouter routing with an Anthropic Messages override", async (t) => {
  const routing = { only: ["anthropic"], data_collection: "deny", zdr: true };
  const { run, requests, runtime } = await fixture(t, "openrouter", undefined, {
    baseUrl: "https://openrouter.ai/api", compat: { openRouterRouting: routing },
  });
  const model = runtime.getModels("openrouter")[0];
  await run(model.id, { api: "anthropic-messages" });
  assert.equal(requests[0].url, "https://openrouter.ai/api/v1/messages");
  assert.deepEqual(requests[0].body.provider, routing);
});

test("preserves configured Vercel gateway routing", async (t) => {
  const routing = { only: ["bedrock"], order: ["bedrock"] };
  const { run, requests, runtime } = await fixture(t, "vercel-ai-gateway", undefined, {
    compat: { vercelGatewayRouting: routing },
  });
  const model = runtime.getModels("vercel-ai-gateway")[0];
  assert.equal(model.api, "anthropic-messages");
  await run(model.id, { files: ["photo.png", "report.pdf"] });
  assert.equal(requests[0].url, "https://ai-gateway.vercel.sh/v1/messages");
  assert.deepEqual(requests[0].body.providerOptions, { gateway: routing });
});

test("preserves OpenRouter image routing without forwarding text sampling parameters", async (t) => {
  const routing = { only: ["bytedance"], allow_fallbacks: false, options: { bytedance: { seed: 123 } } };
  const { run, requests } = await fixture(t, "openrouter", undefined, {
    api: "openai-completions",
    compat: { openRouterRouting: routing },
    models: [{ id: "bytedance-seed/seedream-4.5", samplingParams: { temperature: 0.25 } }],
  });
  await run("bytedance-seed/seedream-4.5", { output: "image.png" });
  assert.equal(requests[0].url, "https://openrouter.ai/api/v1/images");
  assert.deepEqual(requests[0].body.provider, routing);
  assert.equal(requests[0].body.temperature, undefined);
});

test("rejects overrides that cannot preserve gateway routing before sending media", async (t) => {
  const { run, requests, runtime } = await fixture(t, "openrouter", undefined, {
    compat: { openRouterRouting: { only: ["google-ai-studio"], data_collection: "deny" } },
  });
  const model = runtime.getModels("openrouter")[0];
  for (const api of ["google-generative-ai", "google-vertex"]) {
    await assert.rejects(run(model.id, { api, files: ["missing.pdf"] }), /cannot preserve configured gateway routing/);
  }
  await assert.rejects(run(model.id, { api: "openai-responses", output: "image.png" }), /cannot preserve configured gateway routing/);
  await assert.rejects(run(model.id, { output: "image.png" }), /OpenRouter images do not support configured routing option: data_collection/);
  await assert.rejects(run("unknown-image-model", { output: "image.png" }), /Register .* to resolve its gateway routing/);
  await assert.rejects(run("unknown-text-model", { api: "openai-completions" }), /Register .* to resolve its gateway routing/);
  assert.equal(requests.length, 0);
});

test("allows unregistered OpenRouter image models when no routing is configured", async (t) => {
  const { run, requests } = await fixture(t, "openrouter");
  await run("unknown-image-model", { output: "image.png" });
  assert.equal(requests[0].url, "https://openrouter.ai/api/v1/images");
});

test("rejects sampling parameters that overwrite configured routing", async (t) => {
  const { run, requests } = await fixture(t, "openrouter", undefined, {
    api: "openai-completions",
    compat: { openRouterRouting: { only: ["google-ai-studio"], data_collection: "deny" } },
    models: [{ id: "test-model", samplingParams: { provider: { data_collection: "allow" } } }],
  });
  await assert.rejects(run("test-model"), /Configure gateway routing in compat or samplingParams, not both/);
  assert.equal(requests.length, 0);
});
