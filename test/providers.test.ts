import assert from "node:assert/strict";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import { InMemoryCredentialStore, InMemoryModelsStore, type Credential } from "@earendil-works/pi-ai";
import { ModelRegistry, ModelRuntime, type ExtensionAPI, type ExtensionContext, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import register from "../extensions/ask.ts";
import { anthropicResponse } from "./anthropic-response.ts";

async function fixture(t: TestContext, providerId: string, credential: Credential = { type: "api_key", key: "test-key" }) {
  const cwd = await mkdtemp(path.join(tmpdir(), "tiny-ask-providers-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  await writeFile(path.join(cwd, "photo.png"), "image");
  await writeFile(path.join(cwd, "report.pdf"), "pdf");
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(providerId, async () => credential);
  const runtime = await ModelRuntime.create({
    credentials, modelsPath: null, modelsStore: new InMemoryModelsStore(), refreshOnCreate: false,
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
