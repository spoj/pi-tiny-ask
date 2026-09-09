import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import type { ExtensionAPI, ExtensionContext, ToolDefinition } from "@earendil-works/pi-coding-agent";
import register from "../extensions/ask.ts";

async function fixture(t: TestContext, modelApi = "openai-completions", registered = true, providerId = "custom") {
  const cwd = await mkdtemp(path.join(tmpdir(), "tiny-ask-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  for (const name of ["photo.png", "voice.oga", "voice.OGG", "voice.opus", "voice.mp3", "voice.wav", "voice.m4a", "voice.aac", "report.pdf", "clip.mp4"]) {
    await writeFile(path.join(cwd, name), "media");
  }
  const requests: Array<{ url: string; headers: Headers; body: any }> = [];
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push({ url: request.url, headers: request.headers, body: await request.json() });
    return Response.json({
      object: "response",
      choices: [{ message: { content: "ok" } }],
      output: [{ type: "message", content: [{ type: "output_text", text: "ok", annotations: [] }] }],
      content: [{ type: "text", text: "ok" }],
      candidates: [{ content: { parts: [{ text: "ok" }] } }],
      data: [{ b64_json: Buffer.from("image").toString("base64") }],
    });
  });
  let tool!: ToolDefinition;
  register({ registerTool(definition: ToolDefinition) { tool = definition; } } as ExtensionAPI);
  const model = { api: modelApi, headers: { "x-model": "configured" } };
  const ctx = {
    cwd,
    modelRegistry: {
      find: () => registered ? model : undefined,
      getProvider: (id: string) => id === providerId ? {
        baseUrl: "https://gateway.test/v1",
        getModels: () => registered ? [model] : [],
      } : undefined,
      getProviderAuth: async () => ({ auth: { apiKey: "test-key" } }),
    },
  } as unknown as ExtensionContext;
  return {
    cwd, requests, tool,
    run: (params: Record<string, unknown>) => tool.execute("test", {
      model: `${providerId}/test-model`, prompt: "inspect these", ...params,
    }, undefined, undefined, ctx),
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
