import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import { ModelRegistry, ModelRuntime, type ExtensionAPI, type ExtensionContext, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import register from "../extensions/ask.ts";

async function fixture(t: TestContext, providerId: string, modelId: string) {
  const cwd = await mkdtemp(path.join(tmpdir(), "tiny-ask-images-"));
  t.after(() => rm(cwd, { recursive: true, force: true }));
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(providerId, async () => ({ type: "api_key", key: "test-key" }));
  const runtime = await ModelRuntime.create({
    credentials, modelsPath: null, modelsStore: new InMemoryModelsStore(), refreshOnCreate: false,
  });
  let tool!: ToolDefinition;
  register({ registerTool(definition: ToolDefinition) { tool = definition; } } as ExtensionAPI);
  const ctx = { cwd, modelRegistry: new ModelRegistry(runtime) } as ExtensionContext;
  return {
    cwd,
    run: (params: Record<string, unknown>) => tool.execute("test", {
      model: `${providerId}/${modelId}`,
      prompt: "paint a tiny robot",
      ...params,
    }, undefined, undefined, ctx),
  };
}

test("Google image generation saves inline output", async (t) => {
  const image = Buffer.from("google-image");
  const { cwd, run } = await fixture(t, "google", "gemini-3.1-flash-image");
  let request!: Request;
  let body: any;
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    request = new Request(input, init);
    body = await request.clone().json();
    return Response.json({
      candidates: [{
        content: { role: "model", parts: [{ inlineData: { mimeType: "image/png", data: image.toString("base64") } }] },
        finishReason: "STOP",
      }],
    });
  });

  const result = await run({ output: "generated/google.png" });

  assert.match(request.url, /^https:\/\/generativelanguage\.googleapis\.com\/v1beta\/models\/gemini-3\.1-flash-image:generateContent/);
  assert.equal(request.headers.get("x-goog-api-key"), "test-key");
  assert.deepEqual(body.generationConfig.responseModalities, ["TEXT", "IMAGE"]);
  assert.deepEqual(await readFile(path.join(cwd, "generated/google.png")), image);
  assert.equal((result.content[0] as { text: string }).text, "Image saved to generated/google.png");
});

test("Google Vertex image generation uses API-key auth and saves inline output", async (t) => {
  const image = Buffer.from("vertex-image");
  const { cwd, run } = await fixture(t, "google-vertex", "gemini-3.1-flash-image");
  let request!: Request;
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    request = new Request(input, init);
    return Response.json({
      candidates: [{
        content: { role: "model", parts: [{ inlineData: { mimeType: "image/png", data: image.toString("base64") } }] },
        finishReason: "STOP",
      }],
    });
  });

  await run({ output: "vertex.png" });

  assert.match(request.url, /^https:\/\/aiplatform\.googleapis\.com\/v1\/publishers\/google\/models\/gemini-3\.1-flash-image:generateContent/);
  assert.equal(request.headers.get("x-goog-api-key"), "test-key");
  assert.equal(request.headers.get("authorization"), null);
  assert.deepEqual(await readFile(path.join(cwd, "vertex.png")), image);
});

test("OpenAI reference-image edits use multipart FormData", async (t) => {
  const reference = Buffer.from("reference-image");
  const generated = Buffer.from("edited-image");
  const { cwd, run } = await fixture(t, "openai", "gpt-image-2");
  await writeFile(path.join(cwd, "reference.png"), reference);
  let request!: Request;
  let sentFormData = false;
  let form!: FormData;
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    if (input === "data:,") return new Response();
    sentFormData = init?.body instanceof FormData;
    request = new Request(input, init);
    form = await request.clone().formData();
    return Response.json({ data: [{ b64_json: generated.toString("base64") }] });
  });

  await run({ files: ["reference.png"], output: "edited.png" });

  assert.equal(request.url, "https://api.openai.com/v1/images/edits");
  assert.equal(sentFormData, true);
  assert.match(request.headers.get("content-type") ?? "", /^multipart\/form-data; boundary=/);
  assert.equal(form.get("model"), "gpt-image-2");
  assert.equal(form.get("prompt"), "paint a tiny robot");
  const uploaded = form.get("image[]");
  assert.ok(uploaded instanceof File);
  assert.equal(uploaded.name, "reference.png");
  assert.equal(uploaded.type, "image/png");
  assert.deepEqual(Buffer.from(await uploaded.arrayBuffer()), reference);
  assert.deepEqual(await readFile(path.join(cwd, "edited.png")), generated);
});

test("OpenAI downloads URL image responses", async (t) => {
  const image = Buffer.from("downloaded-image");
  const imageUrl = "https://images.test/generated.png";
  const { cwd, run } = await fixture(t, "openai", "gpt-image-2");
  const requests: Request[] = [];
  t.mock.method(globalThis, "fetch", async (input: string | Request, init?: RequestInit) => {
    const request = new Request(input, init);
    requests.push(request);
    return request.url === imageUrl
      ? new Response(image, { headers: { "content-type": "image/png" } })
      : Response.json({ data: [{ url: imageUrl }] });
  });

  await run({ output: "downloaded.png" });

  assert.deepEqual(requests.map((request) => request.url), [
    "https://api.openai.com/v1/images/generations",
    imageUrl,
  ]);
  assert.equal(requests[1].method, "GET");
  assert.equal(requests[1].headers.get("authorization"), null);
  assert.deepEqual(await readFile(path.join(cwd, "downloaded.png")), image);
});
