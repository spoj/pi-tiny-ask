export function responsesStream({
  output = [{ type: "message", content: [{ type: "output_text", text: "ok", annotations: [] }] }],
  ...response
}: { output?: unknown[]; [key: string]: unknown } = {}) {
  const final = {
    id: "resp_1", object: "response", created_at: 0, model: "test-model", status: "completed",
    incomplete_details: null, error: null, ...response, output: [],
  };
  // Like Sign in with ChatGPT: output only in item events, and an event the SDK does not know.
  const events = [
    { type: "response.created", response: { ...final, status: "in_progress" } },
    ...output.map((item) => ({ type: "response.output_item.done", item })),
    { type: "response.metadata", metadata: {} },
    { type: `response.${final.status}`, response: final },
  ];
  return new Response(events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), {
    headers: { "content-type": "text/event-stream" },
  });
}
