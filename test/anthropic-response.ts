export function anthropicResponse(options: { text?: string; stopReason?: string } = {}) {
  const events = [
    { type: "message_start", message: {
      id: "msg_1", type: "message", role: "assistant", model: "test-model",
      content: [], stop_reason: null, stop_sequence: null,
      usage: { input_tokens: 1, output_tokens: 0 },
    } },
    ...(options.text ? [
      { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
      { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: options.text } },
      { type: "content_block_stop", index: 0 },
    ] : []),
    { type: "message_delta", delta: { stop_reason: options.stopReason ?? "end_turn", stop_sequence: null }, usage: { output_tokens: 1 } },
    { type: "message_stop" },
  ];
  return new Response(events.map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`).join(""), {
    headers: { "content-type": "text/event-stream" },
  });
}
