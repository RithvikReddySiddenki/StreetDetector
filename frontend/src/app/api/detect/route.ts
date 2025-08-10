// frontend/src/app/api/detect/route.ts
import { NextResponse } from "next/server";

export async function POST(req: Request) {
  // Incoming form from the browser
  const incoming = await req.formData();
  const file = incoming.get("file") as File | null;
  const render = incoming.get("render"); // "true" when user wants an image

  if (!file) {
    return NextResponse.json({ error: "file is required" }, { status: 400 });
  }

  // Build the outgoing form that the backend expects (only the file)
  const out = new FormData();
  out.append("file", file);

  const base = process.env.INFERENCE_URL ?? "http://127.0.0.1:8000/predict";
  const url = render ? `${base}?render=true` : base;

  const res = await fetch(url, { method: "POST", body: out });

  // Pipe through bytes + headers exactly as backend returned them
  const buf = Buffer.from(await res.arrayBuffer());
  const headers = new Headers(res.headers);
  return new NextResponse(buf, { status: res.status, headers });
}