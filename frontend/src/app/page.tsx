"use client";

import { useEffect, useRef, useState } from "react";
import type { Tensor, InferenceSession } from "onnxruntime-web";

type Det = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  score: number;
};

// -------- YOLO/ONNX helpers --------
const MODEL_INPUT = { size: 640, name: "images" }; // YOLOv5 default
const CLASS_NAMES = [
  "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
  "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
  "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
  "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
  "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
  "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
  "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard",
  "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase",
  "scissors","teddy bear","hair drier","toothbrush",
];

// NMS (very simple, good enough for demo)
function nms(boxes: Det[], iouThresh = 0.45): Det[] {
  const picked: Det[] = [];
  boxes
    .sort((a, b) => b.score - a.score)
    .forEach((cand) => {
      const keep = picked.every((p) => {
        const xx1 = Math.max(cand.x1, p.x1);
        const yy1 = Math.max(cand.y1, p.y1);
        const xx2 = Math.min(cand.x2, p.x2);
        const yy2 = Math.min(cand.y2, p.y2);
        const inter = Math.max(0, xx2 - xx1) * Math.max(0, yy2 - yy1);
        const areaA = (cand.x2 - cand.x1) * (cand.y2 - cand.y1);
        const areaB = (p.x2 - p.x1) * (p.y2 - p.y1);
        const iou = inter / (areaA + areaB - inter + 1e-6);
        return iou < iouThresh;
      });
      if (keep) picked.push(cand);
    });
  return picked;
}

// letterbox -> resize with aspect ratio + pad to 640x640
function letterbox(
  img: HTMLImageElement,
  dst: HTMLCanvasElement,
  size = MODEL_INPUT.size
) {
  const dw = dst.width = size;
  const dh = dst.height = size;
  const ctx = dst.getContext("2d")!;
  ctx.clearRect(0, 0, dw, dh);
  ctx.fillStyle = "black";
  ctx.fillRect(0, 0, dw, dh);

  const ar = img.width / img.height;
  const s = ar > 1 ? size / img.width : size / img.height;
  const nw = Math.round(img.width * s);
  const nh = Math.round(img.height * s);
  const dx = Math.floor((size - nw) / 2);
  const dy = Math.floor((size - nh) / 2);

  ctx.drawImage(img, 0, 0, img.width, img.height, dx, dy, nw, nh);

  // return scale + pad for mapping boxes back
  return { scale: s, padX: dx, padY: dy, netW: nw, netH: nh };
}

function chwFromCanvas(cv: HTMLCanvasElement): Float32Array {
  const { width, height } = cv;
  const ctx = cv.getContext("2d")!;
  const { data } = ctx.getImageData(0, 0, width, height); // RGBA
  const out = new Float32Array(3 * width * height);
  let pi = 0;
  const planeSize = width * height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      const r = data[idx] / 255;
      const g = data[idx + 1] / 255;
      const b = data[idx + 2] / 255;
      out[pi] = r;
      out[pi + planeSize] = g;
      out[pi + 2 * planeSize] = b;
      pi++;
    }
  }
  return out;
}

let _sessionPromise: Promise<InferenceSession> | null = null;
async function getSession(modelUrl: string): Promise<InferenceSession> {
  if (_sessionPromise) return _sessionPromise;
  _sessionPromise = (async () => {
    const ort = await import("onnxruntime-web");
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return session;
  })();
  return _sessionPromise;
}

function getModelUrl() {
  // Works locally and on Vercel
  return `${typeof window !== "undefined" ? window.location.origin : ""}/yolov5m.onnx`;
}

// -------- UI / Page --------
export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "done" | "error">(
    "idle"
  );
  const [heading, setHeading] = useState<"Preview" | "Detections">("Preview");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const imgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const workRef = useRef<HTMLCanvasElement | null>(null); // for preprocessing

  // Load selected file -> draw to canvas as preview
  useEffect(() => {
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;

      // draw original image full-bleed into the main canvas (large)
      const c = canvasRef.current!;
      const ctx = c.getContext("2d")!;
      const maxW = 1280;
      const scale = Math.min(maxW / img.width, 1);
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      ctx.clearRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);

      setHeading("Preview");
      setStatus("idle");
      setErrorMsg(null);
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      setErrorMsg("Failed to load image.");
    };
    img.src = url;
  }, [file]);

  async function onDetect() {
    try {
      if (!imgRef.current || !canvasRef.current) return;
      setStatus("loading");
      setHeading("Detections");
      setErrorMsg(null);

      // 1) Prep working canvas 640x640 with letterbox
      if (!workRef.current) workRef.current = document.createElement("canvas");
      const work = workRef.current;
      const { scale, padX, padY } = letterbox(imgRef.current, work, MODEL_INPUT.size);

      // 2) CHW float32 input
      const chw = chwFromCanvas(work);

      // 3) ONNX session
      const session = await getSession(getModelUrl());
      const ort = await import("onnxruntime-web");

      const feeds: Record<string, Tensor> = {};
      feeds[MODEL_INPUT.name] = new ort.Tensor("float32", chw, [
        1,
        3,
        MODEL_INPUT.size,
        MODEL_INPUT.size,
      ]);

      // 4) Run
      const output = await session.run(feeds);
      const firstKey = Object.keys(output)[0];
      const out = output[firstKey] as Tensor; // [1, 25200, 85]
      const pred = out.data as Float32Array;

      // 5) Parse
      const numDet = out.dims[1];
      const stride = out.dims[2];
      const W = imgRef.current.width;
      const H = imgRef.current.height;

      const dets: Det[] = [];
      for (let i = 0; i < numDet; i++) {
        const base = i * stride;
        const x = pred[base + 0];
        const y = pred[base + 1];
        const w = pred[base + 2];
        const h = pred[base + 3];
        const obj = pred[base + 4];

        // best class
        let best = 0;
        let bestIdx = -1;
        for (let c = 5; c < stride; c++) {
          const sc = pred[base + c];
          if (sc > best) {
            best = sc;
            bestIdx = c - 5;
          }
        }
        const score = obj * best;
        if (score < 0.35) continue;

        // xywh -> xyxy in letterboxed space
        let x1 = x - w / 2;
        let y1 = y - h / 2;
        let x2 = x + w / 2;
        let y2 = y + h / 2;

        // map back to original image space (remove padding, divide by scale)
        x1 = (x1 - padX) / scale;
        y1 = (y1 - padY) / scale;
        x2 = (x2 - padX) / scale;
        y2 = (y2 - padY) / scale;

        // clamp
        x1 = Math.max(0, Math.min(W, x1));
        y1 = Math.max(0, Math.min(H, y1));
        x2 = Math.max(0, Math.min(W, x2));
        y2 = Math.max(0, Math.min(H, y2));

        dets.push({
          x1,
          y1,
          x2,
          y2,
          label: CLASS_NAMES[bestIdx] ?? `cls_${bestIdx}`,
          score,
        });
      }

      const final = nms(dets, 0.45);

      // 6) Draw detections on original-sized canvas (keep original image colors)
      const c = canvasRef.current;
      const ctx = c.getContext("2d")!;
      // redraw original at the same display scale we used in preview
      const maxW = 1280;
      const dispScale = Math.min(maxW / imgRef.current.width, 1);
      c.width = Math.round(imgRef.current.width * dispScale);
      c.height = Math.round(imgRef.current.height * dispScale);
      ctx.drawImage(imgRef.current, 0, 0, c.width, c.height);

      ctx.lineWidth = Math.max(2, Math.round(2 * dispScale));
      ctx.font = `${Math.max(12, Math.round(14 * dispScale))}px ui-sans-serif, system-ui, -apple-system`;
      ctx.textBaseline = "top";

      final.forEach((d) => {
        const x1 = d.x1 * dispScale;
        const y1 = d.y1 * dispScale;
        const w = (d.x2 - d.x1) * dispScale;
        const h = (d.y2 - d.y1) * dispScale;

        // box
        ctx.strokeStyle = "#22c55e"; // green
        ctx.fillStyle = "rgba(34,197,94,0.15)";
        ctx.beginPath();
        ctx.rect(x1, y1, w, h);
        ctx.fill();
        ctx.stroke();

        // label bg
        const label = `${d.label} ${(d.score * 100).toFixed(1)}%`;
        const pad = 4 * dispScale;
        const textW = ctx.measureText(label).width + pad * 2;
        const textH = 18 * dispScale;
        ctx.fillStyle = "#22c55e";
        ctx.fillRect(x1, Math.max(0, y1 - textH), textW, textH);

        // label text
        ctx.fillStyle = "#0a0a0a";
        ctx.fillText(label, x1 + pad, Math.max(0, y1 - textH) + pad / 2);
      });

      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
      setErrorMsg("Detection failed. (Check console for details)");
    }
  }

  return (
    <main className="min-h-screen bg-neutral-950 text-white">
      <div className="mx-auto w-full max-w-screen-xl px-6 py-10 space-y-8">
        <header className="space-y-2">
          <h1 className="text-3xl font-semibold">StreetDetector</h1>
          <p className="text-neutral-300">
            Upload an image 
          </p>
        </header>

        {/* Controls */}
        <div className="flex items-center gap-4">
          <label className="inline-flex cursor-pointer rounded-xl bg-neutral-800 hover:bg-neutral-700 px-4 py-2">
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            Choose image
          </label>
          <button
            onClick={onDetect}
            disabled={!file || status === "loading"}
            className="rounded-xl bg-blue-600 disabled:bg-blue-900 hover:bg-blue-500 px-4 py-2"
          >
            {status === "loading" ? "Detecting..." : "detect"}
          </button>

          {status === "error" && (
            <span className="text-red-400">Backend error: 500</span>
          )}
          {errorMsg && <span className="text-red-400">{errorMsg}</span>}
        </div>

        {/* Canvas area */}
        <section className="space-y-3">
          <h2 className="text-xl font-medium">{heading}</h2>
          <div className="rounded-2xl bg-neutral-900 p-4 shadow-lg">
            <canvas
              ref={canvasRef}
              className="w-full h-auto max-w-[1280px] mx-auto rounded-xl block"
            />
          </div>
          <p className="text-sm text-neutral-400">
            
          </p>
        </section>

        {/* Footer credit */}
        <footer className="pt-8 text-center text-neutral-400">
          Built by <span className="text-neutral-200">Rithvik Reddy Siddenki</span>
        </footer>
      </div>
    </main>
  );
}
