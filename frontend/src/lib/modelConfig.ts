export const MODEL_URL =
  process.env.NEXT_PUBLIC_MODEL_URL?.trim() ||
  // Served from /public on Vercel
  "/yolov5m.onnx";
