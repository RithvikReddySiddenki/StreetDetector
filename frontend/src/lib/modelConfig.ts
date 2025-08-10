export const MODEL_URL =
  process.env.NEXT_PUBLIC_ONNX_MODEL_URL?.trim() ||
  // Default: serve the model from Next.js /public in both local and Vercel
  "/yolov5m.onnx";