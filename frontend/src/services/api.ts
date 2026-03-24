import axios from "axios";

const api = axios.create({
  baseURL: "/api",
});

export interface RecognitionResult {
  text: string;
  confidence: number;
}

export async function recognizeImage(
  file: File
): Promise<RecognitionResult> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await api.post<RecognitionResult>("/recognize", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return response.data;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await api.get("/health");
    return response.data.status === "healthy";
  } catch {
    return false;
  }
}
