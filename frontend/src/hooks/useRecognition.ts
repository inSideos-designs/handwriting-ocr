import { useMutation } from "@tanstack/react-query";
import { recognizeImage } from "../services/api";
import type { RecognitionResult } from "../services/api";

export function useRecognition() {
  return useMutation<RecognitionResult, Error, File>({
    mutationFn: recognizeImage,
  });
}
