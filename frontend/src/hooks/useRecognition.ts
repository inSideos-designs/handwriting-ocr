import { useMutation } from "@tanstack/react-query";
import { recognizePage } from "../services/api";
import type { PageRecognitionResult } from "../services/api";

export function useRecognition() {
  return useMutation<PageRecognitionResult, Error, File>({
    mutationFn: recognizePage,
  });
}
