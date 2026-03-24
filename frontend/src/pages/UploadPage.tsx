import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRecognition } from "../hooks/useRecognition";

export default function UploadPage() {
  const navigate = useNavigate();
  const recognition = useRecognition();
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((accepted: File[]) => {
    if (accepted.length === 0) return;
    const file = accepted[0];
    setPreview(URL.createObjectURL(file));

    recognition.mutate(file, {
      onSuccess: (data) => {
        navigate("/result", { state: { result: data, imageUrl: URL.createObjectURL(file) } });
      },
    });
  }, [recognition, navigate]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "image/*": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024,
  });

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <CardTitle>Handwriting Recognition</CardTitle>
        </CardHeader>
        <CardContent>
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors
              ${isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25 hover:border-primary/50"}`}
          >
            <input {...getInputProps()} />
            {recognition.isPending ? (
              <p className="text-muted-foreground">Processing...</p>
            ) : isDragActive ? (
              <p className="text-primary">Drop the image here</p>
            ) : (
              <div>
                <p className="text-muted-foreground mb-2">
                  Drag and drop a handwritten image, or click to select
                </p>
                <Button variant="secondary">Select File</Button>
              </div>
            )}
          </div>

          {preview && (
            <div className="mt-4">
              <img src={preview} alt="Preview" className="max-h-32 mx-auto rounded" />
            </div>
          )}

          {recognition.isError && (
            <p className="text-destructive mt-4 text-sm">
              Error: {recognition.error.message}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
