import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface LocationState {
  result: { text: string; confidence: number };
  imageUrl: string;
}

export default function ResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as LocationState | null;

  if (!state) {
    navigate("/");
    return null;
  }

  const { result, imageUrl } = state;
  const confidencePercent = Math.round(result.confidence * 100);

  const confidenceColor =
    confidencePercent >= 80
      ? "text-green-600"
      : confidencePercent >= 50
        ? "text-yellow-600"
        : "text-red-600";

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <CardTitle>Recognition Result</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <p className="text-sm text-muted-foreground mb-2">Original Image</p>
            <img src={imageUrl} alt="Uploaded" className="max-h-24 rounded border" />
          </div>

          <div>
            <p className="text-sm text-muted-foreground mb-1">Recognized Text</p>
            <p className="text-2xl font-mono font-semibold">{result.text || "(empty)"}</p>
          </div>

          <div>
            <p className="text-sm text-muted-foreground mb-1">Confidence</p>
            <p className={`text-lg font-semibold ${confidenceColor}`}>
              {confidencePercent}%
            </p>
          </div>

          <div className="flex gap-3">
            <Button onClick={() => navigate("/")} variant="default">
              Upload Another
            </Button>
            <Button
              onClick={() => navigator.clipboard.writeText(result.text)}
              variant="outline"
            >
              Copy Text
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
