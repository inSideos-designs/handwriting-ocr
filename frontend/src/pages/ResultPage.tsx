import { useLocation, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface WordResult {
  text: string;
  confidence: number;
}

interface LineResult {
  text: string;
  confidence: number;
  words: WordResult[];
}

interface LocationState {
  result: {
    text: string;
    lines: LineResult[];
    num_lines: number;
    num_words: number;
  };
  imageUrl: string;
}

function ConfidenceBadge({ confidence }: { confidence: number }) {
  const pct = Math.round(confidence * 100);
  const color =
    pct >= 80
      ? "bg-green-100 text-green-800"
      : pct >= 50
        ? "bg-yellow-100 text-yellow-800"
        : "bg-red-100 text-red-800";

  return (
    <span className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${color}`}>
      {pct}%
    </span>
  );
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

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-4xl mx-auto space-y-4">
        <Card>
          <CardHeader>
            <CardTitle>Recognition Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <p className="text-sm text-muted-foreground mb-2">Original Image</p>
                <img src={imageUrl} alt="Uploaded" className="max-h-64 rounded border" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground mb-2">Recognized Text</p>
                <pre className="text-lg font-mono bg-muted p-4 rounded whitespace-pre-wrap">
                  {result.text || "(no text detected)"}
                </pre>
                <p className="text-sm text-muted-foreground mt-2">
                  {result.num_lines} line{result.num_lines !== 1 ? "s" : ""}, {result.num_words} word{result.num_words !== 1 ? "s" : ""}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {result.lines.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Line-by-Line Breakdown</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {result.lines.map((line, i) => (
                <div key={i} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-muted-foreground">Line {i + 1}</span>
                    <ConfidenceBadge confidence={line.confidence} />
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {line.words.map((word, j) => (
                      <div key={j} className="flex items-center gap-1 bg-muted rounded px-2 py-1">
                        <span className="font-mono text-sm">{word.text}</span>
                        <ConfidenceBadge confidence={word.confidence} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        )}

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
      </div>
    </div>
  );
}
