import { useState } from "react";

interface InstructionsProps {
  onChange?: (text: string) => void;
}

export function Instructions({ onChange }: InstructionsProps) {
  const [text, setText] = useState("");

  function append(newText: string) {
    const updated = text ? `${text}\n${newText}` : newText;
    setText(updated);
    onChange?.(updated);
  }

  return (
    <section style={{ border: "1px solid #ccc", padding: "0.5rem" }}>
      <div style={{ marginBottom: "0.25rem" }}>
        <strong>Instructions</strong>
      </div>
      <div
        style={{
          minHeight: "80px",
          padding: "0.5rem",
          border: "1px solid #eee",
          background: "#fafafa",
          whiteSpace: "pre-wrap",
        }}
      >
        {text || "No instructions yet."}
      </div>
      <div style={{ fontSize: "0.9rem", marginTop: "0.35rem" }}>
        (Transcribed text will appear here.)
      </div>
    </section>
  );
}
