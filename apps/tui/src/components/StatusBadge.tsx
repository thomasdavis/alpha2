import React, { useState, useEffect } from "react";
import { Text } from "ink";
import type { RunStatus } from "../types.js";

export function StatusBadge({ status }: { status: RunStatus }) {
  const [bright, setBright] = useState(false);

  useEffect(() => {
    if (status !== "active") return;
    const id = setInterval(() => setBright(b => !b), 500);
    return () => clearInterval(id);
  }, [status]);

  if (status === "active") {
    return <Text color={bright ? "greenBright" : "green"} bold>{"●"}</Text>;
  }
  if (status === "completed") {
    return <Text color="blueBright">{"◆"}</Text>;
  }
  return <Text color="gray" dimColor>{"○"}</Text>;
}

export function StatusLabel({ status }: { status: RunStatus }) {
  if (status === "active") return <Text color="greenBright" bold>TRAINING</Text>;
  if (status === "completed") return <Text color="blueBright">COMPLETE</Text>;
  return <Text color="gray">STALE</Text>;
}
