/**
 * Command: alpha chatgen
 *
 * Generate synthetic historical conversations for training.
 */
import { chatgenCmd as run } from "@alpha/historic-chat-gen";

export async function chatgenCmd(args: string[]): Promise<void> {
  await run(args);
}
