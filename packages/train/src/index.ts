export {
  AdamW, Lion, Adafactor, SGD, createOptimizerRegistry,
  type AdamWConfig, type LionConfig, type AdafactorConfig,
} from "./optimizers.js";
export {
  DataLoader, ShardedDataLoader, loadPretrainShardManifest, verifyPretrainShardManifest,
  loadText, loadTextSample, loadAndTokenize, loadOrCacheTokens, getSplitByte, splitText,
  SftDataLoader, loadSftExamples, buildSftExample, resolveChatSpecialIds, splitSftExamples,
  RcrUlDataLoader, loadRcrUlExamples, loadSftConversationSha256s,
  CHAT_USER_TOKEN, CHAT_ASSISTANT_TOKEN, CHAT_EOT_TOKEN,
  type DataBatch, type BatchSource, type PretrainShardManifest, type SftExample, type ChatSpecialIds,
  type SftDataLoaderOptions, type RcrUlExample,
} from "./data.js";
export {
  FileCheckpoint, buildCheckpointState, releaseCheckpointSnapshotBuffers, restoreParams,
} from "./checkpoint.js";
export {
  writeSafetensors, checkpointToLlamaStateDict, buildLlamaConfig, buildGenerationConfig,
  exportHfModel, llamaFormViolations, llamaIntermediateSize, resolveBosEosId,
  type SafeTensor,
} from "./hf_export.js";
export { sample } from "./sample.js";
export {
  assessHeliosTrainingDevice, gpuVendorName, repairTerminalValidationMetric, train,
  shouldEvaluateStep, shouldUseCoopBackwardAtStep, validateCheckpointModelCompatibility,
  type TrainerDeps, type StepMetrics, type HeliosTrainingDeviceAssessment,
  type HeliosTrainingDeviceCapabilities,
} from "./trainer.js";
export { evaluate, type EvalResult } from "./eval.js";
export {
  formatFrozenChatPrompt, fourGramRepeatRate, repeatedFourGramCompletionPositions,
  normalizedAnswerTokens, normalizedAnswer,
  answerTokenF1, answerIsContained, type FrozenChatMessage,
} from "./frozen_eval.js";
export { createRemoteReporter, type RemoteReporter, type RemoteReporterConfig, type SampleGeneration } from "./remote-reporter.js";
