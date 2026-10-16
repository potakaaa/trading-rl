import type { Bar } from "./api";
import { createBus } from "./hooks";

/** Completed 1m bars pushed over the WebSocket (`bar` messages). */
export const liveBars = createBus<Bar>();
