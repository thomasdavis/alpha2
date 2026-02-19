import React, { useState, useCallback, useMemo } from "react";
import { Box, useApp, useInput, useStdin, useStdout } from "ink";
import { TABS, type Tab, type ViewMode } from "./types.js";
import { useRuns } from "./hooks/useRuns.js";
import { scanLogs } from "./lib/scanner.js";
import { Header } from "./components/Header.js";
import { Dashboard } from "./components/Dashboard.js";
import { DetailView } from "./components/DetailView.js";
import { LogsView } from "./components/LogsView.js";
import { ModelsView } from "./components/ModelsView.js";
import { HelpBar } from "./components/HelpBar.js";

function KeyHandler({
  tab,
  setTab,
  view,
  setView,
  listLen,
  selectedIndex,
  setSelectedIndex,
  scrollOffset,
  setScrollOffset,
  refresh,
  maxScroll,
}: {
  tab: Tab;
  setTab: React.Dispatch<React.SetStateAction<Tab>>;
  view: ViewMode;
  setView: (v: ViewMode) => void;
  listLen: number;
  selectedIndex: number;
  setSelectedIndex: React.Dispatch<React.SetStateAction<number>>;
  scrollOffset: number;
  setScrollOffset: React.Dispatch<React.SetStateAction<number>>;
  refresh: () => void;
  maxScroll: number;
}) {
  const { exit } = useApp();

  const clampIndex = useCallback((idx: number) => {
    return Math.max(0, Math.min(listLen - 1, idx));
  }, [listLen]);

  useInput((input, key) => {
    if (input === "q") {
      exit();
      return;
    }

    // Tab switching with 1-4 (only from list view)
    if (view === "list") {
      const tabNum = parseInt(input, 10);
      if (tabNum >= 1 && tabNum <= 4) {
        setTab(TABS[tabNum - 1]);
        setSelectedIndex(0);
        setScrollOffset(0);
        return;
      }

      // Tab cycling with tab key
      if (key.tab) {
        setTab(prev => {
          const idx = TABS.indexOf(prev);
          return TABS[(idx + 1) % TABS.length];
        });
        setSelectedIndex(0);
        setScrollOffset(0);
        return;
      }
    }

    if (view === "list") {
      if (input === "j" || key.downArrow) {
        setSelectedIndex(i => clampIndex(i + 1));
      } else if (input === "k" || key.upArrow) {
        setSelectedIndex(i => clampIndex(i - 1));
      } else if (key.return) {
        if (listLen > 0) {
          setView("detail");
          setScrollOffset(0);
        }
      } else if (input === "r") {
        refresh();
      }
    } else if (view === "detail") {
      if (key.escape) {
        setView("list");
        setScrollOffset(0);
      } else if (input === "j" || key.downArrow) {
        setScrollOffset(o => Math.min(o + 1, maxScroll));
      } else if (input === "k" || key.upArrow) {
        setScrollOffset(o => Math.max(o - 1, 0));
      }
    }
  });

  return null;
}

export function App({ outputsDir }: { outputsDir: string }) {
  const { isRawModeSupported } = useStdin();
  const { stdout } = useStdout();
  const rows = stdout?.rows ?? 24;
  const { runs, refresh } = useRuns(outputsDir);
  const [tab, setTab] = useState<Tab>("monitor");
  const [view, setView] = useState<ViewMode>("list");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [scrollOffset, setScrollOffset] = useState(0);

  const logs = useMemo(() => tab === "logs" ? scanLogs(outputsDir) : [], [tab, outputsDir]);

  const listLen = tab === "logs" ? logs.length : runs.length;
  const maxScroll = tab === "logs" && view === "detail" && logs[selectedIndex]
    ? Math.max(0, logs[selectedIndex].lines.length - 15)
    : 15;

  const selectedRun = runs[selectedIndex] ?? null;
  const selectedLog = logs[selectedIndex] ?? null;

  return (
    <Box flexDirection="column" height={rows}>
      {isRawModeSupported && (
        <KeyHandler
          tab={tab}
          setTab={setTab}
          view={view}
          setView={setView}
          listLen={listLen}
          selectedIndex={selectedIndex}
          setSelectedIndex={setSelectedIndex}
          scrollOffset={scrollOffset}
          setScrollOffset={setScrollOffset}
          refresh={refresh}
          maxScroll={maxScroll}
        />
      )}
      <Header runs={runs} activeTab={tab} />

      {/* Content area — grows to fill available space */}
      <Box flexDirection="column" flexGrow={1}>
        {tab === "monitor" && view === "list" && (
          <Dashboard runs={runs} selectedIndex={selectedIndex} />
        )}
        {tab === "monitor" && view === "detail" && selectedRun && (
          <DetailView run={selectedRun} scrollOffset={scrollOffset} />
        )}
        {tab === "logs" && (
          <LogsView
            logs={logs}
            selectedIndex={selectedIndex}
            selectedLog={view === "detail" ? selectedLog : null}
            scrollOffset={scrollOffset}
          />
        )}
        {tab === "runs" && view === "list" && (
          <Dashboard runs={runs} selectedIndex={selectedIndex} />
        )}
        {tab === "runs" && view === "detail" && selectedRun && (
          <DetailView run={selectedRun} scrollOffset={scrollOffset} />
        )}
        {tab === "models" && (
          <ModelsView
            runs={runs}
            selectedIndex={selectedIndex}
            selectedRun={view === "detail" ? selectedRun : null}
          />
        )}
      </Box>

      {/* Help bar pinned to bottom */}
      <HelpBar tab={tab} view={view} />
    </Box>
  );
}
