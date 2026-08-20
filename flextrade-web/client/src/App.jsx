import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";

import { Badge, HealthBanner, LinkBadge } from "./components/ui";
import { probeLive, useApi } from "./lib/api";
import Forecasts from "./pages/Forecasts";
import Methodology from "./pages/Methodology";
import Operations from "./pages/Operations";
import Overview from "./pages/Overview";
import Renewables from "./pages/Renewables";
import Roadmap from "./pages/Roadmap";
import Landing from "./pages/Landing";
import Reserves from "./pages/Reserves";
import Sizing from "./pages/Sizing";
import Solutions from "./pages/Solutions";
import StateWorkspace from "./pages/StateWorkspace";
import States from "./pages/States";
import TradingDesk from "./pages/TradingDesk";

// grouped sidebar — reads as a full SaaS product, not a flat link list.
// [path, icon, label, tag?]
const NAV = [
  ["/", "◆", "Home"],
  ["group", "Monitor"],
  ["/overview", "◉", "Live Overview"],
  ["/states", "🗺", "Multi-State India", "23"],
  ["/state", "◎", "State Workspace"],
  ["/forecasts", "◍", "Forecast Lab", "4"],
  ["group", "Trade & Optimize"],
  ["/trading", "⇄", "Trading Desk"],
  ["/operations", "⚙", "Operations Suite"],
  ["/renewables", "☀", "Renewables & DSM"],
  ["group", "Plan & Grow"],
  ["/sizing", "▤", "Sizing & Bankability"],
  ["/reserves", "⚖", "Reserves & Regulation"],
  ["/solutions", "◈", "Solutions"],
  ["/roadmap", "➜", "Roadmap"],
  ["group", "Platform"],
  ["/methodology", "∑", "Methodology"],
];

export default function App() {
  const [theme, setTheme] = useState(
    () => localStorage.getItem("ft-theme") ||
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("ft-theme", theme);
  }, [theme]);

  // decide live-vs-snapshot once, before the panels start asking for data
  useEffect(() => { probeLive(); }, []);

  const { data: live } = useApi("/api/live", { refreshMs: 120_000 });
  const { data: meta } = useApi("/api/meta", { refreshMs: 300_000 });

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="bolt">⚡</div>
          <div>
            <b>FlexTrade</b>
            <small>AI Energy Trading &amp; Optimization</small>
          </div>
        </div>
        <nav className="nav">
          {NAV.map(([to, icon, label, tag], i) =>
            to === "group" ? (
              <div className="nav-group" key={`g${i}`}>{icon}</div>
            ) : (
              <NavLink key={to} to={to} end={to === "/"}
                className={({ isActive }) => (isActive ? "active" : "")}>
                <span className="icon">{icon}</span>
                {label}
                {tag ? <span className="tag">{tag}</span> : null}
              </NavLink>
            )
          )}
        </nav>
        <div className="foot">
          <div style={{ marginBottom: 8 }}>
            Live Indian grid &amp; market data<br />
            <span style={{ color: "var(--ink-2)" }}>Delhi SLDC · IEX · MERIT · Open-Meteo</span>
          </div>
          <button className="btn" style={{ width: "100%" }}
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "☀ Light mode" : "◐ Dark mode"}
          </button>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <Badge live={live?.delhi?.live} label="Delhi SLDC" asof={live?.delhi?.asof} />
          <Badge live={live?.dam?.live} label="IEX DAM" asof={live?.dam?.asof} />
          <Badge live={live?.rtm?.live} label="IEX RTM" asof={live?.rtm?.asof} />
          <Badge live={live?.northern_region?.live} label="Northern Grid" asof={live?.northern_region?.asof} />
          <Badge live={live?.bess?.live} label="BRPL BESS" />
          <LinkBadge />
          <div className="spacer" />
          <span style={{ color: "var(--muted)", fontSize: 12 }}>
            {live?.generated_at ? `data as of ${new Date(String(live.generated_at).replace(" ", "T")).toLocaleString("en-IN", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", hour12: false })}` : ""}
          </span>
        </div>
        <div className="content">
          <HealthBanner health={meta?.health} />
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/overview" element={<Overview />} />
            <Route path="/trading" element={<TradingDesk />} />
            <Route path="/operations" element={<Operations />} />
            <Route path="/sizing" element={<Sizing />} />
            <Route path="/reserves" element={<Reserves />} />
            <Route path="/solutions" element={<Solutions />} />
            <Route path="/renewables" element={<Renewables />} />
            <Route path="/states" element={<States />} />
            <Route path="/state" element={<StateWorkspace />} />
            <Route path="/forecasts" element={<Forecasts />} />
            <Route path="/methodology" element={<Methodology />} />
            <Route path="/roadmap" element={<Roadmap />} />
          </Routes>
        </div>
      </div>
    </div>
  );
}
